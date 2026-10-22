"""Agent node builder: ReAct agents, handoff tools, submit_output, agent-as-tool, guardrails."""

from __future__ import annotations

import re
from typing import Any, Callable, NotRequired, Optional, Sequence

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model
from typing import Annotated
from typing_extensions import TypedDict

from compiler.loader import WorkflowIR
from compiler.prompts import PromptEngine, render_prompt
from compiler.schema import AgentResource, Guardrail
from compiler.state import canonical_type, collect_state_fields, python_type


def build_agent_state_schema(ir: WorkflowIR) -> type:
    """AgentState + persona/user_context + all workflow stateFields (shared channels)."""
    annotations: dict[str, Any] = dict(AgentState.__annotations__)
    annotations["persona"] = NotRequired[dict]
    annotations["user_context"] = NotRequired[dict]
    annotations["current_agent"] = NotRequired[str]
    for name, sf in collect_state_fields(ir).items():
        annotations[name] = NotRequired[Optional[python_type(sf.type)]]
    return TypedDict("AgentWorkflowStateDyn", annotations)  # type: ignore[misc]


def _slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "agent"


# ── Handoff tools ─────────────────────────────────────────────────────────────

def make_handoff_tool(target: AgentResource) -> BaseTool:
    """Tool that returns Command(goto=target) for parent-graph routing."""
    tool_name = f"transfer_to_{_slug(target.name)}"
    description = (
        f"Transfer conversation to {target.name}. "
        f"{target.description or ''}".strip()
    )
    target_id = target.id

    class _HandoffArgs(BaseModel):
        context: str = Field(default="", description="Brief handoff context")
        user_language: str = Field(default="English")
        user_script: str = Field(default="Native Unicode Script")
        tool_call_id: Annotated[str, InjectedToolCallId]
        state: Annotated[dict, InjectedState]

    def _handoff(
        context: str = "",
        user_language: str = "English",
        user_script: str = "Native Unicode Script",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
        state: Annotated[dict, InjectedState] = None,  # type: ignore[assignment]
    ) -> Command:
        # Propagate full agent message history to parent. When the model fires
        # parallel tool calls (e.g. submit_output + transfer), Command(PARENT)
        # must still emit a ToolMessage for every tool_call_id.
        prior = list((state or {}).get("messages") or [])
        last_ai: AIMessage | None = None
        for m in reversed(prior):
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                last_ai = m
                break

        tool_msgs: list[ToolMessage] = []
        field_updates: dict[str, Any] = {}
        for tc in (last_ai.tool_calls if last_ai else []):
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            tc_args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)) or {}
            if tc_id == tool_call_id or tc_name == f"transfer_to_{_slug(target.name)}":
                tool_msgs.append(
                    ToolMessage(
                        content=f"Transferred to {target.name}.",
                        tool_call_id=tc_id,
                    )
                )
            elif tc_name == "submit_output":
                field_updates.update({k: v for k, v in tc_args.items() if v is not None})
                tool_msgs.append(
                    ToolMessage(content="Recorded before handoff.", tool_call_id=tc_id)
                )
            else:
                tool_msgs.append(
                    ToolMessage(
                        content="Cancelled due to handoff.",
                        tool_call_id=tc_id,
                    )
                )

        if not tool_msgs:
            tool_msgs = [
                ToolMessage(
                    content=f"Transferred to {target.name}.",
                    tool_call_id=tool_call_id,
                )
            ]

        update: dict[str, Any] = {
            "messages": prior + tool_msgs,
            "current_agent": target_id,
            "user_context": {
                "last_agent": target_id,
                "handoff_context": context,
                "user_language": user_language,
                "user_script": user_script,
            },
            **field_updates,
        }
        return Command(
            goto=target_id,
            update=update,
            graph=Command.PARENT,
        )

    _handoff.__name__ = tool_name
    _handoff.__doc__ = description
    return StructuredTool.from_function(
        func=_handoff,
        name=tool_name,
        description=description,
        args_schema=_HandoffArgs,
    )


# ── submit_output ─────────────────────────────────────────────────────────────

def make_submit_output_tool(agent: AgentResource) -> BaseTool | None:
    """Tool that writes agent stateFields into graph state via Command(update=...)."""
    fields = agent.stateFields
    if not fields:
        return None

    model_fields: dict[str, Any] = {}
    for sf in fields:
        py = python_type(canonical_type(sf.type))
        default = ... if sf.required else None
        model_fields[sf.name] = (
            Optional[py] if not sf.required else py,
            Field(default=default, description=sf.description or sf.name),
        )
    model_fields["tool_call_id"] = (
        Annotated[str, InjectedToolCallId],
        Field(default=""),
    )
    ArgsModel = create_model(f"SubmitOutput_{_slug(agent.id)}", **model_fields)

    def _submit_cmd(**kwargs: Any) -> Command:
        # Update in-subgraph only (no PARENT). Shared channels sync to parent
        # when the agent finishes. Must include ToolMessage for the tool_call_id.
        tool_call_id = kwargs.pop("tool_call_id", "") or ""
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        content = "Recorded: " + ", ".join(f"{k}={v!r}" for k, v in cleaned.items())
        return Command(
            update={
                **cleaned,
                "messages": [
                    ToolMessage(content=content or "Recorded.", tool_call_id=tool_call_id)
                ],
            }
        )

    return StructuredTool.from_function(
        func=_submit_cmd,
        name="submit_output",
        description=(
            "Capture structured output fields for this agent. "
            "Call when you have extracted the relevant data from the conversation."
        ),
        args_schema=ArgsModel,
    )


# ── Agent-as-tool wrapper ─────────────────────────────────────────────────────

def make_agent_as_tool(
    target: AgentResource,
    *,
    ir: WorkflowIR,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt_engine: PromptEngine,
) -> BaseTool:
    """Wrap an Agent-as-Tool as a synchronous StructuredTool (mini agent invoke)."""
    template = prompt_engine.template_for(target.id)
    tool_name = _slug(target.name)
    description = target.description or f"Invoke {target.name}"

    class _Args(BaseModel):
        query: str = Field(description="Request / query for the agent-as-tool")
        context: str = Field(default="", description="Optional extra context")

    def _invoke(query: str, context: str = "") -> str:
        system = render_prompt(
            template,
            persona={},
            state={"user_context": {"user_query": query}},
        )
        mini = create_agent(llm, tools=list(tools), system_prompt=system)
        result = mini.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=query if not context else f"{query}\n\nContext: {context}"
                    ),
                ]
            }
        )
        msgs = result.get("messages") or []
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                return str(m.content)
        return str(msgs[-1].content) if msgs else ""

    return StructuredTool.from_function(
        func=_invoke,
        name=tool_name,
        description=description,
        args_schema=_Args,
    )


# ── Guardrail wrappers ────────────────────────────────────────────────────────

def _run_guardrail_check(
    llm: BaseChatModel,
    prompt_text: str,
    content: str,
) -> bool:
    """Return True if content is allowed. Uses a simple LLM yes/no check."""
    if not prompt_text:
        return True
    resp = llm.invoke(
        [
            SystemMessage(content=prompt_text),
            HumanMessage(content=f"Evaluate the following content. Reply ONLY 'ALLOW' or 'BLOCK'.\n\n{content}"),
        ]
    )
    text = (resp.content or "").strip().upper()
    return "BLOCK" not in text


def wrap_with_guardrails(
    node_fn: Callable[[dict], Any],
    *,
    agent: AgentResource,
    llm: BaseChatModel,
    prompt_engine: PromptEngine,
) -> Callable[[dict], Any]:
    """Wrap an agent node with input/output guardrail checks."""
    details = agent.details
    guardrails: list[Guardrail] = list(details.guardrails) if details else []
    enabled = [g for g in guardrails if g.enabled]
    if not enabled:
        return node_fn

    input_gs = [g for g in enabled if g.type == "input"]
    output_gs = [g for g in enabled if g.type == "output"]

    def _guarded(state: dict) -> Any:
        # Input checks
        user_text = ""
        msgs = state.get("messages") or []
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                user_text = str(m.content)
                break
            if isinstance(m, dict) and m.get("role") == "user":
                user_text = str(m.get("content", ""))
                break

        for g in input_gs:
            prompt_text = ""
            if g.promptId and g.promptId in prompt_engine.prompts:
                prompt_text = prompt_engine.prompts[g.promptId].get_template()
            if prompt_text and not _run_guardrail_check(llm, prompt_text, user_text):
                return {
                    "messages": [
                        AIMessage(content="I'm unable to process that request.")
                    ]
                }

        result = node_fn(state)

        # Output checks (only for dict results with messages)
        if output_gs and isinstance(result, dict) and "messages" in result:
            out_msgs = result["messages"]
            last = out_msgs[-1] if out_msgs else None
            content = ""
            if isinstance(last, AIMessage):
                content = str(last.content)
            elif isinstance(last, BaseMessage):
                content = str(last.content)
            for g in output_gs:
                prompt_text = ""
                if g.promptId and g.promptId in prompt_engine.prompts:
                    prompt_text = prompt_engine.prompts[g.promptId].get_template()
                if prompt_text and content and not _run_guardrail_check(llm, prompt_text, content):
                    return {
                        "messages": [
                            AIMessage(content="I need to rephrase that. Could you ask again?")
                        ]
                    }
        return result

    return _guarded


# ── Agent node construction ───────────────────────────────────────────────────

def build_agent_tools(
    agent_id: str,
    ir: WorkflowIR,
    tool_registry: dict[str, BaseTool],
    llm_registry: dict[str, BaseChatModel],
    prompt_engine: PromptEngine,
) -> list[BaseTool]:
    agent = ir.agents[agent_id]
    tools: list[BaseTool] = []

    # Function / HTTP / etc.
    for tid in ir.agent_tools.get(agent_id, []):
        if tid in tool_registry:
            tools.append(tool_registry[tid])

    # Handoffs
    for target_id in ir.agent_handoffs.get(agent_id, []):
        target = ir.agents.get(target_id)
        if target:
            tools.append(make_handoff_tool(target))

    # Agent-as-tool bindings
    for aat_id in ir.agent_as_tool_bindings.get(agent_id, []):
        aat = ir.agents.get(aat_id)
        if not aat:
            continue
        from compiler.llm import resolve_agent_llm

        aat_llm = resolve_agent_llm(ir, aat_id, llm_registry)
        aat_tools = [
            tool_registry[tid]
            for tid in ir.agent_tools.get(aat_id, [])
            if tid in tool_registry
        ]
        tools.append(
            make_agent_as_tool(
                aat,
                ir=ir,
                llm=aat_llm,
                tools=aat_tools,
                prompt_engine=prompt_engine,
            )
        )

    # Also treat agents typed "Agent as Tool" that appear as handoff targets
    # from this agent? Plan says agent-as-tool edges only — skip.

    submit = make_submit_output_tool(agent)
    if submit:
        tools.append(submit)

    return tools


def build_agent_node(
    agent_id: str,
    ir: WorkflowIR,
    tool_registry: dict[str, BaseTool],
    llm_registry: dict[str, BaseChatModel],
    prompt_engine: PromptEngine,
    state_schema: type | None = None,
) -> Any:
    """
    Create a compiled agent subgraph suitable for ``graph.add_node``.

    Must be added as a subgraph (not wrapped in a manual ``.invoke``) so that
    handoff tools returning ``Command(..., graph=Command.PARENT)`` route correctly.
    """
    from compiler.llm import resolve_agent_llm

    agent = ir.agents[agent_id]
    llm = resolve_agent_llm(ir, agent_id, llm_registry)
    tools = build_agent_tools(agent_id, ir, tool_registry, llm_registry, prompt_engine)
    template = prompt_engine.template_for(agent_id)

    def _render_system(state: dict) -> str:
        persona = state.get("persona") or {}
        render_state = {
            "user_context": state.get("user_context") or {},
            **{k: v for k, v in state.items() if k not in ("messages", "persona")},
        }
        uc = dict(render_state.get("user_context") or {})
        if not uc.get("chat_history"):
            history = []
            for m in state.get("messages") or []:
                if isinstance(m, HumanMessage):
                    history.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage) and m.content:
                    history.append({"role": "assistant", "content": m.content})
            uc["chat_history"] = history
        if not uc.get("last_agent"):
            uc["last_agent"] = state.get("current_agent") or agent_id
        render_state["user_context"] = uc
        return render_prompt(template, persona=persona, state=render_state)

    @dynamic_prompt
    def _agent_prompt(request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if not isinstance(state, dict):
            try:
                state = dict(state)
            except Exception:
                state = {}
        return _render_system(state)

    agent_state = build_agent_state_schema(ir)
    react = create_agent(
        model=llm,
        tools=tools,
        middleware=[_agent_prompt],
        state_schema=agent_state,
        name=agent_id,
    )

    # Thin wrapper: stamp current_agent, then delegate to subgraph so Command works.
    # Returning the compiled graph from a lambda breaks Command.PARENT — use a
    # node that returns Command/dict from subgraph.ainvoke is wrong; instead add
    # the compiled graph directly and stamp current_agent via a side channel.
    # Pattern: prepend a small node is overkill — stamp in parent via update on entry.
    # We attach metadata and return the compiled agent for graph.add_node.
    react = react.with_config({"metadata": {"agent_id": agent_id}})
    # Store agent ref for guardrail introspection / tests
    react._workflow_agent_id = agent_id  # type: ignore[attr-defined]
    react._workflow_agent = agent  # type: ignore[attr-defined]
    react._workflow_llm = llm  # type: ignore[attr-defined]
    react._workflow_prompt_engine = prompt_engine  # type: ignore[attr-defined]
    return react
