"""Main graph compiler — assembles parent StateGraph from WorkflowIR."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from compiler.agents import build_agent_node
from compiler.loader import WorkflowIR, load_workflow
from compiler.llm import build_llms
from compiler.prompts import PromptEngine
from compiler.state import build_state_schema
from compiler.tools import build_tools


class CompileError(ValueError):
    pass


def compile_workflow(
    workflow_json: dict[str, Any] | WorkflowIR,
    *,
    checkpointer: Any | None = None,
) -> tuple[Any, WorkflowIR, Any]:
    """
    Compile a fully-resolved workflow JSON into a LangGraph CompiledGraph.

    Returns:
        (compiled_graph, ir, checkpointer)
    """
    if isinstance(workflow_json, WorkflowIR):
        ir = workflow_json
    else:
        ir = load_workflow(workflow_json)

    state_cls = build_state_schema(ir)
    tool_registry = build_tools(ir)
    llm_registry = build_llms(ir)
    prompt_engine = PromptEngine(ir)

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph: StateGraph = StateGraph(state_cls)

    # Add agent nodes (skip Agent-as-Tool — wrapped as tools, not graph nodes)
    graph_agents: list[str] = []
    for agent_id, agent in ir.agents.items():
        if agent.type == "Agent as Tool":
            continue
        node = build_agent_node(
            agent_id,
            ir,
            tool_registry,
            llm_registry,
            prompt_engine,
            state_schema=state_cls,
        )
        graph.add_node(agent_id, node)
        graph_agents.append(agent_id)

    if ir.entry_agent_id not in graph_agents:
        raise CompileError(
            f"Entry agent '{ir.entry_agent_id}' is not a graph node "
            f"(type may be Agent as Tool or missing)"
        )

    graph.add_edge(START, ir.entry_agent_id)

    # Default terminal edges: agent → END when no handoff Command overrides
    for agent_id in ir.end_capable_agents:
        if agent_id in graph_agents:
            graph.add_edge(agent_id, END)

    # Agents that are not end-capable still need an outgoing path if they
    # only hand off (Command handles routing). Give them END as fallback too.
    for agent_id in graph_agents:
        if agent_id not in ir.end_capable_agents:
            graph.add_edge(agent_id, END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, ir, checkpointer
