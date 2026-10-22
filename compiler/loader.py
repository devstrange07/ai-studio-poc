"""Parse and validate workflow JSON into WorkflowIR with edge adjacency maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compiler.schema import (
    AgentResource,
    LLMProfileResource,
    PromptResource,
    ToolResource,
    WorkflowExport,
)


class WorkflowLoadError(ValueError):
    """Raised when the workflow JSON is invalid or incomplete."""


@dataclass
class WorkflowIR:
    """Internal representation with pre-computed adjacency maps."""

    export: WorkflowExport
    entry_agent_id: str
    end_capable_agents: set[str] = field(default_factory=set)
    agent_tools: dict[str, list[str]] = field(default_factory=dict)
    agent_handoffs: dict[str, list[str]] = field(default_factory=dict)
    agent_llm: dict[str, str] = field(default_factory=dict)
    agent_prompt: dict[str, str] = field(default_factory=dict)
    agent_as_tool_bindings: dict[str, list[str]] = field(default_factory=dict)
    agents: dict[str, AgentResource] = field(default_factory=dict)
    tools: dict[str, ToolResource] = field(default_factory=dict)
    prompts: dict[str, PromptResource] = field(default_factory=dict)
    llm_profiles: dict[str, LLMProfileResource] = field(default_factory=dict)
    # node id -> resource id lookups
    node_to_agent: dict[str, str] = field(default_factory=dict)
    node_to_tool: dict[str, str] = field(default_factory=dict)
    trigger_ids: set[str] = field(default_factory=set)
    end_ids: set[str] = field(default_factory=set)


def _resource_id_for_agent_node(node_id: str, agent_id: str | None) -> str:
    return agent_id or node_id


def load_workflow(data: dict[str, Any] | WorkflowExport) -> WorkflowIR:
    """Validate JSON (or already-parsed model) and build WorkflowIR."""
    export = (
        data
        if isinstance(data, WorkflowExport)
        else WorkflowExport.model_validate(data)
    )

    agents = {a.id: a for a in export.resources.agents}
    tools = {t.id: t for t in export.resources.tools}
    prompts = {p.id: p for p in export.resources.prompts}
    llm_profiles = {p.id: p for p in export.resources.llmProfiles}

    node_to_agent: dict[str, str] = {}
    node_to_tool: dict[str, str] = {}
    trigger_ids: set[str] = set()
    end_ids: set[str] = set()

    for node in export.nodes:
        if node.type == "trigger":
            trigger_ids.add(node.id)
        elif node.type == "end":
            end_ids.add(node.id)
        elif node.type == "agent":
            aid = _resource_id_for_agent_node(node.id, node.data.agentId)
            node_to_agent[node.id] = aid
            # Seed agent from node if resources.agents is empty / missing
            if aid not in agents:
                agents[aid] = AgentResource(
                    id=aid,
                    name=node.data.label or aid,
                    type="Sub Agent",
                    description=node.data.description or "",
                    llmProfile=node.data.llmProfile,
                    status=node.data.status or "Active",
                )
            elif node.data.llmProfile and not agents[aid].llmProfile:
                agents[aid].llmProfile = node.data.llmProfile
        elif node.type == "tool":
            tid = node.data.toolId or node.id
            node_to_tool[node.id] = tid
            if tid not in tools:
                tools[tid] = ToolResource(
                    id=tid,
                    name=node.data.label or tid,
                    description=node.data.description or "",
                    kind=node.data.kind,
                )

    # Classify edges
    entry_agent_id: str | None = None
    end_capable: set[str] = set()
    agent_tools: dict[str, list[str]] = {aid: [] for aid in agents}
    agent_handoffs: dict[str, list[str]] = {aid: [] for aid in agents}
    agent_llm: dict[str, str] = {}
    agent_prompt: dict[str, str] = {}
    agent_as_tool_bindings: dict[str, list[str]] = {aid: [] for aid in agents}

    def _agent_for_node(nid: str) -> str | None:
        return node_to_agent.get(nid)

    for edge in export.edges:
        kind = edge.data.kind
        src, tgt = edge.source, edge.target

        if kind == "flow":
            if src in trigger_ids and tgt in node_to_agent:
                if entry_agent_id is not None and entry_agent_id != node_to_agent[tgt]:
                    raise WorkflowLoadError(
                        f"Multiple entry agents: {entry_agent_id} and {node_to_agent[tgt]}"
                    )
                entry_agent_id = node_to_agent[tgt]
            elif src in node_to_agent and tgt in end_ids:
                end_capable.add(node_to_agent[src])

        elif kind == "handoff":
            src_a = _agent_for_node(src)
            tgt_a = _agent_for_node(tgt)
            if src_a and tgt_a:
                if tgt_a not in agent_handoffs[src_a]:
                    agent_handoffs[src_a].append(tgt_a)

        elif kind == "uses-tool":
            src_a = _agent_for_node(src)
            tid = node_to_tool.get(tgt)
            if src_a and tid and tid not in agent_tools[src_a]:
                agent_tools[src_a].append(tid)

        elif kind == "uses-llm":
            src_a = _agent_for_node(src)
            # target may be an llm-profile node; fall back to target id as profile id
            profile_id = tgt
            for n in export.nodes:
                if n.id == tgt and n.type == "llm-profile":
                    profile_id = n.data.llmProfile or n.id
                    break
            if src_a:
                agent_llm[src_a] = profile_id

        elif kind == "uses-prompt":
            src_a = _agent_for_node(src)
            if src_a:
                agent_prompt[src_a] = tgt

        elif kind == "agent-as-tool":
            src_a = _agent_for_node(src)
            tgt_a = _agent_for_node(tgt)
            if src_a and tgt_a and tgt_a not in agent_as_tool_bindings[src_a]:
                agent_as_tool_bindings[src_a].append(tgt_a)

    # Fallback LLM / prompt from agent resource fields
    for aid, agent in agents.items():
        if aid not in agent_llm and agent.llmProfile:
            agent_llm[aid] = agent.llmProfile
        if aid not in agent_prompt and agent.promptId:
            agent_prompt[aid] = agent.promptId
        # Merge tools listed on agent resource with edge-derived tools
        for tid in agent.tools:
            if tid not in agent_tools[aid]:
                agent_tools[aid].append(tid)

    if not entry_agent_id:
        # Prefer Root Agent if present
        roots = [a.id for a in agents.values() if a.type == "Root Agent"]
        if len(roots) == 1:
            entry_agent_id = roots[0]
        elif agents:
            entry_agent_id = next(iter(agents))
        else:
            raise WorkflowLoadError("No entry agent: missing trigger→agent flow edge")

    if not end_capable:
        # All non agent-as-tool agents can terminate
        end_capable = {
            a.id for a in agents.values() if a.type != "Agent as Tool"
        }

    return WorkflowIR(
        export=export,
        entry_agent_id=entry_agent_id,
        end_capable_agents=end_capable,
        agent_tools=agent_tools,
        agent_handoffs=agent_handoffs,
        agent_llm=agent_llm,
        agent_prompt=agent_prompt,
        agent_as_tool_bindings=agent_as_tool_bindings,
        agents=agents,
        tools=tools,
        prompts=prompts,
        llm_profiles=llm_profiles,
        node_to_agent=node_to_agent,
        node_to_tool=node_to_tool,
        trigger_ids=trigger_ids,
        end_ids=end_ids,
    )
