"""Unit + integration tests for the workflow compiler (no LLM required for unit)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "sample" / "workflow-example.json"


@pytest.fixture
def workflow_json() -> dict:
    with SAMPLE.open(encoding="utf-8") as f:
        return json.load(f)


def test_load_workflow_ir(workflow_json):
    from compiler.loader import load_workflow

    ir = load_workflow(workflow_json)
    assert ir.entry_agent_id == "root-agent"
    assert "sales-agent" in ir.agent_handoffs["root-agent"]
    assert "negotiation-agent" in ir.agent_handoffs["root-agent"]
    assert "demo-agent" in ir.agent_handoffs["root-agent"]
    assert "subscription-search" in ir.agent_tools["sales-agent"]
    assert "calculate-discount" in ir.agent_tools["negotiation-agent"]
    assert "book-demo-slot" in ir.agent_tools["demo-agent"]
    assert ir.end_capable_agents >= {
        "root-agent",
        "sales-agent",
        "negotiation-agent",
        "demo-agent",
    }
    assert ir.agent_llm["root-agent"] == "orchestrator-fast"
    assert ir.agent_prompt["sales-agent"] == "prompt-sales-agent"


def test_state_schema_fields(workflow_json):
    from compiler.loader import load_workflow
    from compiler.state import build_state_schema, collect_state_fields

    ir = load_workflow(workflow_json)
    fields = collect_state_fields(ir)
    assert "user_name" in fields
    assert "budget" in fields
    assert "cta_taken" in fields
    assert fields["budget"].type in ("number", "float")
    schema = build_state_schema(ir)
    assert "messages" in schema.__annotations__
    assert "budget" in schema.__annotations__


def test_state_collision_raises():
    from compiler.loader import WorkflowIR
    from compiler.schema import AgentResource, StateField, WorkflowExport
    from compiler.state import StateSchemaError, collect_state_fields

    agents = {
        "a": AgentResource(
            id="a",
            name="A",
            stateFields=[StateField(name="x", type="string")],
        ),
        "b": AgentResource(
            id="b",
            name="B",
            stateFields=[StateField(name="x", type="number")],
        ),
    }
    ir = WorkflowIR(
        export=WorkflowExport(),
        entry_agent_id="a",
        agents=agents,
    )
    with pytest.raises(StateSchemaError):
        collect_state_fields(ir)


def test_function_tools_execute(workflow_json):
    from compiler.loader import load_workflow
    from compiler.tools import build_tools

    ir = load_workflow(workflow_json)
    tools = build_tools(ir)
    assert "subscription-search" in tools
    out = tools["subscription-search"].invoke({"query": "crm"})
    assert "CRM Pro" in out
    out2 = tools["calculate-discount"].invoke({"budget": 200})
    assert "10%" in out2
    out3 = tools["book-demo-slot"].invoke({"preferred_slot": "Friday at 2pm"})
    assert "Friday at 2pm" in out3


def test_prompt_render(workflow_json):
    from compiler.loader import load_workflow
    from compiler.prompts import PromptEngine

    ir = load_workflow(workflow_json)
    pe = PromptEngine(ir)
    text = pe.render_for_agent(
        "root-agent",
        persona={"company_name": "Acme"},
        state={"user_context": {"user_query": "hi"}},
    )
    assert "Root Orchestrator" in text or "enterprise sales" in text.lower()


def test_handoff_tool_returns_command(workflow_json):
    from langgraph.types import Command

    from compiler.agents import make_handoff_tool
    from compiler.loader import load_workflow

    ir = load_workflow(workflow_json)
    tool = make_handoff_tool(ir.agents["sales-agent"])
    from langchain_core.messages import AIMessage, HumanMessage

    result = tool.invoke(
        {
            "type": "tool_call",
            "name": tool.name,
            "args": {
                "context": "user wants CRM",
                "state": {
                    "messages": [
                        HumanMessage(content="hi"),
                        AIMessage(content="", tool_calls=[{"name": tool.name, "args": {}, "id": "call_test"}]),
                    ]
                },
            },
            "id": "call_test",
        }
    )
    assert isinstance(result, Command)
    assert result.goto == "sales-agent"
    assert result.graph == Command.PARENT
    assert result.update["messages"][-1].tool_call_id == "call_test"


def test_submit_output_tool(workflow_json):
    from langgraph.types import Command

    from compiler.agents import make_submit_output_tool
    from compiler.loader import load_workflow

    ir = load_workflow(workflow_json)
    tool = make_submit_output_tool(ir.agents["negotiation-agent"])
    assert tool is not None
    result = tool.invoke(
        {
            "type": "tool_call",
            "name": tool.name,
            "args": {"budget": 250, "discount_percent": 20},
            "id": "call_test",
        }
    )
    assert isinstance(result, Command)
    assert result.update["budget"] == 250
    assert result.update["messages"][0].tool_call_id == "call_test"


def test_compile_graph_structure(workflow_json):
    """Compile without needing live LLM calls for graph topology checks.

    build_llms still constructs ChatModel clients (needs env only at invoke).
    """
    from compiler.graph import compile_workflow
    from compiler.loader import load_workflow

    ir = load_workflow(workflow_json)
    compiled, ir2, cp = compile_workflow(ir)
    assert compiled is not None
    assert ir2.entry_agent_id == "root-agent"
    # Graph nodes: root + 3 specialists
    spec = compiled.get_graph()
    node_ids = set(spec.nodes.keys())
    assert "root-agent" in node_ids
    assert "sales-agent" in node_ids
    assert "negotiation-agent" in node_ids
    assert "demo-agent" in node_ids


def test_e2e_handoff_routing_with_scripted_nodes(workflow_json, monkeypatch):
    """End-to-end without LLM: scripted nodes prove handoff Command routing + tools."""
    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    from compiler import graph as graph_mod
    from compiler.loader import load_workflow
    from compiler.runner import WorkflowRunner
    from compiler.tools import build_tools

    ir = load_workflow(workflow_json)
    tools = build_tools(ir)

    def scripted_builder(agent_id, ir, tool_registry, llm_registry, prompt_engine, state_schema=None):
        def _root(state):
            return Command(
                goto="sales-agent",
                update={"current_agent": "sales-agent"},
            )

        def _sales(state):
            catalog = tool_registry["subscription-search"].invoke({"query": "crm"})
            return {
                "messages": [
                    AIMessage(content=f"Here is what I found: {catalog}")
                ],
                "interested_product": "CRM Pro",
                "current_agent": "sales-agent",
            }

        def _other(state):
            return {
                "messages": [AIMessage(content=f"[{agent_id}] idle")],
                "current_agent": agent_id,
            }

        if agent_id == "root-agent":
            return _root
        if agent_id == "sales-agent":
            return _sales
        return _other

    monkeypatch.setattr(graph_mod, "build_agent_node", scripted_builder)
    compiled, ir2, cp = graph_mod.compile_workflow(ir)
    runner = WorkflowRunner(compiled, ir2, cp)
    reply = runner.invoke(
        "Tell me about CRM",
        thread_id="scripted-handoff-1",
        persona={"company_name": "Acme"},
    )
    assert "CRM Pro" in reply
    state = runner.get_state("scripted-handoff-1")
    assert state.get("current_agent") == "sales-agent"
    assert state.get("interested_product") == "CRM Pro"
    # Tool registry still works independently
    assert "CRM Pro" in tools["subscription-search"].invoke({"query": "crm"})


@pytest.mark.integration
def test_e2e_invoke(workflow_json):
    """Requires OPENAI_API_KEY. Skipped if missing."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from compiler.runner import WorkflowRunner

    runner = WorkflowRunner.from_json(workflow_json)
    reply = runner.invoke(
        "Hi, I'm Neel. Tell me about your CRM product.",
        thread_id="test-e2e-1",
        persona={
            "name": "Alex",
            "company_name": "Acme Corp",
            "personality": "friendly",
            "rules": "be concise",
            "language": "English",
        },
    )
    assert isinstance(reply, str)
    assert len(reply) > 0
