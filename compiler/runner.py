"""Execution engine with multi-turn checkpointing and persona injection."""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage

from compiler.graph import compile_workflow
from compiler.loader import WorkflowIR
from compiler.state import initial_state_defaults


class WorkflowRunner:
    """Runs a compiled workflow graph with thread-scoped checkpoints."""

    def __init__(
        self,
        compiled_graph: Any,
        ir: WorkflowIR,
        checkpointer: Any | None = None,
    ):
        self.graph = compiled_graph
        self.ir = ir
        self.checkpointer = checkpointer

    @classmethod
    def from_json(
        cls,
        workflow_json: dict[str, Any],
        *,
        checkpointer: Any | None = None,
    ) -> "WorkflowRunner":
        compiled, ir, cp = compile_workflow(workflow_json, checkpointer=checkpointer)
        return cls(compiled, ir, cp)

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def invoke(
        self,
        user_query: str,
        *,
        thread_id: str = "default",
        persona: dict[str, Any] | None = None,
    ) -> str:
        """Run one conversational turn; returns the latest assistant text."""
        config = self._config(thread_id)
        persona = persona or {}

        # Seed defaults on first turn if no checkpoint yet
        existing = None
        try:
            existing = self.graph.get_state(config)
        except Exception:
            existing = None

        state_update: dict[str, Any] = {
            "messages": [HumanMessage(content=user_query)],
            "user_context": {
                "user_query": user_query,
            },
            "persona": persona,
        }

        if not existing or not existing.values:
            defaults = initial_state_defaults(self.ir)
            defaults["persona"] = persona
            defaults["user_context"] = {
                **defaults["user_context"],
                "user_query": user_query,
            }
            defaults["messages"] = [HumanMessage(content=user_query)]
            state_update = defaults
        elif not (existing.values or {}).get("current_agent"):
            state_update["current_agent"] = self.ir.entry_agent_id

        result = self.graph.invoke(state_update, config)
        return self._extract_reply(result)

    def stream(
        self,
        user_query: str,
        *,
        thread_id: str = "default",
        persona: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream graph events for one turn."""
        config = self._config(thread_id)
        persona = persona or {}
        state_update: dict[str, Any] = {
            "messages": [HumanMessage(content=user_query)],
            "user_context": {"user_query": user_query},
            "persona": persona,
        }
        yield from self.graph.stream(state_update, config, stream_mode="updates")

    @staticmethod
    def _extract_reply(result: dict[str, Any]) -> str:
        msgs = result.get("messages") or []
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                # Skip empty / pure tool-call placeholders
                if isinstance(m.content, str) and m.content.strip():
                    return m.content
                if isinstance(m.content, list):
                    parts = [
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in m.content
                    ]
                    text = "".join(parts).strip()
                    if text:
                        return text
        return ""

    def get_state(self, thread_id: str = "default") -> dict[str, Any]:
        snap = self.graph.get_state(self._config(thread_id))
        return dict(snap.values) if snap and snap.values else {}
