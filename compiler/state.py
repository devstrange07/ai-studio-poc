"""Dynamic TypedDict state schema built from agents' stateFields."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, Optional, TypedDict

from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps

from compiler.loader import WorkflowIR
from compiler.schema import ParamType, StateField

# re-export for agents.build_agent_state_schema
__all__ = [
    "StateSchemaError",
    "build_state_schema",
    "collect_state_fields",
    "canonical_type",
    "python_type",
    "initial_state_defaults",
    "merge_dicts",
]


class StateSchemaError(ValueError):
    """Raised on state field name/type collisions."""


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Shallow-merge dict updates (right wins on key conflicts)."""
    out = dict(left or {})
    out.update(right or {})
    return out


# Canonical type map (text→string, list→array at compile)
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "text": str,
    "number": float,
    "float": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "list": list,
    "object": dict,
}

_CANONICAL: dict[str, str] = {
    "text": "string",
    "list": "array",
    "float": "number",
}


def canonical_type(t: ParamType | str) -> str:
    return _CANONICAL.get(t, t)


def python_type(t: ParamType | str) -> type:
    return _TYPE_MAP.get(t, str)


def collect_state_fields(ir: WorkflowIR) -> dict[str, StateField]:
    """Merge all agents' stateFields; raise on type collisions."""
    merged: dict[str, StateField] = {}
    owners: dict[str, str] = {}

    for agent_id, agent in ir.agents.items():
        for sf in agent.stateFields:
            name = sf.name
            ctype = canonical_type(sf.type)
            if name in merged:
                existing = canonical_type(merged[name].type)
                if existing != ctype:
                    raise StateSchemaError(
                        f"State field '{name}' type collision: "
                        f"{owners[name]} declares {existing}, "
                        f"{agent_id} declares {ctype}"
                    )
                # same type — keep first, optionally enrich description
                if not merged[name].description and sf.description:
                    merged[name] = sf
            else:
                merged[name] = sf
                owners[name] = agent_id
    return merged


def build_state_schema(ir: WorkflowIR) -> type:
    """
    Build a TypedDict subclass for the workflow.

    Always includes:
      - messages (Annotated with add_messages)
      - current_agent
      - user_context
      - persona
    Plus every field from agents' stateFields.
    """
    fields = collect_state_fields(ir)

    annotations: dict[str, Any] = {
        "messages": Annotated[list, add_messages],
        "remaining_steps": NotRequired[RemainingSteps],
        "current_agent": str,
        "user_context": Annotated[dict, merge_dicts],
        "persona": Annotated[dict, merge_dicts],
    }

    for name, sf in fields.items():
        py = python_type(sf.type)
        annotations[name] = Optional[py]  # noqa: UP007 — keep Optional for TypedDict

    # TypedDict created dynamically
    return TypedDict("WorkflowState", annotations, total=False)  # type: ignore[misc]


def initial_state_defaults(ir: WorkflowIR) -> dict[str, Any]:
    """Default values for a fresh thread."""
    defaults: dict[str, Any] = {
        "messages": [],
        "current_agent": ir.entry_agent_id,
        "user_context": {
            "user_query": "",
            "chat_history": [],
            "last_agent": None,
            "human_requested": False,
        },
        "persona": {},
    }
    for name, sf in collect_state_fields(ir).items():
        defaults[name] = None
    return defaults
