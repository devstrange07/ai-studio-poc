"""
Temporary interactive chat wrapper.

Type queries in the terminal; each turn prints the agent/tool flow and the
assistant reply. Multi-turn state is kept via a stable thread_id.

Does not modify the workflow compiler or runner — only calls public APIs.

Usage:
  python scripts/chat.py
  python scripts/chat.py sample/workflow-example.json --thread my-session
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from compiler.runner import WorkflowRunner  # noqa: E402
from compiler.state import initial_state_defaults  # noqa: E402

DEFAULT_PERSONA = {
    "name": "Alex",
    "company_name": "Acme Corp",
    "industry": "SaaS",
    "category": "Software",
    "sub_category": "CRM",
    "business_type": "B2B",
    "company_description": "Enterprise sales and CRM platform",
    "core_usps": "Lead tracking, pipeline analytics",
    "core_features": "CRM Pro, Analytics Suite, Enterprise Growth",
    "contact_info": "sales@acme.example",
    "offer_description": "20% annual discount for qualified budgets",
    "personality": "Friendly, concise, professional",
    "goal_type": "Book demos and qualify leads",
    "current_cta": "Book a demo",
    "rules": "Never invent pricing. Always use tools for facts.",
    "language": "English",
}


def _tool_label(m: ToolMessage) -> str:
    text = str(m.content or "")
    if text.startswith("Transferred to "):
        return f"handoff ({text.rstrip('.')})"
    if text.startswith("Recorded"):
        return "submit_output"
    # Prefer short content preview for function tools
    preview = text.replace("\n", " ")[:60]
    return preview or "tool"


def _ai_tool_names(m: AIMessage) -> list[str]:
    names: list[str] = []
    for tc in getattr(m, "tool_calls", None) or []:
        if isinstance(tc, dict):
            names.append(str(tc.get("name") or "tool"))
        else:
            names.append(str(getattr(tc, "name", None) or "tool"))
    return names


def format_flow(updates: list[dict[str, Any]]) -> list[str]:
    """Turn stream updates into short flow lines: agent -> tool/handoff."""
    lines: list[str] = []
    for upd in updates:
        if not isinstance(upd, dict):
            continue
        for node_id, payload in upd.items():
            if not isinstance(payload, dict):
                lines.append(f"  {node_id}")
                continue
            msgs = payload.get("messages") or []
            hops: list[str] = []
            for m in msgs:
                if isinstance(m, AIMessage):
                    hops.extend(_ai_tool_names(m))
                elif isinstance(m, ToolMessage):
                    hops.append(_tool_label(m))
            if hops:
                # Dedupe consecutive duplicates while keeping order
                seen: list[str] = []
                for h in hops:
                    if not seen or seen[-1] != h:
                        seen.append(h)
                lines.append(f"  {node_id} -> {', '.join(seen)}")
            else:
                lines.append(f"  {node_id}")
    return lines


def build_turn_input(
    runner: WorkflowRunner,
    user_query: str,
    *,
    thread_id: str,
    persona: dict[str, Any],
) -> dict[str, Any]:
    """Mirror invoke()'s first-turn seeding without changing the runner."""
    config = {"configurable": {"thread_id": thread_id}}
    existing = None
    try:
        existing = runner.graph.get_state(config)
    except Exception:
        existing = None

    if not existing or not existing.values:
        defaults = initial_state_defaults(runner.ir)
        defaults["persona"] = persona
        defaults["user_context"] = {
            **defaults["user_context"],
            "user_query": user_query,
        }
        defaults["messages"] = [HumanMessage(content=user_query)]
        return defaults

    state_update: dict[str, Any] = {
        "messages": [HumanMessage(content=user_query)],
        "user_context": {"user_query": user_query},
        "persona": persona,
    }
    if not (existing.values or {}).get("current_agent"):
        state_update["current_agent"] = runner.ir.entry_agent_id
    return state_update


def dump_state(runner: WorkflowRunner, thread_id: str) -> None:
    state = runner.get_state(thread_id)
    print(f"  current_agent: {state.get('current_agent')}")
    skip = {"messages", "persona", "user_context", "remaining_steps"}
    for key, val in sorted(state.items()):
        if key in skip:
            continue
        if val is not None:
            print(f"  {key}: {val!r}")
    uc = state.get("user_context") or {}
    if uc:
        print(f"  user_context.user_query: {uc.get('user_query')!r}")
        print(f"  user_context.last_agent: {uc.get('last_agent')!r}")


def run_turn(
    runner: WorkflowRunner,
    user_query: str,
    *,
    thread_id: str,
    persona: dict[str, Any],
) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    state_update = build_turn_input(
        runner, user_query, thread_id=thread_id, persona=persona
    )

    updates: list[dict[str, Any]] = []
    print("Flow:")
    try:
        for upd in runner.graph.stream(state_update, config, stream_mode="updates"):
            updates.append(upd)
            for line in format_flow([upd]):
                print(line)
    except Exception as e:
        print(f"  [error] {e}")
        raise

    if not updates:
        print("  (no graph updates)")

    final = runner.get_state(thread_id)
    return WorkflowRunner._extract_reply(final)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive chat over a compiled agentflow workflow"
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        default=str(ROOT / "sample" / "workflow-example.json"),
        help="Path to fully-resolved workflow JSON",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread id for multi-turn continuity (default: random)",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="Optional persona JSON path",
    )
    args = parser.parse_args(argv)

    path = Path(args.workflow)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        print(f"Workflow not found: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    persona = dict(DEFAULT_PERSONA)
    if args.persona:
        with open(args.persona, encoding="utf-8") as f:
            persona.update(json.load(f))

    print("Compiling workflow…")
    runner = WorkflowRunner.from_json(data)
    thread_id = args.thread or str(uuid.uuid4())
    name = data.get("workflow", {}).get("name") or path.name

    print(f"Workflow: {name}")
    print(f"Thread:   {thread_id}")
    print("Commands: quit | exit | q | state")
    print()

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        lower = user.lower()
        if lower in ("quit", "exit", "q"):
            break
        if lower == "state":
            dump_state(runner, thread_id)
            print()
            continue

        try:
            reply = run_turn(
                runner, user, thread_id=thread_id, persona=persona
            )
        except Exception:
            print("Assistant: (turn failed — see error above)\n")
            continue

        print(f"Assistant: {reply or '(empty)'}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
