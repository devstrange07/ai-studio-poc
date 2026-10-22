"""CLI entry point for the LangGraph Workflow Compiler."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from compiler.runner import WorkflowRunner


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


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compile and run an agentflow workflow")
    parser.add_argument(
        "workflow",
        nargs="?",
        default="sample/workflow-example.json",
        help="Path to fully-resolved workflow JSON",
    )
    parser.add_argument("--thread", default=None, help="Thread id (default: random)")
    parser.add_argument("--persona", default=None, help="Path to persona JSON")
    parser.add_argument(
        "--query",
        default=None,
        help="Single query (non-interactive). Omit for REPL.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Validate and compile only; do not invoke the LLM",
    )
    args = parser.parse_args(argv)

    path = Path(args.workflow)
    if not path.exists():
        print(f"Workflow not found: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    persona = dict(DEFAULT_PERSONA)
    if args.persona:
        with open(args.persona, encoding="utf-8") as f:
            persona.update(json.load(f))

    if args.compile_only:
        from compiler.graph import compile_workflow
        from compiler.loader import load_workflow

        ir = load_workflow(data)
        compiled, ir, _ = compile_workflow(ir)
        print(f"Compiled OK: {ir.export.workflow.name or path.name}")
        print(f"  entry={ir.entry_agent_id}")
        print(f"  agents={list(ir.agents)}")
        print(f"  handoffs={ {k: v for k, v in ir.agent_handoffs.items() if v} }")
        print(f"  tools={ {k: v for k, v in ir.agent_tools.items() if v} }")
        print(f"  end_capable={ir.end_capable_agents}")
        _ = compiled
        return 0

    runner = WorkflowRunner.from_json(data)
    thread_id = args.thread or str(uuid.uuid4())
    print(f"Thread: {thread_id}")
    print(f"Workflow: {data.get('workflow', {}).get('name', path.name)}")
    print("Type 'quit' to exit.\n")

    if args.query:
        reply = runner.invoke(args.query, thread_id=thread_id, persona=persona)
        print(f"Assistant: {reply}")
        return 0

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            break
        reply = runner.invoke(user, thread_id=thread_id, persona=persona)
        print(f"Assistant: {reply}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
