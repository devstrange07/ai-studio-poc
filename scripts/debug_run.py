"""One-shot debug invoke — stream updates + final state."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from compiler.runner import WorkflowRunner

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "sample" / "workflow-example.json").read_text(encoding="utf-8"))
runner = WorkflowRunner.from_json(data)

persona = {
    "name": "Alex",
    "company_name": "Acme Corp",
    "personality": "friendly",
    "rules": "be concise",
    "language": "English",
}
query = "Hi, I'm Neel. Tell me about your CRM product."
config = {"configurable": {"thread_id": "debug-run-3"}}
state_update = {
    "messages": [HumanMessage(content=query)],
    "user_context": {"user_query": query},
    "persona": persona,
    "current_agent": "root-agent",
}

print("=== STREAM updates ===")
try:
    for i, upd in enumerate(runner.graph.stream(state_update, config, stream_mode="updates")):
        print(f"UPDATE {i}:", list(upd.keys()) if isinstance(upd, dict) else type(upd))
        if isinstance(upd, dict):
            for node, payload in upd.items():
                print("  node:", node)
                if isinstance(payload, dict):
                    msgs = payload.get("messages")
                    print("  keys:", list(payload.keys()))
                    if msgs is not None:
                        print("  n_msgs:", len(msgs))
                        for m in msgs[-3:]:
                            print("   ", type(m).__name__, repr(getattr(m, "content", None))[:200], getattr(m, "tool_calls", None))
                else:
                    print("  payload type:", type(payload), repr(payload)[:300])
except Exception:
    traceback.print_exc()

print("\n=== FINAL STATE ===")
snap = runner.graph.get_state(config)
vals = snap.values if snap else {}
msgs = vals.get("messages") or []
print("current_agent", vals.get("current_agent"))
print("NUM_MSGS", len(msgs))
for m in msgs[-8:]:
    print(type(m).__name__, repr(getattr(m, "content", None))[:300], getattr(m, "tool_calls", None))
print("EXTRACTED:", repr(WorkflowRunner._extract_reply(vals)))
