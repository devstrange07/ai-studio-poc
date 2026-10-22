"""Debug single agent + raw LLM call."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("key prefix:", (os.getenv("OPENAI_API_KEY") or "")[:7])

from compiler.loader import load_workflow
from compiler.llm import build_llms, resolve_agent_llm
from compiler.prompts import PromptEngine
from compiler.tools import build_tools
from compiler.agents import build_agent_node
from compiler.state import build_state_schema

data = json.loads((ROOT / "sample" / "workflow-example.json").read_text(encoding="utf-8"))
ir = load_workflow(data)
llms = build_llms(ir)
tools = build_tools(ir)
pe = PromptEngine(ir)
schema = build_state_schema(ir)

# Raw LLM smoke test
llm = resolve_agent_llm(ir, "root-agent", llms)
print("LLM class:", type(llm).__name__)
try:
    resp = llm.invoke([HumanMessage(content="Say hello in one short sentence.")])
    print("RAW LLM OK:", repr(resp.content)[:300])
except Exception:
    traceback.print_exc()

# Agent node smoke test
node = build_agent_node("root-agent", ir, tools, llms, pe, state_schema=schema)
state = {
    "messages": [HumanMessage(content="Hi, tell me about CRM")],
    "persona": {"company_name": "Acme", "name": "Alex", "language": "English"},
    "user_context": {"user_query": "Hi, tell me about CRM"},
    "current_agent": "root-agent",
}
try:
    out = node(state)
    print("NODE OUT TYPE:", type(out))
    if hasattr(out, "goto"):
        print("Command goto:", out.goto, "update:", getattr(out, "update", None))
    elif isinstance(out, dict):
        msgs = out.get("messages") or []
        print("NODE MSGS", len(msgs))
        for m in msgs[-5:]:
            print(type(m).__name__, repr(getattr(m, "content", None))[:400], getattr(m, "tool_calls", None))
    else:
        print("OUT:", out)
except Exception:
    traceback.print_exc()
