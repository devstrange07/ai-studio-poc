# Compiler

Compiles fully-resolved agentflow / canvas JSON into a LangGraph multi-agent graph.

## Public API

```python
from compiler import compile_workflow, WorkflowRunner

ir = load_workflow(data)           # via compiler.loader
compiled, ir, meta = compile_workflow(ir)
runner = WorkflowRunner.from_json(data)
reply = runner.invoke(query, thread_id=..., persona=...)
```

## Modules

| Module | Role |
|--------|------|
| `schema.py` | Pydantic models for workflow export resources |
| `loader.py` | Validate JSON → `WorkflowIR` (adjacency maps, entry agent) |
| `graph.py` | Wire agents/tools/handoffs into a compiled LangGraph |
| `agents.py` | Build agent nodes (LLM + tools + handoffs) |
| `tools.py` | Materialize tool callables from tool resources |
| `llm.py` | Build LLM clients from `llmProfiles` (OpenAI / Azure / Gemini) |
| `prompts.py` | Jinja-style prompt rendering with persona / context |
| `state.py` | Shared graph state schema |
| `runner.py` | `WorkflowRunner` — invoke / stream compiled workflows |
