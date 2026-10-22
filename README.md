# AI Studio POC

LangGraph workflow compiler: turn canvas/agentflow JSON into an executable multi-agent graph and run it from the CLI.

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
```

Set at least `OPENAI_API_KEY` for the sample workflow (provider: `openai`). Azure OpenAI and Gemini keys are optional.

## Run

```bash
# Interactive REPL (default workflow: sample/workflow-example.json)
python main.py

# Compile/validate only (no LLM calls)
python main.py --compile-only

# Single query
python main.py --query "Tell me about your product"

# Custom workflow / thread / persona
python main.py path/to/workflow.json --thread my-session --persona path/to/persona.json
```

## Layout

| Path | Description |
|------|-------------|
| [`compiler/`](compiler/README.md) | Compile JSON → LangGraph; `WorkflowRunner` |
| [`scripts/`](scripts/README.md) | Interactive chat and debug helpers |
| [`sample/`](sample/README.md) | Example workflow and canvas artifacts |
| [`tests/`](tests/README.md) | Pytest suite |

## Tests

```bash
pytest
```

See [tests/README.md](tests/README.md) for details.
