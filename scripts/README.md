# Scripts

Dev helpers. Run from the **repo root** with `.env` loaded (or keys in the environment).

## Interactive chat

Multi-turn REPL that prints agent/tool flow and replies:

```bash
python scripts/chat.py
python scripts/chat.py sample/workflow-example.json --thread my-session
```

## Debug

One-shot invoke with streamed updates:

```bash
python scripts/debug_run.py
```

Single-agent / raw LLM path (useful when isolating model or prompt issues):

```bash
python scripts/debug_agent.py
```

These scripts call the public compiler APIs; they do not change compile behavior.
