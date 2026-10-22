# Sample workflows

Example assets for the compiler POC.

## Runnable workflow

| File | Use |
|------|-----|
| `workflow-example.json` | Fully resolved workflow — default input for `main.py` and the debug scripts |

```bash
python main.py sample/workflow-example.json --compile-only
```

## Canvas / source artifacts

These mirror pieces of an exported canvas (agents, prompts, flows). They are useful as reference or for building a resolved export; the compiler expects a **single** fully-resolved workflow JSON (like `workflow-example.json`).

| Pattern | Contents |
|---------|----------|
| `agent-*.json` | Agent resource definitions |
| `prompt-*.json` | System prompt resources |
| `main-flow-*.json` | Flow / graph export snippets |
