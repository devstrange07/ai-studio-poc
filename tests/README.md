# Tests

Run from the **repo root**:

```bash
pip install -r requirements.txt
pip install pytest   # if not already installed
pytest
# or
python -m pytest
```

## Markers

- `integration` — tests that call live LLM APIs (need keys in `.env`)

```bash
pytest -m "not integration"   # unit / compile-only style tests
pytest -m integration         # live API tests
```

Config lives in [`pytest.ini`](../pytest.ini) at the repo root.
