# Tests

```bash
cd solution
../.venv/Scripts/python.exe -m pytest tests          # offline, ~4s
../.venv/Scripts/python.exe -m pytest tests --llm    # also calls the model
```

The default run makes no API calls: database tools, long-term memory, graph
wiring, round-robin arithmetic, and context resolution.

`--llm` adds classification and the three end-to-end branches. It is opt-in
because it is slow and spends quota — roughly 45 sequential model calls.

Tests clean up after themselves. Memory tests use throwaway customer ids, and
the escalation test restores the ticket status it changes.
