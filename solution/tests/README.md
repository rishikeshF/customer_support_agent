# Tests

```bash
cd solution
../.venv/Scripts/python.exe -m pytest tests          # offline, ~6s
../.venv/Scripts/python.exe -m pytest tests --mcp    # also starts the MCP server
../.venv/Scripts/python.exe -m pytest tests --llm    # also calls the model
```

| File | Covers |
| --- | --- |
| `test_index.py` | database tools, long-term memory, graph wiring, round-robin arithmetic, context resolution, logging, and the end-to-end branches |
| `test_mcp.py` | the support operations, their ownership and idempotence checks, and the client that attaches them |

The default run makes no API calls and starts no subprocess.

`--mcp` adds the two tests that launch `agentic.mcp.server` as a child process
and ask it for its tools. It costs a few seconds, no quota.

`--llm` adds classification and the end-to-end branches. It is opt-in because
it is slow and spends quota — roughly 45 sequential model calls.

Tests clean up after themselves. Memory tests use throwaway customer ids, the
escalation test restores the ticket status it changes, the MCP tests run
against a copy of `cultpass.db`, and every test's log events go to a temporary
file rather than `data/logs/uda-hub.jsonl`.
