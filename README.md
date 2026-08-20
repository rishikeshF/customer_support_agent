# UDA-Hub

A multi-agent customer support system built with LangGraph. It reads a support
ticket, decides how to handle it, and either resolves it or hands it to a human.

![UDA-Hub pipeline: a ticket is classified, checked against the knowledge base, then routed to a round-robin team, a single domain expert, or the escalation agent, and finalized into the ticket record, long-term memory and the log](solution/agentic/design/pipeline_diagram.svg)

A ticket reaches a human for either of two reasons: the customer asked, or the
knowledge base cannot support an answer. The second check is what stops an
agent inventing one.

Escalation beats urgency: "this is broken, get me a manager" goes to a human,
because no automated answer will satisfy it.

Answers are grounded: experts cite the knowledge base articles they used, and
the ids are logged, so any claim can be traced back to the article behind it.

Every decision is logged as JSON to `solution/data/logs/uda-hub.jsonl`. Replay
a ticket with `read_log(ticket_id=...)`.

Account operations — refunds, cancellations, plan and account changes — sit
behind an MCP server in a separate process, and are off by default. See
[Support operations](#support-operations) below.

Design notes: [solution/agentic/design/architecture.md](solution/agentic/design/architecture.md)

## Setup

Needs **Python 3.13** (`faiss-cpu` has no 3.14 wheels).

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
```

Put your key in `solution/.env`:

```
OPENAI_API_KEY=...
```

Or `VOCAREUM_API_KEY=...` to go through the Vocareum proxy instead. Setting
`OPENAI_BASE_URL` overrides either choice.

Then run these notebooks from `solution/` to build the data:

| Notebook | Builds |
| --- | --- |
| `01_external_db_setup.ipynb` | `data/external/cultpass.db` |
| `02_core_db_setup.ipynb` | `data/core/udahub.db` |
| `02_rag_db_setup.ipynb` | `data/vectorstore/` |

## Run

```bash
cd solution
../.venv/Scripts/python.exe index.py
```

`03_agentic_app.ipynb` is the notebook entry point:

```python
from agentic.workflow import orchestrator
chat_interface(orchestrator, TICKET_ID)
```

## Test

```bash
cd solution
../.venv/Scripts/python.exe -m pytest tests          # offline, ~6s
../.venv/Scripts/python.exe -m pytest tests --mcp    # also starts the MCP server
../.venv/Scripts/python.exe -m pytest tests --llm    # also calls the model
```

The default run makes no API calls and starts no subprocess. The `--mcp` tier
costs a few seconds; the `--llm` tier is opt-in because it is slow and spends
quota.

## Support operations

The agents are read-only by default: they look things up and explain. The
operations that change an account live behind an MCP server that runs as its
own process, so the agents reach them over a boundary they cannot bypass, and
the server can be pointed at a real CultPass backend later without touching
agent code.

| Operation | What it does |
| --- | --- |
| `process_refund` | Refunds a reservation and records the refund |
| `cancel_reservation` | Cancels a booking and releases the slot |
| `set_subscription_status` | Pauses, resumes or cancels a subscription |
| `change_subscription_tier` | Moves between basic and premium |
| `set_account_blocked` | Blocks or unblocks an account |

Switch them on:

```python
from agentic.mcp import enable_mcp
enable_mcp()   # rebuilds the experts and teams with the operations attached
```

`index_mcp.ipynb` is `index.ipynb` with that switch thrown. If the server
cannot start, the system carries on read-only rather than failing.

## Layout

```
solution/
├── agentic/
│   ├── agents/      the agents
│   ├── design/      architecture notes
│   ├── mcp/         the support-operation server, and its client
│   ├── tools/       what the agents can call
│   ├── config.py    paths, model clients, vector store
│   ├── observability.py  structured logging
│   └── workflow.py  the orchestrator graph
├── data/            databases, vector index, logs
├── tests/           the test suite
├── index.py         runnable entry point
├── index.ipynb      the package flattened into a notebook (generated)
├── index_mcp.ipynb  the same notebook with the operations attached (generated)
└── utils.py         chat_interface and database helpers
```

Both notebooks are generated from the modules by `build_index_nb.py`. Edit
`agentic/` and re-run the generator, not the notebook cells.

## Built with

LangGraph, LangChain, FAISS, `gpt-4o-mini`, SQLite, FastMCP.

## License

[License](LICENSE)
