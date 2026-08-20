# UDA-Hub

A multi-agent customer support system built with LangGraph. It reads a support
ticket, decides how to handle it, and either resolves it or hands it to a human.

```
user -> resolve_context -> load_memory -> classify_urgency
                                            |-- escalation ------------.
                                            '-- check_knowledge --.    |
                                                   |              |    |
                                        confident  |              '----+-> escalation
                                                   v                   |
                                      urgent -> round-robin team       |
                                      normal -> single expert          |
                                                   |                   |
                                                   '-------> finalize <'
```

A ticket reaches a human for either of two reasons: the customer asked, or the
knowledge base cannot support an answer. The second check is what stops an
agent inventing one.

Escalation beats urgency: "this is broken, get me a manager" goes to a human,
because no automated answer will satisfy it.

Every decision is logged as JSON to `solution/data/logs/uda-hub.jsonl`. Replay
a ticket with `read_log(ticket_id=...)`.

Design notes: [solution/agentic/design/architecture.md](solution/agentic/design/architecture.md)

## Setup

Needs **Python 3.13** (`faiss-cpu` has no 3.14 wheels).

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
```

Put your key in `solution/.env`:

```
VOCAREUM_API_KEY=...
```

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
../.venv/Scripts/python.exe -m pytest tests          # offline, ~4s
../.venv/Scripts/python.exe -m pytest tests --llm    # also calls the model
```

The default run makes no API calls. The `--llm` tier is opt-in because it is
slow and spends quota.

## Layout

```
solution/
├── agentic/
│   ├── agents/      the agents
│   ├── design/      architecture notes
│   ├── tools/       what the agents can call
│   ├── config.py    paths, model clients, vector store
│   ├── observability.py  structured logging
│   └── workflow.py  the orchestrator graph
├── data/            databases, vector index, logs
├── tests/           the test suite
├── index.py         runnable entry point
├── index.ipynb      the package flattened into a notebook (generated)
└── utils.py         chat_interface and database helpers
```

`index.ipynb` is generated from the modules by `build_index_nb.py`. Edit
`agentic/` and re-run the generator, not the notebook cells.

## Built with

LangGraph, LangChain, FAISS, `gpt-4o-mini`, SQLite.

## License

[License](LICENSE)
