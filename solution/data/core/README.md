# Core database

`udahub.db` goes here. Built by `02_core_db_setup.ipynb`.

Tables: `accounts`, `users`, `tickets`, `ticket_metadata`, `ticket_messages`,
`knowledge`.

Two more tables are created automatically the first time the agent runs, so the
setup notebook does not need to know about them:

| Table | Holds |
| --- | --- |
| `customer_memory` | Long-term customer preferences |
| `resolved_issues` | How past tickets ended, recalled in later sessions |

`checkpoints.db` also lands here. It is the short-term memory of in-flight
conversations, written by LangGraph, and is gitignored as a runtime artifact.
