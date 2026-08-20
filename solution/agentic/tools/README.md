# Tools

| File | Tools |
| --- | --- |
| `knowledge.py` | `search_rag_knowledge_base` — semantic search over the help articles |
| `customer.py` | User, subscription and reservation lookups |
| `tickets.py` | Ticket details, ticket messages, `escalate_ticket` |
| `memory.py` | Save and recall durable customer preferences |
| `classification.py` | `urgency_detector`, `domain_detector` |

`agentic_tools` in `__init__.py` is the set handed to the experts.
`escalate_ticket` is deliberately left out of it, so only the escalation agent
can escalate — an expert cannot escape a hard question that way.
