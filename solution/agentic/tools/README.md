# Tools

| File | Tools |
| --- | --- |
| `knowledge.py` | `search_rag_knowledge_base` — semantic search over the help articles. `assess_knowledge_confidence` — scores whether we can answer at all, which is what triggers escalation. |
| `customer.py` | User, subscription and reservation lookups |
| `tickets.py` | Ticket details and messages, `append_ticket_message`, `escalate_ticket` |
| `memory.py` | Customer preferences and resolved issues, saved and recalled |
| `classification.py` | `urgency_detector`, `domain_detector` |

Not everything here is a tool an agent can call. `append_ticket_message` and
`remember_resolved_issue` are plain functions used by the workflow, because
recording what happened is the graph's job, not a decision for the agent
answering the question.

`agentic_tools` in `__init__.py` is the set handed to the experts.
`escalate_ticket` is deliberately left out of it, so only the escalation agent
can escalate — an expert cannot escape a hard question that way.
