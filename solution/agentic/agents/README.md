# Agents

| File | What it holds |
| --- | --- |
| `experts.py` | One expert per domain, plus three interchangeable agents per domain for the urgent path. All built by `make_expert()`. |
| `teams.py` | The round-robin team graph. `pick_agent` decides whose turn it is. |
| `escalation.py` | The escalation agent — the only one allowed to hand a ticket to a human. |

Domains: `general`, `billing`, `reservation`, `technical`, `subscription`.

The rotation counter lives in the orchestrator's state (`rr_index`), not in the
team graph, so it survives from one ticket to the next.
