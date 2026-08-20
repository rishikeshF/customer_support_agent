"""Structured logging for agent decisions, routing, tool usage and outcomes.

Every event is one JSON object on one line in `data/logs/uda-hub.jsonl`, so the
log can be filtered with `read_log()` below, or with `grep` / `jq` from a shell.

Events emitted by the workflow:

    ticket_received     a turn started
    memory_recalled     long-term preferences loaded
    classified          urgency decided
    knowledge_checked   knowledge base confidence scored
    routed              which agent or team took the ticket
    tool_used           a tool an agent called, one event per call
    resolved            final outcome of the turn
"""

import json
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from agentic.config import LOG_DIR

LOG_FILE = LOG_DIR / "uda-hub.jsonl"


def log_event(event: str, **fields: Any) -> dict:
    """Append one structured event to the log and return it."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{k: v for k, v in fields.items() if v is not None},
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")

    return record


def read_log(
    event: Optional[str] = None,
    ticket_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Read the log back, optionally filtered. This is what makes it searchable:

        read_log(event="routed")                  every routing decision
        read_log(ticket_id="233314ae-...")        one ticket's whole history
        read_log(event="tool_used", limit=20)     the last 20 tool calls
    """
    records = [
        record
        for record in _iter_records()
        if (event is None or record.get("event") == event)
        and (ticket_id is None or record.get("ticket_id") == ticket_id)
        and (thread_id is None or record.get("thread_id") == thread_id)
    ]
    return records[-limit:] if limit else records


def _iter_records() -> Iterator[dict]:
    if not LOG_FILE.exists():
        return
    with open(LOG_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # never let a damaged line break reading the log


def log_tool_calls(messages: list, **context: Any) -> list[str]:
    """
    Record which tools an agent used, reading them back off its messages.

    Logging here rather than inside each tool keeps the tools themselves free
    of logging code, and still captures every call an agent made.
    """
    used = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if not name:
                continue
            used.append(name)
            log_event("tool_used", tool=name, **context)
    return used
