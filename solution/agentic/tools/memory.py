"""Long-term memory: durable customer preferences that outlive a session.

Short-term memory is the checkpointer in `agentic.workflow`, which keeps the
conversation alive within one thread. This module is the other half: facts we
want to still know the next time the customer writes in, stored in SQLite.
"""

from datetime import datetime, timezone

from langchain_core.tools import tool

from agentic.config import CORE_DB, connect


def init_long_term_memory() -> None:
    """Create the table that holds durable customer preferences."""
    conn = connect(CORE_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_memory (
                customer_id TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (customer_id, key)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_long_term_memory()


@tool
def remember_customer_preference(customer_id: str, key: str, value: str) -> dict:
    """
    Save a durable fact or preference about a customer to long-term memory.

    Use this only for things that stay true after this ticket is closed, for
    example key="preferred_contact" value="email", or key="city" value="Lisbon".
    Do not store the current problem, ticket status, or anything sensitive such
    as card numbers.
    """
    conn = connect(CORE_DB)
    try:
        conn.execute(
            """
            INSERT INTO customer_memory (customer_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(customer_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (customer_id, key, value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"saved": True, "key": key, "value": value}
    finally:
        conn.close()


@tool
def recall_customer_preferences(customer_id: str) -> dict:
    """Read everything stored in long-term memory about a customer."""
    conn = connect(CORE_DB)
    try:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM customer_memory WHERE customer_id = ?",
            (customer_id,),
        ).fetchall()
        return {"found": len(rows) > 0, "preferences": [dict(r) for r in rows]}
    finally:
        conn.close()


def load_preferences_text(customer_id: str) -> str:
    """Same lookup as the tool above, formatted for a prompt."""
    result = recall_customer_preferences.invoke({"customer_id": customer_id})
    if not result["found"]:
        return "none on file"
    return "; ".join(f"{p['key']}={p['value']}" for p in result["preferences"])
