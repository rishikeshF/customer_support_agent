"""Tools that read and write ticket data, and hand a ticket to a human."""

import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

from agentic.config import CORE_DB, connect


@tool
def get_ticket_details(ticket_id: str) -> dict:
    """Get ticket metadata and linked user/account info."""
    conn = connect(CORE_DB)
    try:
        row = conn.execute(
            """
            SELECT t.ticket_id, t.account_id, t.user_id, t.channel, t.created_at,
                   tm.status, tm.main_issue_type, tm.tags,
                   u.external_user_id, u.user_name
            FROM tickets t
            LEFT JOIN ticket_metadata tm ON t.ticket_id = tm.ticket_id
            JOIN users u ON t.user_id = u.user_id
            WHERE t.ticket_id = ?
            """,
            (ticket_id,),
        ).fetchone()
        if not row:
            return {"found": False, "message": "Ticket not found."}
        return {"found": True, "ticket": dict(row)}
    finally:
        conn.close()


@tool
def get_ticket_messages(ticket_id: str) -> dict:
    """Get the message history for a ticket."""
    conn = connect(CORE_DB)
    try:
        rows = conn.execute(
            """
            SELECT message_id, role, content, created_at
            FROM ticket_messages
            WHERE ticket_id = ?
            ORDER BY created_at ASC
            """,
            (ticket_id,),
        ).fetchall()
        return {"found": len(rows) > 0, "messages": [dict(r) for r in rows]}
    finally:
        conn.close()


def append_ticket_message(ticket_id: str, role: str, content: str) -> dict:
    """
    Write one turn of the conversation to the ticket.

    This is what makes history survive a restart: the checkpointer keeps a live
    session going, but `ticket_messages` is the durable record a returning
    customer's next ticket can be read against.

    Not a tool. The workflow calls it after every turn, so an agent cannot
    forget to record the conversation.
    """
    if not ticket_id or not content:
        return {"saved": False, "message": "Need both a ticket id and content."}

    if role not in {"user", "agent", "ai", "system"}:
        return {"saved": False, "message": f"Unknown role '{role}'."}

    conn = connect(CORE_DB)
    try:
        message_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO ticket_messages (message_id, ticket_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                message_id,
                ticket_id,
                role,
                content,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return {"saved": True, "message_id": message_id}
    except Exception as error:  # a logging failure must not sink the answer
        return {"saved": False, "message": str(error)}
    finally:
        conn.close()


@tool
def escalate_ticket(ticket_id: str) -> dict:
    """Mark an existing support ticket as escalated."""
    conn = connect(CORE_DB)
    try:
        cursor = conn.execute(
            "UPDATE ticket_metadata SET status = ? WHERE ticket_id = ?",
            ("escalated", ticket_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return {"success": False, "message": "Ticket not found."}
        return {"success": True, "message": "Ticket escalated."}
    finally:
        conn.close()
