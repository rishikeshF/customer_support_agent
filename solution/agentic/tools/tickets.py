"""Tools that read ticket data and hand a ticket to a human."""

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
