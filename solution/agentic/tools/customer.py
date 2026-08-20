"""Tools that read live customer data from the core and external databases."""

from langchain_core.tools import tool

from agentic.config import CORE_DB, EXTERNAL_DB, connect


@tool
def get_account_user_by_external_id(account_id: str, external_user_id: str) -> dict:
    """Look up a UDA-Hub user by account and external user ID."""
    conn = connect(CORE_DB)
    try:
        row = conn.execute(
            """
            SELECT user_id, account_id, external_user_id, user_name, created_at
            FROM users
            WHERE account_id = ? AND external_user_id = ?
            """,
            (account_id, external_user_id),
        ).fetchone()
        if not row:
            return {"found": False, "message": "No matching user found."}
        return {"found": True, "user": dict(row)}
    finally:
        conn.close()


@tool
def get_user_subscription(external_user_id: str) -> dict:
    """Get a CultPass user's subscription details."""
    conn = connect(EXTERNAL_DB)
    try:
        row = conn.execute(
            """
            SELECT subscription_id, user_id, status, tier,
                   monthly_quota, started_at, ended_at
            FROM subscriptions
            WHERE user_id = ?
            """,
            (external_user_id,),
        ).fetchone()
        if not row:
            return {"found": False, "message": "No subscription found."}
        return {"found": True, "subscription": dict(row)}
    finally:
        conn.close()


@tool
def get_user_reservations(external_user_id: str) -> dict:
    """Get reservations and linked experience details for a CultPass user."""
    conn = connect(EXTERNAL_DB)
    try:
        # "when" is a SQLite reserved word and has to stay quoted.
        rows = conn.execute(
            """
            SELECT r.reservation_id,
                   r.status AS reservation_status,
                   r.created_at AS reserved_at,
                   e.experience_id, e.title, e.description,
                   e.location, e."when", e.slots_available, e.is_premium
            FROM reservations r
            JOIN experiences e ON r.experience_id = e.experience_id
            WHERE r.user_id = ?
            ORDER BY e."when" ASC
            """,
            (external_user_id,),
        ).fetchall()
        return {"found": len(rows) > 0, "reservations": [dict(r) for r in rows]}
    finally:
        conn.close()
