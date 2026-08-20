"""A FastMCP server exposing CultPass support operations.

These are the *write* operations — refunds, cancellations, subscription and
account changes. They live behind MCP rather than in `agentic/tools/` because
they change a customer's account: putting them in a separate process means the
support agent talks to them over a protocol boundary it cannot bypass, and the
same server can be pointed at a real CultPass backend later without touching
the agent code.

Run it directly to serve over stdio:

    python -m agentic.mcp.server

Read operations stay in `agentic/tools/`, which the agents call in-process.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# Resolved here rather than imported from agentic.config: this runs as its own
# process and must not drag in the model clients or the vector store.
BASE_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_DB = BASE_DIR / "data" / "external" / "cultpass.db"

mcp = FastMCP("cultpass-operations")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(EXTERNAL_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _init_refunds_table() -> None:
    """CultPass has no refunds table of its own, so we own one."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refunds (
                refund_id      TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                reservation_id TEXT NOT NULL,
                reason         TEXT,
                status         TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


_init_refunds_table()


def _owned_reservation(conn: sqlite3.Connection, user_id: str, reservation_id: str):
    """A customer may only ever act on their own reservation."""
    return conn.execute(
        """
        SELECT r.reservation_id, r.status, r.experience_id, e.title
        FROM reservations r
        JOIN experiences e ON r.experience_id = e.experience_id
        WHERE r.reservation_id = ? AND r.user_id = ?
        """,
        (reservation_id, user_id),
    ).fetchone()


@mcp.tool
def process_refund(user_id: str, reservation_id: str, reason: str = "") -> dict:
    """
    Refund a reservation and mark it refunded.

    Only refunds a reservation that belongs to this customer and has not
    already been refunded. Returns the refund id on success.

    Args:
        user_id: The CultPass user id (external_user_id).
        reservation_id: The reservation to refund.
        reason: Short explanation, stored with the refund record.
    """
    conn = _connect()
    try:
        reservation = _owned_reservation(conn, user_id, reservation_id)
        if not reservation:
            return {"success": False, "message": "No such reservation for this customer."}
        if reservation["status"] == "refunded":
            return {"success": False, "message": "This reservation was already refunded."}

        refund_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO refunds (refund_id, user_id, reservation_id, reason, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (refund_id, user_id, reservation_id, reason, "issued", _now()),
        )
        conn.execute(
            "UPDATE reservations SET status = ?, updated_at = ? WHERE reservation_id = ?",
            ("refunded", _now(), reservation_id),
        )
        conn.commit()

        return {
            "success": True,
            "refund_id": refund_id,
            "experience": reservation["title"],
            "message": "Refund issued. It reaches the original payment method in 5 to 10 business days.",
        }
    finally:
        conn.close()


@mcp.tool
def cancel_reservation(user_id: str, reservation_id: str) -> dict:
    """
    Cancel a customer's reservation and return the slot to the experience.

    Args:
        user_id: The CultPass user id (external_user_id).
        reservation_id: The reservation to cancel.
    """
    conn = _connect()
    try:
        reservation = _owned_reservation(conn, user_id, reservation_id)
        if not reservation:
            return {"success": False, "message": "No such reservation for this customer."}
        if reservation["status"] in {"cancelled", "refunded"}:
            return {
                "success": False,
                "message": f"This reservation is already {reservation['status']}.",
            }

        conn.execute(
            "UPDATE reservations SET status = ?, updated_at = ? WHERE reservation_id = ?",
            ("cancelled", _now(), reservation_id),
        )
        conn.execute(
            "UPDATE experiences SET slots_available = slots_available + 1 WHERE experience_id = ?",
            (reservation["experience_id"],),
        )
        conn.commit()

        return {
            "success": True,
            "experience": reservation["title"],
            "message": "Reservation cancelled and the slot released.",
        }
    finally:
        conn.close()


@mcp.tool
def set_subscription_status(user_id: str, status: str) -> dict:
    """
    Pause, resume or cancel a customer's subscription.

    Args:
        user_id: The CultPass user id (external_user_id).
        status: One of "active", "paused", "cancelled".
    """
    allowed = {"active", "paused", "cancelled"}
    if status not in allowed:
        return {"success": False, "message": f"Status must be one of {sorted(allowed)}."}

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT subscription_id, status FROM subscriptions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"success": False, "message": "No subscription found for this customer."}
        if row["status"] == status:
            return {"success": False, "message": f"The subscription is already {status}."}

        conn.execute(
            "UPDATE subscriptions SET status = ?, updated_at = ? WHERE user_id = ?",
            (status, _now(), user_id),
        )
        conn.commit()
        return {
            "success": True,
            "previous_status": row["status"],
            "status": status,
            "message": f"Subscription is now {status}.",
        }
    finally:
        conn.close()


@mcp.tool
def change_subscription_tier(user_id: str, tier: str) -> dict:
    """
    Move a customer between the basic and premium plans.

    Args:
        user_id: The CultPass user id (external_user_id).
        tier: Either "basic" or "premium".
    """
    quotas = {"basic": 5, "premium": 10}
    if tier not in quotas:
        return {"success": False, "message": 'Tier must be "basic" or "premium".'}

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT tier FROM subscriptions WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return {"success": False, "message": "No subscription found for this customer."}
        if row["tier"] == tier:
            return {"success": False, "message": f"The customer is already on {tier}."}

        conn.execute(
            """
            UPDATE subscriptions
            SET tier = ?, monthly_quota = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (tier, quotas[tier], _now(), user_id),
        )
        conn.commit()
        return {
            "success": True,
            "previous_tier": row["tier"],
            "tier": tier,
            "monthly_quota": quotas[tier],
            "message": f"Plan changed to {tier}.",
        }
    finally:
        conn.close()


@mcp.tool
def set_account_blocked(user_id: str, blocked: bool, reason: Optional[str] = None) -> dict:
    """
    Block or unblock a customer's account.

    Unblocking is the common support case: a customer locked out after a failed
    payment or a security hold.

    Args:
        user_id: The CultPass user id (external_user_id).
        blocked: True to block the account, False to restore access.
        reason: Optional note explaining the change.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT is_blocked, full_name FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return {"success": False, "message": "No such customer."}
        if bool(row["is_blocked"]) == blocked:
            state = "blocked" if blocked else "active"
            return {"success": False, "message": f"The account is already {state}."}

        conn.execute(
            "UPDATE users SET is_blocked = ?, updated_at = ? WHERE user_id = ?",
            (1 if blocked else 0, _now(), user_id),
        )
        conn.commit()
        return {
            "success": True,
            "customer": row["full_name"],
            "blocked": blocked,
            "reason": reason,
            "message": "Account blocked." if blocked else "Account unblocked, access restored.",
        }
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")
