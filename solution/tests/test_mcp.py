"""
Tests for the MCP layer: the support-operation server and the client that
attaches its tools to the agents.

    pytest tests/test_mcp.py              the operation logic, offline
    pytest tests/test_mcp.py --mcp        also starts the server over stdio

Every operation test runs against a throwaway copy of cultpass.db, so nothing
here touches the seeded database.
"""

import shutil
import sqlite3

import pytest

from agentic.mcp import server


DEMO_USER = "a4ab87"  # Alice Kingsley, seeded in cultpass.db
OTHER_USER = "b1c2d3"


@pytest.fixture(autouse=True)
def sandbox_db(tmp_path, monkeypatch):
    """Point the server at a copy of the database for the length of one test."""
    copy = tmp_path / "cultpass.db"
    shutil.copy(server.EXTERNAL_DB, copy)
    monkeypatch.setattr(server, "EXTERNAL_DB", copy)
    return copy


@pytest.fixture
def db(sandbox_db):
    conn = sqlite3.connect(sandbox_db)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def reservation(db):
    """A reservation of the demo customer's that is neither cancelled nor refunded."""
    row = db.execute(
        "SELECT reservation_id, experience_id FROM reservations "
        "WHERE user_id = ? AND status NOT IN ('cancelled', 'refunded') LIMIT 1",
        (DEMO_USER,),
    ).fetchone()
    assert row is not None, "the seed data should contain an open reservation"
    return row


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

def test_refund_marks_the_reservation_and_records_the_refund(db, reservation):
    result = server.process_refund(DEMO_USER, reservation["reservation_id"], "charged twice")

    assert result["success"] is True
    assert result["refund_id"]

    status = db.execute(
        "SELECT status FROM reservations WHERE reservation_id = ?",
        (reservation["reservation_id"],),
    ).fetchone()["status"]
    assert status == "refunded"

    refund = db.execute(
        "SELECT * FROM refunds WHERE refund_id = ?", (result["refund_id"],)
    ).fetchone()
    assert refund["reason"] == "charged twice"
    assert refund["status"] == "issued"


def test_refund_is_not_issued_twice(reservation):
    server.process_refund(DEMO_USER, reservation["reservation_id"])
    again = server.process_refund(DEMO_USER, reservation["reservation_id"])

    assert again["success"] is False
    assert "already refunded" in again["message"]


def test_refund_refuses_someone_elses_reservation(db, reservation):
    result = server.process_refund(OTHER_USER, reservation["reservation_id"])

    assert result["success"] is False
    status = db.execute(
        "SELECT status FROM reservations WHERE reservation_id = ?",
        (reservation["reservation_id"],),
    ).fetchone()["status"]
    assert status != "refunded"


# ---------------------------------------------------------------------------
# Cancellations
# ---------------------------------------------------------------------------

def test_cancelling_releases_the_slot(db, reservation):
    before = db.execute(
        "SELECT slots_available FROM experiences WHERE experience_id = ?",
        (reservation["experience_id"],),
    ).fetchone()["slots_available"]

    result = server.cancel_reservation(DEMO_USER, reservation["reservation_id"])
    assert result["success"] is True

    after = db.execute(
        "SELECT slots_available FROM experiences WHERE experience_id = ?",
        (reservation["experience_id"],),
    ).fetchone()["slots_available"]
    assert after == before + 1


def test_cancelling_twice_does_not_release_two_slots(db, reservation):
    server.cancel_reservation(DEMO_USER, reservation["reservation_id"])
    before = db.execute(
        "SELECT slots_available FROM experiences WHERE experience_id = ?",
        (reservation["experience_id"],),
    ).fetchone()["slots_available"]

    again = server.cancel_reservation(DEMO_USER, reservation["reservation_id"])
    assert again["success"] is False

    after = db.execute(
        "SELECT slots_available FROM experiences WHERE experience_id = ?",
        (reservation["experience_id"],),
    ).fetchone()["slots_available"]
    assert after == before


def test_cancelling_refuses_someone_elses_reservation(reservation):
    result = server.cancel_reservation(OTHER_USER, reservation["reservation_id"])
    assert result["success"] is False


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

def test_subscription_status_changes_and_reports_the_previous_one(db):
    before = db.execute(
        "SELECT status FROM subscriptions WHERE user_id = ?", (DEMO_USER,)
    ).fetchone()["status"]
    target = "paused" if before != "paused" else "active"

    result = server.set_subscription_status(DEMO_USER, target)

    assert result["success"] is True
    assert result["previous_status"] == before
    assert (
        db.execute(
            "SELECT status FROM subscriptions WHERE user_id = ?", (DEMO_USER,)
        ).fetchone()["status"]
        == target
    )


def test_subscription_status_rejects_an_unknown_status():
    result = server.set_subscription_status(DEMO_USER, "hibernating")
    assert result["success"] is False


def test_subscription_status_is_a_no_op_when_already_set(db):
    current = db.execute(
        "SELECT status FROM subscriptions WHERE user_id = ?", (DEMO_USER,)
    ).fetchone()["status"]

    result = server.set_subscription_status(DEMO_USER, current)
    assert result["success"] is False


def test_changing_tier_moves_the_monthly_quota(db):
    before = db.execute(
        "SELECT tier FROM subscriptions WHERE user_id = ?", (DEMO_USER,)
    ).fetchone()["tier"]
    target = "premium" if before != "premium" else "basic"

    result = server.change_subscription_tier(DEMO_USER, target)

    assert result["success"] is True
    row = db.execute(
        "SELECT tier, monthly_quota FROM subscriptions WHERE user_id = ?", (DEMO_USER,)
    ).fetchone()
    assert row["tier"] == target
    assert row["monthly_quota"] == result["monthly_quota"]


def test_changing_tier_rejects_an_unknown_tier():
    result = server.change_subscription_tier(DEMO_USER, "platinum")
    assert result["success"] is False


def test_subscription_operations_need_a_known_customer():
    assert server.set_subscription_status(OTHER_USER, "paused")["success"] is False
    assert server.change_subscription_tier(OTHER_USER, "premium")["success"] is False


# ---------------------------------------------------------------------------
# Account blocking
# ---------------------------------------------------------------------------

def test_unblocking_restores_access(db):
    db.execute("UPDATE users SET is_blocked = 1 WHERE user_id = ?", (DEMO_USER,))
    db.commit()

    result = server.set_account_blocked(DEMO_USER, False, reason="payment cleared")

    assert result["success"] is True
    assert result["blocked"] is False
    assert (
        db.execute(
            "SELECT is_blocked FROM users WHERE user_id = ?", (DEMO_USER,)
        ).fetchone()["is_blocked"]
        == 0
    )


def test_blocking_an_already_blocked_account_is_a_no_op(db):
    db.execute("UPDATE users SET is_blocked = 1 WHERE user_id = ?", (DEMO_USER,))
    db.commit()

    result = server.set_account_blocked(DEMO_USER, True)
    assert result["success"] is False


def test_blocking_needs_a_known_customer():
    assert server.set_account_blocked(OTHER_USER, True)["success"] is False


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def tool_names(agent) -> set:
    """The tools a compiled ReAct agent can actually call."""
    return set(agent.get_graph().nodes["tools"].data.tools_by_name)


OPERATIONS = {
    "process_refund",
    "cancel_reservation",
    "set_subscription_status",
    "change_subscription_tier",
    "set_account_blocked",
}


def test_the_client_launches_the_server_as_a_module():
    from agentic.mcp import client

    connection = client.SERVER_CONNECTION["cultpass_operations"]
    assert connection["transport"] == "stdio"
    assert connection["args"] == ["-m", "agentic.mcp.server"]


def test_read_only_agents_have_no_operation_tools():
    """The default build must not be able to change a customer's account."""
    from agentic.agents import experts

    for expert in experts.values():
        assert not (tool_names(expert) & OPERATIONS)


@pytest.mark.mcp
def test_server_serves_every_operation_over_stdio():
    """Actually start the server in a child process and ask it for its tools."""
    from agentic.mcp.client import load_mcp_tools

    tools = load_mcp_tools()
    assert {tool.name for tool in tools} == OPERATIONS


@pytest.mark.mcp
def test_enable_mcp_attaches_the_operations_to_the_experts():
    """
    Rebuilding the registries in place is what lets the running graph pick the
    operation tools up without being recompiled.
    """
    from agentic.agents import agent_swarm_map, experts
    from agentic.mcp import enable_mcp, mcp_enabled

    attached = enable_mcp()
    if not attached:
        pytest.skip("MCP server unavailable")

    assert mcp_enabled() is True
    for expert in experts.values():
        assert OPERATIONS <= tool_names(expert)
    for pool in agent_swarm_map.values():
        assert len(pool) == 3
