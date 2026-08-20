"""
Tests for the UDA-Hub support agent.

Two tiers:

    pytest test_index.py              fast, offline, no API calls
    pytest test_index.py --llm        also runs the tests that call the model

The offline tier covers the database tools, long-term memory and graph wiring.
The --llm tier covers classification and the two end-to-end branches; it is slow
(each ticket is several model round-trips) and costs API quota, so it is opt-in.
"""

import sqlite3
import uuid

import pytest

import index as ix

DEMO_CUSTOMER = "a4ab87"  # Alice Kingsley, seeded in cultpass.db
DEMO_TICKET = "233314ae-815b-46aa-8466-c4163835b224"


# The --llm flag is defined in conftest.py.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_customer():
    """A customer id that exists only for one test, so memory tests stay isolated."""
    customer_id = f"test-{uuid.uuid4().hex[:8]}"
    yield customer_id
    conn = sqlite3.connect(ix.CORE_DB)
    conn.execute("DELETE FROM customer_memory WHERE customer_id = ?", (customer_id,))
    conn.commit()
    conn.close()


@pytest.fixture
def restore_ticket_status():
    """Put ticket_metadata.status back the way we found it."""
    conn = sqlite3.connect(ix.CORE_DB)
    before = conn.execute(
        "SELECT status FROM ticket_metadata WHERE ticket_id = ?", (DEMO_TICKET,)
    ).fetchone()[0]
    conn.close()

    yield before

    conn = sqlite3.connect(ix.CORE_DB)
    conn.execute(
        "UPDATE ticket_metadata SET status = ? WHERE ticket_id = ?",
        (before, DEMO_TICKET),
    )
    conn.commit()
    conn.close()


def ticket_status() -> str:
    conn = sqlite3.connect(ix.CORE_DB)
    status = conn.execute(
        "SELECT status FROM ticket_metadata WHERE ticket_id = ?", (DEMO_TICKET,)
    ).fetchone()[0]
    conn.close()
    return status


# ---------------------------------------------------------------------------
# Offline: database tools
# ---------------------------------------------------------------------------

def test_get_ticket_details_found():
    result = ix.get_ticket_details.invoke({"ticket_id": DEMO_TICKET})
    assert result["found"] is True
    assert result["ticket"]["external_user_id"] == DEMO_CUSTOMER


def test_get_ticket_details_missing():
    result = ix.get_ticket_details.invoke({"ticket_id": "no-such-ticket"})
    assert result["found"] is False


def test_get_user_subscription():
    result = ix.get_user_subscription.invoke({"external_user_id": DEMO_CUSTOMER})
    assert result["found"] is True
    assert result["subscription"]["user_id"] == DEMO_CUSTOMER


def test_get_user_reservations():
    """Regression: `when` is a reserved word and must stay quoted in the SQL."""
    result = ix.get_user_reservations.invoke({"external_user_id": DEMO_CUSTOMER})
    assert result["found"] is True
    assert len(result["reservations"]) > 0
    # The join must bring the experience columns across.
    assert "title" in result["reservations"][0]
    assert "when" in result["reservations"][0]


def test_get_account_user_by_external_id():
    result = ix.get_account_user_by_external_id.invoke(
        {"account_id": "cultpass", "external_user_id": DEMO_CUSTOMER}
    )
    assert result["found"] is True
    assert result["user"]["user_name"] == "Alice Kingsley"


def test_get_ticket_messages():
    result = ix.get_ticket_messages.invoke({"ticket_id": DEMO_TICKET})
    assert result["found"] is True


# ---------------------------------------------------------------------------
# Offline: long-term memory
# ---------------------------------------------------------------------------

def test_memory_save_and_recall(temp_customer):
    ix.remember_customer_preference.invoke(
        {"customer_id": temp_customer, "key": "preferred_contact", "value": "email"}
    )
    result = ix.recall_customer_preferences.invoke({"customer_id": temp_customer})

    assert result["found"] is True
    assert {"preferred_contact": "email"} == {
        p["key"]: p["value"] for p in result["preferences"]
    }


def test_memory_overwrites_same_key(temp_customer):
    """A second write to the same key updates it rather than duplicating it."""
    for value in ["email", "sms"]:
        ix.remember_customer_preference.invoke(
            {"customer_id": temp_customer, "key": "preferred_contact", "value": value}
        )

    result = ix.recall_customer_preferences.invoke({"customer_id": temp_customer})
    assert len(result["preferences"]) == 1
    assert result["preferences"][0]["value"] == "sms"


def test_memory_is_per_customer(temp_customer):
    ix.remember_customer_preference.invoke(
        {"customer_id": temp_customer, "key": "city", "value": "Lisbon"}
    )
    assert ix.load_preferences_text("someone-else") == "none on file"


def test_load_preferences_text_when_empty():
    assert ix.load_preferences_text("nobody-at-all") == "none on file"


# ---------------------------------------------------------------------------
# Offline: graph wiring
# ---------------------------------------------------------------------------

def test_graph_has_expected_nodes():
    nodes = set(ix.support_graph.get_graph().nodes)
    assert {
        "resolve_context",
        "load_memory",
        "classify_urgency",
        "handle_normal",
        "handle_urgent",
        "handle_escalation",
    } <= nodes


def test_one_team_and_one_expert_per_domain():
    domains = set(ix.EXPERT_BRIEFS)
    assert set(ix.experts) == domains
    assert set(ix.agent_teams) == {f"{d}_team" for d in domains}


def test_each_team_has_three_agents():
    for team_name, pool in ix.agent_swarm_map.items():
        assert len(pool) == 3, team_name
        assert len({agent.name for agent in pool}) == 3, team_name


def test_experts_cannot_escalate():
    """Only the escalation agent may hand a ticket to a human."""
    assert "escalate_ticket" not in {t.name for t in ix.agentic_tools}


def test_pick_agent_wraps_around():
    """The rotation counter is taken modulo the team size."""
    names = ["a", "b", "c"]
    for given, expected in [(0, 0), (1, 1), (2, 2), (3, 0), (7, 1)]:
        state = {"agent_names": names, "current_agent_index": given}
        assert ix.pick_agent(state)["current_agent_index"] == expected


def test_route_round_robin_selects_named_agent():
    state = {"agent_names": ["a", "b", "c"], "current_agent_index": 1}
    assert ix.route_round_robin(state) == "b"


def test_route_by_urgency_reads_state():
    assert ix.route_by_urgency({"urgency": "escalation"}) == "escalation"


def test_build_agent_messages_includes_context_and_history():
    state = {
        "customer_id": DEMO_CUSTOMER,
        "ticket_id": DEMO_TICKET,
        "preferences": "preferred_contact=email",
        "messages": [ix.HumanMessage(content="hello")],
    }
    messages = ix.build_agent_messages(state, "Resolve this.")

    assert len(messages) == 2
    context = messages[0].content
    assert DEMO_CUSTOMER in context
    assert DEMO_TICKET in context
    assert "preferred_contact=email" in context
    assert messages[1].content == "hello"


def test_resolve_context_from_thread_id_alone():
    """chat_interface only passes messages and a thread_id; the rest is looked up."""
    state = {"messages": [ix.HumanMessage(content="hi there")]}
    resolved = ix.resolve_context(state, {"configurable": {"thread_id": DEMO_TICKET}})

    assert resolved["query"] == "hi there"
    assert resolved["ticket_id"] == DEMO_TICKET
    assert resolved["customer_id"] == DEMO_CUSTOMER


def test_resolve_context_with_unknown_thread_id():
    """A thread id that is not a ticket must not be mistaken for one."""
    state = {"messages": [ix.HumanMessage(content="hi")]}
    resolved = ix.resolve_context(state, {"configurable": {"thread_id": "1"}})

    assert resolved["ticket_id"] is None
    assert resolved["customer_id"] == "unknown"


def test_resolve_context_prefers_explicit_values():
    state = {
        "messages": [ix.HumanMessage(content="ignored")],
        "query": "the real query",
        "customer_id": DEMO_CUSTOMER,
        "ticket_id": DEMO_TICKET,
    }
    resolved = ix.resolve_context(state, {"configurable": {"thread_id": "something-else"}})

    assert resolved["query"] == "the real query"
    assert resolved["ticket_id"] == DEMO_TICKET


def test_orchestrator_is_the_support_graph():
    """03_agentic_app.ipynb imports `orchestrator`; it must be the same graph."""
    from agentic.workflow import orchestrator

    assert orchestrator is ix.support_graph


def test_escalation_without_ticket_id_does_not_call_agent():
    state = {"customer_id": DEMO_CUSTOMER, "ticket_id": None, "messages": []}
    result = ix.handle_escalation(state)

    assert result["status"] == "pending"
    assert "ticket ID" in result["response"]


# ---------------------------------------------------------------------------
# LLM tier: classification
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.parametrize(
    "query, expected",
    [
        ("I want to talk to a real human agent right now.", "escalation"),
        ("Get me a manager, this bot is useless.", "escalation"),
        ("If this is not fixed I am calling my lawyer.", "escalation"),
        ("My card was charged twice this morning, I need this fixed now!", "urgent"),
        ("I cannot log in and my event starts in an hour!", "urgent"),
        ("How do refunds work on my CultPass card?", "normal"),
        ("What experiences are available next month?", "normal"),
    ],
)
def test_urgency_classification(query, expected):
    """Asking for a human outranks urgency, even when the message sounds angry."""
    assert ix.urgency_detector.invoke({"query": query})["urgency"] == expected


@pytest.mark.llm
@pytest.mark.parametrize(
    "query, expected",
    [
        ("I was charged twice for my subscription.", "billing"),
        ("I need to cancel my booking for Saturday.", "reservation"),
        ("The website will not let me sign in.", "technical"),
        ("Can I pause my monthly plan?", "subscription"),
    ],
)
def test_domain_classification(query, expected):
    assert ix.domain_detector.invoke({"query": query})["domain"] == expected


# ---------------------------------------------------------------------------
# LLM tier: end-to-end branches
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_normal_ticket_reaches_an_expert():
    result = ix.run_support_query(
        "How do refunds work on my CultPass card?",
        DEMO_CUSTOMER,
        DEMO_TICKET,
        thread_id="test-normal",
    )
    assert result["urgency"] == "normal"
    assert result["status"] == "resolved"
    assert result["handled_by"] in {e.name for e in ix.experts.values()}


@pytest.mark.llm
def test_escalation_marks_the_ticket_escalated(restore_ticket_status):
    assert ticket_status() != "escalated"

    result = ix.run_support_query(
        "I want to talk to a real human agent right now.",
        DEMO_CUSTOMER,
        DEMO_TICKET,
        thread_id="test-escalation",
    )

    assert result["urgency"] == "escalation"
    assert result["status"] == "escalated"
    assert ticket_status() == "escalated"


@pytest.mark.llm
def test_urgent_tickets_rotate_through_the_team():
    """Four urgent billing tickets on one thread must not all land on one agent."""
    handlers = [
        ix.run_support_query(
            "My card was charged twice, fix it now!",
            DEMO_CUSTOMER,
            DEMO_TICKET,
            thread_id="test-round-robin",
        )["handled_by"]
        for _ in range(4)
    ]

    assert len(set(handlers)) == 3, f"expected all three agents, got {handlers}"
    assert handlers[0] == handlers[3], f"expected wrap-around, got {handlers}"
