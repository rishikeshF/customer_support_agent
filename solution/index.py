"""
UDA-Hub: runnable entry point.

The implementation lives in the `agentic` package:

    agentic/config.py       paths, model clients, vector store
    agentic/tools/          knowledge base, customer data, tickets, memory,
                            classification
    agentic/agents/         domain experts, round-robin teams, escalation
    agentic/workflow.py     the orchestrator graph

This file re-exports the pieces so `python index.py` runs a demo and so the
tests have one module to import.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic.agents import (
    EXPERT_BRIEFS,
    RoundRobinState,
    agent_swarm_map,
    agent_teams,
    create_team,
    escalation_agent,
    experts,
    make_expert,
    pick_agent,
    route_round_robin,
)
from agentic.config import (
    BASE_DIR,
    CHECKPOINT_DB,
    CORE_DB,
    EXTERNAL_DB,
    KNOWLEDGE_CONFIDENCE_THRESHOLD,
    LOG_DIR,
    VECTOR_DIR,
    embeddings,
    llm,
    vectorstore,
)
from agentic.observability import LOG_FILE, log_event, log_tool_calls, read_log
from agentic.tools import (
    agentic_tools,
    append_ticket_message,
    assess_knowledge_confidence,
    domain_detector,
    escalate_ticket,
    get_account_user_by_external_id,
    get_ticket_details,
    get_ticket_messages,
    get_user_reservations,
    get_user_subscription,
    load_history_text,
    load_preferences_text,
    recall_customer_preferences,
    recall_past_issues,
    remember_customer_preference,
    remember_resolved_issue,
    search_rag_knowledge_base,
    urgency_detector,
)
from agentic.workflow import (
    SupportState,
    as_text,
    build_agent_messages,
    chat,
    check_knowledge,
    classify_urgency,
    finalize,
    handle_escalation,
    handle_normal,
    handle_urgent,
    load_memory,
    orchestrator,
    resolve_context,
    route_after_knowledge,
    route_by_urgency,
    run_support_query,
    support_graph,
)

# Re-exported on purpose: this module is the flat public surface for the demo
# and the tests, so the names below are "unused" here by design.
__all__ = [
    # config
    "BASE_DIR", "CORE_DB", "EXTERNAL_DB", "VECTOR_DIR", "LOG_DIR", "CHECKPOINT_DB",
    "KNOWLEDGE_CONFIDENCE_THRESHOLD", "llm", "embeddings", "vectorstore",
    # observability
    "log_event", "log_tool_calls", "read_log", "LOG_FILE",
    # tools
    "agentic_tools", "search_rag_knowledge_base", "assess_knowledge_confidence",
    "get_account_user_by_external_id", "get_user_subscription", "get_user_reservations",
    "get_ticket_details", "get_ticket_messages", "append_ticket_message", "escalate_ticket",
    "remember_customer_preference", "recall_customer_preferences", "load_preferences_text",
    "remember_resolved_issue", "recall_past_issues", "load_history_text",
    "urgency_detector", "domain_detector",
    # agents
    "EXPERT_BRIEFS", "make_expert", "experts", "agent_swarm_map", "agent_teams",
    "create_team", "RoundRobinState", "pick_agent", "route_round_robin", "escalation_agent",
    # workflow
    "SupportState", "support_graph", "orchestrator", "run_support_query", "chat",
    "resolve_context", "load_memory", "classify_urgency", "check_knowledge",
    "handle_normal", "handle_urgent", "handle_escalation", "finalize",
    "route_by_urgency", "route_after_knowledge", "build_agent_messages", "as_text",
]


if __name__ == "__main__":
    # Demo values come from the seeded databases.
    DEMO_CUSTOMER = "a4ab87"  # external_user_id of Alice Kingsley
    DEMO_TICKET = "233314ae-815b-46aa-8466-c4163835b224"

    for demo_query in [
        "How do refunds work on my CultPass card?",                        # normal
        "My card was charged twice this morning, I need this fixed now!",  # urgent
        "I want to talk to a real human agent right now.",                 # escalation
    ]:
        print(f"\n=== {demo_query}")
        print(run_support_query(demo_query, DEMO_CUSTOMER, DEMO_TICKET))

    # Uncomment for an interactive session:
    # chat(DEMO_CUSTOMER, DEMO_TICKET)
