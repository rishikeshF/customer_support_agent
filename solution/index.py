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
from agentic.config import BASE_DIR, CORE_DB, EXTERNAL_DB, VECTOR_DIR, embeddings, llm, vectorstore
from agentic.tools import (
    agentic_tools,
    domain_detector,
    escalate_ticket,
    get_account_user_by_external_id,
    get_ticket_details,
    get_ticket_messages,
    get_user_reservations,
    get_user_subscription,
    load_preferences_text,
    recall_customer_preferences,
    remember_customer_preference,
    search_rag_knowledge_base,
    urgency_detector,
)
from agentic.workflow import (
    SupportState,
    as_text,
    build_agent_messages,
    chat,
    classify_urgency,
    handle_escalation,
    handle_normal,
    handle_urgent,
    load_memory,
    orchestrator,
    resolve_context,
    route_by_urgency,
    run_support_query,
    support_graph,
)

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
