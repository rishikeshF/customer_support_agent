"""Domain expert agents.

Two collections are built from the same factory:

- `experts`      one agent per domain, used by the normal-priority path.
- `agent_swarm_map`  three interchangeable agents per domain, used by the
  urgent path so a burst of urgent tickets is spread across a team.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agentic.config import llm
from agentic.tools import agentic_tools

EXPERT_BRIEFS = {
    "general": "Handle basic product questions, account setup, and general inquiries.",
    "billing": "Handle billing issues, refunds, and failed payments.",
    "reservation": "Handle reservation inquiries, bookings, and cancellations.",
    "technical": "Handle login, website, and other technical problems.",
    "subscription": "Handle subscription plans, changes, pauses, and quotas.",
}


def make_expert(domain: str, label: str, name: str) -> CompiledStateGraph:
    """Build one ReAct expert agent for a support domain."""
    return create_react_agent(
        name=name,
        model=llm,
        tools=agentic_tools,
        prompt=SystemMessage(
            content=(
                f"You are a {domain.title()} Support Expert at UdaHub. "
                f"{EXPERT_BRIEFS[domain]} "
                "Use your tools to look up real customer data before answering, and "
                "save any lasting customer preference with remember_customer_preference. "
                f"ALWAYS start your reply with '{label}' and be helpful."
            )
        ),
    )


# One expert per domain: the normal-priority path.
experts = {
    domain: make_expert(domain, f"[{domain.upper()} EXPERT]", f"{domain}_expert")
    for domain in EXPERT_BRIEFS
}

# Three agents per domain: the urgent path.
agent_swarm_map = {
    f"{domain}_team": [
        make_expert(domain, f"[{domain.upper()} AGENT]", f"{domain}_agent_{i}")
        for i in range(1, 4)
    ]
    for domain in EXPERT_BRIEFS
}
