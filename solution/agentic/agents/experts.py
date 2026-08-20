"""Domain expert agents.

Two collections are built from the same factory:

- `experts`      one agent per domain, used by the normal-priority path.
- `agent_swarm_map`  three interchangeable agents per domain, used by the
  urgent path so a burst of urgent tickets is spread across a team.
"""

from typing import Dict, List, Sequence

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
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


def expert_prompt(domain: str, label: str, has_operations: bool = False) -> str:
    """
    The system prompt for one domain expert.

    Built here rather than inline so the instructions can be read back and
    checked without standing an agent up.

    Two things it insists on. The expert must ground policy answers in the
    knowledge base, and it must end with a `Sources:` line naming the article
    ids it used — that line is what `extract_citations` reads back, so grounding
    becomes checkable rather than merely claimed.
    """
    operations = (
        " You can also perform account operations such as refunds, cancellations "
        "and plan changes. Confirm the exact reservation or plan with the customer "
        "before making a change, and never make one they did not ask for."
        if has_operations
        else ""
    )

    return (
        f"You are a {domain.title()} Support Expert at UdaHub. "
        f"{EXPERT_BRIEFS[domain]} "
        "Use your tools to look up real customer data before answering, and "
        "save any lasting customer preference with remember_customer_preference."
        f"{operations} "
        "Ground policy and how-to answers in the knowledge base: search it, "
        "answer from what you find, and do not state a policy no article "
        "supports. End every reply with a 'Sources:' line listing each article "
        "you used as 'Title (article_id)' — for example "
        "'How refunds are processed (billing_001)'. Copy the article_id exactly "
        "as it appears in the retrieved articles or in your search results, and "
        "never invent or reword one. If the answer came only from this "
        "customer's own records, write 'Sources: customer records' instead. "
        f"ALWAYS start your reply with '{label}' and be helpful."
    )


def make_expert(
    domain: str,
    label: str,
    name: str,
    extra_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
    """
    Build one ReAct expert agent for a support domain.

    `extra_tools` is how the MCP operations get attached (see `agentic.mcp`).
    With none, the expert can only read; with them, it can also act on the
    account, so the prompt tells it to confirm before it does.
    """
    return create_react_agent(
        name=name,
        model=llm,
        tools=[*agentic_tools, *extra_tools],
        prompt=SystemMessage(content=expert_prompt(domain, label, bool(extra_tools))),
    )


def build_experts(extra_tools: Sequence[BaseTool] = ()) -> Dict[str, CompiledStateGraph]:
    """One expert per domain: the normal-priority path."""
    return {
        domain: make_expert(
            domain, f"[{domain.upper()} EXPERT]", f"{domain}_expert", extra_tools
        )
        for domain in EXPERT_BRIEFS
    }


def build_agent_swarm(
    extra_tools: Sequence[BaseTool] = (),
) -> Dict[str, List[CompiledStateGraph]]:
    """Three interchangeable agents per domain: the urgent path."""
    return {
        f"{domain}_team": [
            make_expert(
                domain, f"[{domain.upper()} AGENT]", f"{domain}_agent_{i}", extra_tools
            )
            for i in range(1, 4)
        ]
        for domain in EXPERT_BRIEFS
    }


# The defaults are read-only. `agentic.mcp.enable_mcp()` rebuilds them with the
# operation tools attached.
experts = build_experts()
agent_swarm_map = build_agent_swarm()
