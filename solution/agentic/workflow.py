"""The UDA-Hub orchestrator: the graph that ties every agent together.

    user -> resolve_context -> load_memory -> classify_urgency
                                                |-- escalation -> escalation agent
                                                |-- urgent     -> round-robin team
                                                |-- normal     -> single expert

See `design/architecture.md` for the diagram.

Memory works on two levels:
- short term, the checkpointer below, which keeps `messages` and the
  round-robin counter alive for the duration of a thread;
- long term, the `customer_memory` table, recalled by `load_memory` at the
  start of every turn and written to by the experts.
"""

from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from agentic.agents import agent_swarm_map, agent_teams, escalation_agent, experts
from agentic.tools import domain_detector, get_ticket_details, load_preferences_text, urgency_detector


class SupportState(MessagesState):
    """State carried through one support conversation."""

    query: str
    customer_id: str
    ticket_id: Optional[str]
    preferences: Optional[str]          # long-term memory, loaded per turn
    urgency: Optional[Literal["urgent", "normal", "escalation"]]
    domain: Optional[str]
    handled_by: Optional[str]
    response: Optional[str]
    status: Optional[str]
    rr_index: Optional[Dict[str, int]]  # next agent to use, per team


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_agent_messages(state: SupportState, header: str) -> List[Any]:
    """
    Give a sub-agent the ticket context plus the running conversation.

    state["messages"] is the short-term memory: every human turn and every
    answer we produced in this session, kept alive by the checkpointer.
    """
    context = SystemMessage(
        content=(
            f"Customer ID: {state['customer_id']}\n"
            f"Ticket ID: {state.get('ticket_id') or 'unknown'}\n"
            f"Known customer preferences: {state.get('preferences') or 'none on file'}\n"
            f"{header}"
        )
    )
    return [context] + state["messages"]


def as_text(content: Any) -> str:
    """Model content is sometimes a list of blocks; flatten it to a string."""
    return content if isinstance(content, str) else str(content)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def resolve_context(state: SupportState, config: RunnableConfig) -> dict:
    """
    Work out which ticket and customer this turn belongs to.

    `run_support_query` supplies these directly. A bare chat client only sends
    messages and a thread_id, so we treat the thread as the ticket and look the
    customer up from it.
    """
    configurable = (config or {}).get("configurable", {})

    query = state.get("query") or as_text(state["messages"][-1].content)
    ticket_id = state.get("ticket_id") or configurable.get("ticket_id")
    customer_id = state.get("customer_id") or configurable.get("customer_id")

    if not ticket_id:
        # A thread id that matches a real ticket is treated as that ticket.
        candidate = configurable.get("thread_id")
        if candidate and get_ticket_details.invoke({"ticket_id": candidate})["found"]:
            ticket_id = candidate

    if not customer_id and ticket_id:
        ticket = get_ticket_details.invoke({"ticket_id": ticket_id})
        if ticket["found"]:
            customer_id = ticket["ticket"]["external_user_id"]

    return {
        "query": query,
        "ticket_id": ticket_id,
        "customer_id": customer_id or "unknown",
    }


def load_memory(state: SupportState) -> dict:
    """Recall long-term memory for this customer before doing anything else."""
    return {"preferences": load_preferences_text(state["customer_id"])}


def classify_urgency(state: SupportState) -> dict:
    return {"urgency": urgency_detector.invoke({"query": state["query"]})["urgency"]}


def route_by_urgency(state: SupportState) -> str:
    return state["urgency"]


def handle_normal(state: SupportState) -> dict:
    """Supervisor 2: detect the domain, then hand off to that single expert."""
    domain = domain_detector.invoke({"query": state["query"]})["domain"]
    expert = experts[domain]

    result = expert.invoke(
        {"messages": build_agent_messages(state, "Resolve this support request.")}
    )
    answer = as_text(result["messages"][-1].content)

    return {
        "messages": [AIMessage(content=answer)],
        "domain": domain,
        "handled_by": expert.name,
        "response": answer,
        "status": "resolved",
    }


def handle_urgent(state: SupportState) -> dict:
    """Supervisor 1: detect the domain, then dispatch to that team's next agent."""
    domain = domain_detector.invoke({"query": state["query"]})["domain"]
    team_name = f"{domain}_team"
    team = agent_teams[team_name]
    agent_names = [agent.name for agent in agent_swarm_map[team_name]]

    rr_index = dict(state.get("rr_index") or {})
    result = team.invoke(
        {
            "messages": build_agent_messages(
                state, "This is an URGENT request. Resolve it quickly and concretely."
            ),
            "agent_names": agent_names,
            "current_agent_index": rr_index.get(team_name, 0),
        }
    )
    # Advance the rotation so the next urgent ticket goes to the next agent.
    rr_index[team_name] = result["current_agent_index"] + 1

    answer = as_text(result["messages"][-1].content)
    return {
        "messages": [AIMessage(content=answer)],
        "domain": domain,
        "handled_by": agent_names[result["current_agent_index"]],
        "response": answer,
        "status": "urgent_handled",
        "rr_index": rr_index,
    }


def handle_escalation(state: SupportState) -> dict:
    """Hand the ticket to a human."""
    if not state.get("ticket_id"):
        answer = (
            "I can connect you with a human support representative, but I need a "
            "valid ticket ID to mark this request as escalated."
        )
        return {
            "messages": [AIMessage(content=answer)],
            "response": answer,
            "status": "pending",
        }

    result = escalation_agent.invoke(
        {"messages": build_agent_messages(state, "Escalate this ticket to a human.")}
    )
    answer = as_text(result["messages"][-1].content)

    return {
        "messages": [AIMessage(content=answer)],
        "handled_by": "escalation_agent",
        "response": answer,
        "status": "escalated",
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(SupportState)

workflow.add_node("resolve_context", resolve_context)
workflow.add_node("load_memory", load_memory)
workflow.add_node("classify_urgency", classify_urgency)
workflow.add_node("handle_normal", handle_normal)
workflow.add_node("handle_urgent", handle_urgent)
workflow.add_node("handle_escalation", handle_escalation)

workflow.add_edge(START, "resolve_context")
workflow.add_edge("resolve_context", "load_memory")
workflow.add_edge("load_memory", "classify_urgency")
workflow.add_conditional_edges(
    "classify_urgency",
    route_by_urgency,
    {
        "normal": "handle_normal",
        "urgent": "handle_urgent",
        "escalation": "handle_escalation",
    },
)
workflow.add_edge("handle_normal", END)
workflow.add_edge("handle_urgent", END)
workflow.add_edge("handle_escalation", END)

# The checkpointer is the short-term memory: it keeps `messages` and the
# round-robin counter alive between turns of the same thread_id.
support_graph = workflow.compile(checkpointer=MemorySaver())

# `orchestrator` is the name the rest of the project imports.
orchestrator = support_graph


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_support_query(
    query: str,
    customer_id: str,
    ticket_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Run one turn of a support conversation.

    thread_id identifies the conversation; reuse it to continue the same session.
    """
    result = orchestrator.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "customer_id": customer_id,
            "ticket_id": ticket_id,
        },
        # A graph compiled with a checkpointer requires a thread_id.
        config={"configurable": {"thread_id": thread_id or ticket_id or customer_id}},
    )
    return {
        "urgency": result.get("urgency"),
        "domain": result.get("domain"),
        "handled_by": result.get("handled_by"),
        "status": result.get("status"),
        "response": result.get("response"),
    }


def chat(customer_id: str, ticket_id: Optional[str] = None) -> None:
    """Simple multi-turn console session against one thread."""
    print("UDA-Hub support. Type 'quit' to exit.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Assistant: Goodbye!")
            break
        if not user_input:
            continue
        result = run_support_query(user_input, customer_id, ticket_id)
        print(
            f"\n[{result['urgency']} | {result['domain']} | "
            f"{result['handled_by']} | {result['status']}]"
        )
        print(f"Assistant: {result['response']}\n")
