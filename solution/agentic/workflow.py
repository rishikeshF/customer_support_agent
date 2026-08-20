"""The UDA-Hub orchestrator: the graph that ties every agent together.

    user -> resolve_context -> load_memory -> classify_urgency
                                                |-- escalation ------------.
                                                |                          |
                                                '-- check_knowledge --.    |
                                                       |              |    |
                                            confident  |              '----+-> escalation agent
                                                       |                   |   (customer asked, or
                                                       v                   |    nothing to answer with)
                                          urgent -> round-robin team       |
                                          normal -> single expert          |
                                                       |                   |
                                                       '-------> finalize <'

See `design/architecture.md` for the diagram and the reasoning.

A ticket escalates for either of two reasons: the customer asked for a person,
or the knowledge base and the customer's own records between them cannot
support an answer. The second check is what stops an agent inventing one.

Memory works on three levels:
- short term, the checkpointer, which keeps one thread's messages alive;
- the ticket record, written every turn, which survives a restart;
- long term, preferences and resolved issues, which cross sessions.
"""

from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from agentic.agents import agent_swarm_map, agent_teams, escalation_agent, experts
from agentic.config import KNOWLEDGE_CONFIDENCE_THRESHOLD, build_checkpointer
from agentic.observability import log_event, log_tool_calls
from agentic.tools import (
    append_ticket_message,
    assess_knowledge_confidence,
    domain_detector,
    extract_citations,
    get_ticket_details,
    load_history_text,
    load_preferences_text,
    remember_resolved_issue,
    urgency_detector,
)


class SupportState(MessagesState):
    """State carried through one support conversation."""

    query: str
    customer_id: str
    ticket_id: Optional[str]
    ticket_metadata: Optional[Dict[str, Any]]   # channel, tags, status, age
    preferences: Optional[str]                  # long-term memory
    history: Optional[str]                      # previously resolved issues
    urgency: Optional[Literal["urgent", "normal", "escalation"]]
    domain: Optional[str]
    knowledge_confidence: Optional[float]
    knowledge_reason: Optional[str]
    knowledge_articles: Optional[List[Dict[str, Any]]]  # what check_knowledge retrieved
    citations: Optional[List[str]]              # article ids the answer cited
    escalation_reason: Optional[str]
    handled_by: Optional[str]
    tools_used: Optional[List[str]]
    response: Optional[str]
    status: Optional[str]
    rr_index: Optional[Dict[str, int]]          # next agent to use, per team


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_articles(articles: Optional[List[Dict[str, Any]]]) -> str:
    """
    List the articles `check_knowledge` already retrieved, ids included.

    The expert gets these before it starts, so it can cite an article id without
    having to search again for something we have already looked up.
    """
    if not articles:
        return "  none"
    return "\n".join(f"  - {a['title']} ({a['article_id']})" for a in articles)


def build_agent_messages(state: SupportState, header: str) -> List[Any]:
    """
    Give a sub-agent the ticket context plus the running conversation.

    state["messages"] is the short-term memory: every human turn and every
    answer we produced in this session, kept alive by the checkpointer.
    """
    metadata = state.get("ticket_metadata") or {}
    context = SystemMessage(
        content=(
            f"Customer ID: {state['customer_id']}\n"
            f"Ticket ID: {state.get('ticket_id') or 'unknown'}\n"
            f"Channel: {metadata.get('channel') or 'unknown'}\n"
            f"Ticket tags: {metadata.get('tags') or 'none'}\n"
            f"Known customer preferences: {state.get('preferences') or 'none on file'}\n"
            f"Previously resolved issues: {state.get('history') or 'no previous tickets'}\n"
            f"Knowledge base articles already retrieved for this question:\n"
            f"{render_articles(state.get('knowledge_articles'))}\n"
            f"{header}"
        )
    )
    return [context] + state["messages"]


def as_text(content: Any) -> str:
    """Model content is sometimes a list of blocks; flatten it to a string."""
    return content if isinstance(content, str) else str(content)


def log_context(state: SupportState) -> dict:
    """The fields every log line carries, so the log can be filtered by ticket."""
    return {
        "ticket_id": state.get("ticket_id"),
        "customer_id": state.get("customer_id"),
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def resolve_context(state: SupportState, config: RunnableConfig) -> dict:
    """
    Work out which ticket and customer this turn belongs to, and load the
    ticket's metadata so routing can use more than the message text.

    `run_support_query` supplies the ids directly. A bare chat client only sends
    messages and a thread_id, so we treat the thread as the ticket and look the
    customer up from it.
    """
    configurable = (config or {}).get("configurable", {})

    query = state.get("query") or as_text(state["messages"][-1].content)
    ticket_id = state.get("ticket_id") or configurable.get("ticket_id")
    customer_id = state.get("customer_id") or configurable.get("customer_id")

    if not ticket_id:
        candidate = configurable.get("thread_id")
        if candidate and get_ticket_details.invoke({"ticket_id": candidate})["found"]:
            ticket_id = candidate

    metadata = {}
    if ticket_id:
        ticket = get_ticket_details.invoke({"ticket_id": ticket_id})
        if ticket["found"]:
            metadata = {
                "channel": ticket["ticket"].get("channel"),
                "tags": ticket["ticket"].get("tags"),
                "status": ticket["ticket"].get("status"),
                "created_at": ticket["ticket"].get("created_at"),
                "main_issue_type": ticket["ticket"].get("main_issue_type"),
            }
            customer_id = customer_id or ticket["ticket"]["external_user_id"]

    customer_id = customer_id or "unknown"

    log_event(
        "ticket_received",
        ticket_id=ticket_id,
        customer_id=customer_id,
        thread_id=configurable.get("thread_id"),
        channel=metadata.get("channel"),
        query=query,
    )

    return {
        "query": query,
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "ticket_metadata": metadata,
    }


def load_memory(state: SupportState) -> dict:
    """Recall long-term memory for this customer before doing anything else."""
    preferences = load_preferences_text(state["customer_id"])
    history = load_history_text(state["customer_id"])

    log_event(
        "memory_recalled",
        preferences=preferences,
        past_issues=history,
        **log_context(state),
    )
    return {"preferences": preferences, "history": history}


def classify_urgency(state: SupportState) -> dict:
    """Classify urgency from the message text plus the ticket's own metadata."""
    metadata = state.get("ticket_metadata") or {}

    # Metadata the customer never typed, but which changes how urgent this is.
    described = state["query"]
    if metadata.get("tags") or metadata.get("channel"):
        described = (
            f"{state['query']}\n\n"
            f"[ticket metadata] channel={metadata.get('channel') or 'unknown'}; "
            f"tags={metadata.get('tags') or 'none'}; "
            f"opened={metadata.get('created_at') or 'unknown'}; "
            f"status={metadata.get('status') or 'unknown'}"
        )

    urgency = urgency_detector.invoke({"query": described})["urgency"]

    log_event(
        "classified",
        urgency=urgency,
        used_metadata=bool(metadata),
        **log_context(state),
    )
    return {"urgency": urgency}


def route_by_urgency(state: SupportState) -> str:
    """A customer asking for a person skips straight past the knowledge check."""
    if state["urgency"] == "escalation":
        return "handle_escalation"
    return "check_knowledge"


def check_knowledge(state: SupportState) -> dict:
    """
    Decide whether we have anything to answer this with, before trying.

    This is the second route into escalation: no relevant article and nothing in
    the customer's own records means a human should take it.
    """
    assessment = assess_knowledge_confidence(state["query"])

    log_event(
        "knowledge_checked",
        confidence=assessment["confidence"],
        answerable_by=assessment["answerable_by"],
        reason=assessment["reason"],
        articles=[a["article_id"] for a in assessment["articles"]],
        **log_context(state),
    )

    return {
        "knowledge_confidence": assessment["confidence"],
        "knowledge_reason": assessment["reason"],
        "knowledge_articles": assessment["articles"],
    }


def route_after_knowledge(state: SupportState) -> str:
    """Escalate when the knowledge base cannot support an answer."""
    if state["knowledge_confidence"] < KNOWLEDGE_CONFIDENCE_THRESHOLD:
        return "handle_escalation"
    return "handle_urgent" if state["urgency"] == "urgent" else "handle_normal"


def handle_normal(state: SupportState) -> dict:
    """Supervisor 2: detect the domain, then hand off to that single expert."""
    domain = domain_detector.invoke({"query": state["query"]})["domain"]
    expert = experts[domain]

    log_event("routed", path="normal", domain=domain, agent=expert.name, **log_context(state))

    result = expert.invoke(
        {"messages": build_agent_messages(state, "Resolve this support request.")}
    )
    tools_used = log_tool_calls(result["messages"], agent=expert.name, **log_context(state))
    answer = as_text(result["messages"][-1].content)
    citations = extract_citations(answer)

    return {
        "messages": [AIMessage(content=answer)],
        "domain": domain,
        "handled_by": expert.name,
        "tools_used": tools_used,
        "citations": citations,
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
    chosen = agent_names[result["current_agent_index"]]
    # Advance the rotation so the next urgent ticket goes to the next agent.
    rr_index[team_name] = result["current_agent_index"] + 1

    log_event("routed", path="urgent", domain=domain, team=team_name, agent=chosen,
              **log_context(state))
    tools_used = log_tool_calls(result["messages"], agent=chosen, **log_context(state))
    answer = as_text(result["messages"][-1].content)
    citations = extract_citations(answer)

    return {
        "messages": [AIMessage(content=answer)],
        "domain": domain,
        "handled_by": chosen,
        "tools_used": tools_used,
        "citations": citations,
        "response": answer,
        "status": "urgent_handled",
        "rr_index": rr_index,
    }


def handle_escalation(state: SupportState) -> dict:
    """Hand the ticket to a human, for either of the two reasons."""
    reason = (
        "customer_request"
        if state.get("urgency") == "escalation"
        else "no_supporting_knowledge"
    )

    log_event(
        "routed",
        path="escalation",
        reason=reason,
        confidence=state.get("knowledge_confidence"),
        agent="escalation_agent",
        **log_context(state),
    )

    if not state.get("ticket_id"):
        answer = (
            "I can connect you with a human support representative, but I need a "
            "valid ticket ID to mark this request as escalated."
        )
        return {
            "messages": [AIMessage(content=answer)],
            "escalation_reason": reason,
            "citations": [],
            "response": answer,
            "status": "pending",
        }

    header = (
        "Escalate this ticket to a human."
        if reason == "customer_request"
        else (
            "Our knowledge base does not cover this question, so it cannot be "
            f"answered reliably ({state.get('knowledge_reason') or 'no relevant article'}). "
            "Escalate this ticket to a human. Do not attempt to answer the question "
            "yourself; explain that a specialist will follow up."
        )
    )

    result = escalation_agent.invoke({"messages": build_agent_messages(state, header)})
    tools_used = log_tool_calls(result["messages"], agent="escalation_agent", **log_context(state))
    answer = as_text(result["messages"][-1].content)

    return {
        "messages": [AIMessage(content=answer)],
        "handled_by": "escalation_agent",
        "escalation_reason": reason,
        "tools_used": tools_used,
        # An escalated turn answers nothing, so it cites nothing. Clearing this
        # stops a previous turn's citations being logged against this one.
        "citations": [],
        "response": answer,
        "status": "escalated",
    }


def finalize(state: SupportState) -> dict:
    """
    Record the outcome: on the ticket, in long-term memory, and in the log.

    Both sides of the exchange are written here rather than at either end of the
    graph, so one node owns durable persistence and the nodes before it stay
    free of side effects.
    """
    response = state.get("response") or ""

    append_ticket_message(state.get("ticket_id"), "user", state["query"])
    append_ticket_message(state.get("ticket_id"), "agent", response)

    remember_resolved_issue(
        customer_id=state["customer_id"],
        domain=state.get("domain") or "escalation",
        summary=state["query"],
        outcome=state.get("status") or "unknown",
    )

    log_event(
        "resolved",
        urgency=state.get("urgency"),
        domain=state.get("domain"),
        handled_by=state.get("handled_by"),
        status=state.get("status"),
        escalation_reason=state.get("escalation_reason"),
        confidence=state.get("knowledge_confidence"),
        tools_used=state.get("tools_used"),
        citations=state.get("citations") or None,
        **log_context(state),
    )
    return {}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(SupportState)

workflow.add_node("resolve_context", resolve_context)
workflow.add_node("load_memory", load_memory)
workflow.add_node("classify_urgency", classify_urgency)
workflow.add_node("check_knowledge", check_knowledge)
workflow.add_node("handle_normal", handle_normal)
workflow.add_node("handle_urgent", handle_urgent)
workflow.add_node("handle_escalation", handle_escalation)
workflow.add_node("finalize", finalize)

workflow.add_edge(START, "resolve_context")
workflow.add_edge("resolve_context", "load_memory")
workflow.add_edge("load_memory", "classify_urgency")

# A customer asking for a person escalates without a knowledge check.
workflow.add_conditional_edges(
    "classify_urgency",
    route_by_urgency,
    {"check_knowledge": "check_knowledge", "handle_escalation": "handle_escalation"},
)

# Everything else escalates only if there is nothing to answer with.
workflow.add_conditional_edges(
    "check_knowledge",
    route_after_knowledge,
    {
        "handle_normal": "handle_normal",
        "handle_urgent": "handle_urgent",
        "handle_escalation": "handle_escalation",
    },
)

workflow.add_edge("handle_normal", "finalize")
workflow.add_edge("handle_urgent", "finalize")
workflow.add_edge("handle_escalation", "finalize")
workflow.add_edge("finalize", END)

# Short-term memory, on disk so a thread survives a restart.
support_graph = workflow.compile(checkpointer=build_checkpointer())

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
        "confidence": result.get("knowledge_confidence"),
        "handled_by": result.get("handled_by"),
        "escalation_reason": result.get("escalation_reason"),
        "tools_used": result.get("tools_used"),
        "citations": result.get("citations"),
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
            f"confidence {result['confidence']} | {result['handled_by']} | {result['status']}]"
        )
        print(f"Assistant: {result['response']}\n")
