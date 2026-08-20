"""
UDA-Hub: a multi-agent customer support system built with LangGraph.

Flow (see agentic/design/architecture.md):

    user -> load_memory -> classify_urgency
                             |-- escalation -> escalation_agent (human handoff)
                             |-- urgent     -> urgent supervisor  -> round-robin team
                             |-- normal     -> normal supervisor  -> single expert

Run it with:  python index.py
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Configuration and shared clients
# ---------------------------------------------------------------------------

# Paths are resolved relative to this file so the script runs from any folder.
BASE_DIR = Path(__file__).resolve().parent
CORE_DB = BASE_DIR / "data" / "core" / "udahub.db"
EXTERNAL_DB = BASE_DIR / "data" / "external" / "cultpass.db"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"

API_KEY = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url=BASE_URL,
    api_key=API_KEY,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=BASE_URL,
    api_key=API_KEY,
)

vectorstore = FAISS.load_local(
    str(VECTOR_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# 2. Long-term memory (survives restarts, stored next to the core data)
# ---------------------------------------------------------------------------

def init_long_term_memory() -> None:
    """Create the table that holds durable customer preferences."""
    conn = _connect(CORE_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_memory (
                customer_id TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (customer_id, key)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_long_term_memory()


@tool
def remember_customer_preference(customer_id: str, key: str, value: str) -> dict:
    """
    Save a durable fact or preference about a customer to long-term memory.

    Use this when the customer states a lasting preference or detail, for example
    key="preferred_contact" value="email", or key="city" value="Lisbon".
    Do not store one-off questions or anything sensitive such as card numbers.
    """
    conn = _connect(CORE_DB)
    try:
        conn.execute(
            """
            INSERT INTO customer_memory (customer_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(customer_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (customer_id, key, value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"saved": True, "key": key, "value": value}
    finally:
        conn.close()


@tool
def recall_customer_preferences(customer_id: str) -> dict:
    """Read everything stored in long-term memory about a customer."""
    conn = _connect(CORE_DB)
    try:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM customer_memory WHERE customer_id = ?",
            (customer_id,),
        ).fetchall()
        return {"found": len(rows) > 0, "preferences": [dict(r) for r in rows]}
    finally:
        conn.close()


def load_preferences_text(customer_id: str) -> str:
    """Same lookup as the tool above, formatted for a prompt."""
    result = recall_customer_preferences.invoke({"customer_id": customer_id})
    if not result["found"]:
        return "none on file"
    return "; ".join(f"{p['key']}={p['value']}" for p in result["preferences"])


# ---------------------------------------------------------------------------
# 3. Tools available to the expert agents
# ---------------------------------------------------------------------------

@tool
def search_rag_knowledge_base(
    query: str, category: Optional[str] = None, top_n: int = 3
) -> list[dict]:
    """
    Search the support knowledge base for help articles and policy information.

    The vector database contains semantic-searchable support documents in these
    categories: billing, reservation, technical, subscription, and general.

    Use this tool for how-to, troubleshooting, policy, or product-support questions.
    Do not use it for live customer data like account details, reservations,
    subscriptions, or tickets.

    Args:
        query: Natural-language search query.
        category: Optional category filter.
        top_n: Maximum number of results to return, capped at 3.

    Returns:
        A list of matching articles with article_id, title, category, tags, and snippet.
    """
    top_n = min(max(top_n, 1), 3)
    results = vectorstore.similarity_search(query, k=8)

    if category:
        filtered = [d for d in results if d.metadata.get("category") == category]
        # Fall back to the unfiltered hits rather than returning nothing.
        results = filtered or results

    return [
        {
            "article_id": doc.metadata.get("article_id"),
            "title": doc.metadata.get("title"),
            "category": doc.metadata.get("category"),
            "tags": doc.metadata.get("tags"),
            "snippet": doc.page_content[:220],
        }
        for doc in results[:top_n]
    ]


@tool
def get_account_user_by_external_id(account_id: str, external_user_id: str) -> dict:
    """Look up a UDA-Hub user by account and external user ID."""
    conn = _connect(CORE_DB)
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
    conn = _connect(EXTERNAL_DB)
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
    conn = _connect(EXTERNAL_DB)
    try:
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


@tool
def get_ticket_details(ticket_id: str) -> dict:
    """Get ticket metadata and linked user/account info."""
    conn = _connect(CORE_DB)
    try:
        row = conn.execute(
            """
            SELECT t.ticket_id, t.account_id, t.user_id, t.channel, t.created_at,
                   tm.status, tm.main_issue_type, tm.tags,
                   u.external_user_id, u.user_name
            FROM tickets t
            LEFT JOIN ticket_metadata tm ON t.ticket_id = tm.ticket_id
            JOIN users u ON t.user_id = u.user_id
            WHERE t.ticket_id = ?
            """,
            (ticket_id,),
        ).fetchone()
        if not row:
            return {"found": False, "message": "Ticket not found."}
        return {"found": True, "ticket": dict(row)}
    finally:
        conn.close()


@tool
def get_ticket_messages(ticket_id: str) -> dict:
    """Get the message history for a ticket."""
    conn = _connect(CORE_DB)
    try:
        rows = conn.execute(
            """
            SELECT message_id, role, content, created_at
            FROM ticket_messages
            WHERE ticket_id = ?
            ORDER BY created_at ASC
            """,
            (ticket_id,),
        ).fetchall()
        return {"found": len(rows) > 0, "messages": [dict(r) for r in rows]}
    finally:
        conn.close()


@tool
def escalate_ticket(ticket_id: str) -> dict:
    """Mark an existing support ticket as escalated."""
    conn = _connect(CORE_DB)
    try:
        cursor = conn.execute(
            "UPDATE ticket_metadata SET status = ? WHERE ticket_id = ?",
            ("escalated", ticket_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return {"success": False, "message": "Ticket not found."}
        return {"success": True, "message": "Ticket escalated."}
    finally:
        conn.close()


# Experts can read the knowledge base and customer data, and write to memory.
# escalate_ticket is deliberately kept out of this list: only the escalation
# agent is allowed to hand a ticket to a human.
agentic_tools = [
    search_rag_knowledge_base,
    get_account_user_by_external_id,
    get_user_subscription,
    get_user_reservations,
    get_ticket_details,
    get_ticket_messages,
    remember_customer_preference,
    recall_customer_preferences,
]


# ---------------------------------------------------------------------------
# 4. Classification tools
# ---------------------------------------------------------------------------

class UrgencyDataType(BaseModel):
    urgency: Literal["urgent", "normal", "escalation"] = Field(
        default="normal",
        description="Indicates if the user's query is urgent, normal or needs escalation.",
    )


@tool
def urgency_detector(query: str) -> Dict[str, str]:
    """
    Classify the urgency level of a user's support request.

    - "urgent": the request suggests immediate attention is needed.
    - "normal": the request can be handled through the standard support flow.
    - "escalation": the user asks for a human, or the case must go to a person.
    """
    messages = [
        SystemMessage(
            content=(
                "You are an urgency classification expert for customer support tickets. "
                "Analyze the user's request, intent, tone, sentiment, and severity of impact.\n"
                "Apply these rules IN ORDER and stop at the first one that matches:\n"
                "1. 'escalation' - the user asks to speak to a human, a real person, a manager, "
                "an agent or a supervisor; or threatens legal action, chargeback or cancellation; "
                "or says the bot cannot help them. This rule wins even if the message also sounds "
                "angry or time-critical.\n"
                "2. 'urgent' - something is actively broken or costing the customer money or "
                "access right now, and an automated answer can still fix it.\n"
                "3. 'normal' - everything else, including questions, how-tos and policy lookups.\n"
                "Return only the structured output."
            )
        ),
        HumanMessage(content=f"Classify the urgency of this user query: {query}"),
    ]
    result = llm.with_structured_output(UrgencyDataType).invoke(messages)
    return {"urgency": result.urgency}


class DomainDataType(BaseModel):
    domain: Literal["billing", "reservation", "technical", "subscription", "general"] = Field(
        default="general", description="Domain category of the user's query."
    )


@tool
def domain_detector(query: str) -> Dict[str, str]:
    """
    Classify a query into one of: billing, reservation, technical, subscription, general.
    """
    messages = [
        SystemMessage(
            content=(
                "You are a domain classification expert for customer support tickets. "
                "Determine which domain best matches the user's query.\n"
                "- 'billing': billing issues, refunds, payments not working.\n"
                "- 'reservation': reservation inquiries, bookings, cancellations.\n"
                "- 'technical': login issues, website problems, sign in/out, registration.\n"
                "- 'subscription': plans, monthly/yearly billing cycles, pausing or changing a plan.\n"
                "- 'general': anything not covered by the categories above.\n"
                "Return only the structured output."
            )
        ),
        HumanMessage(content=f"Classify the domain of this user query: {query}"),
    ]
    result = llm.with_structured_output(DomainDataType).invoke(messages)
    return {"domain": result.domain}


# ---------------------------------------------------------------------------
# 5. Expert agents (normal path) and agent pools (urgent path)
# ---------------------------------------------------------------------------

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


# One expert per domain: used by the normal-priority supervisor.
experts = {
    domain: make_expert(domain, f"[{domain.upper()} EXPERT]", f"{domain}_expert")
    for domain in EXPERT_BRIEFS
}

# Three interchangeable agents per domain: used by the urgent path so that
# concurrent urgent tickets are spread across a team instead of one agent.
agent_swarm_map = {
    f"{domain}_team": [
        make_expert(domain, f"[{domain.upper()} AGENT]", f"{domain}_agent_{i}")
        for i in range(1, 4)
    ]
    for domain in EXPERT_BRIEFS
}


# ---------------------------------------------------------------------------
# 6. Round-robin team graph (urgent path)
# ---------------------------------------------------------------------------

class RoundRobinState(MessagesState):
    """State for a team graph: which agents exist and whose turn it is."""

    agent_names: List[str]
    current_agent_index: int


def pick_agent(state: RoundRobinState) -> dict:
    """Normalize the incoming counter into a valid position in the rotation."""
    index = state.get("current_agent_index", 0) % len(state["agent_names"])
    return {"current_agent_index": index}


def route_round_robin(state: RoundRobinState) -> str:
    """Send the ticket to whichever agent's turn it is."""
    return state["agent_names"][state["current_agent_index"]]


def create_team(name: str, agent_pool: List[CompiledStateGraph]) -> CompiledStateGraph:
    workflow = StateGraph(RoundRobinState)
    workflow.add_node("pick_agent", pick_agent)
    for agent in agent_pool:
        workflow.add_node(agent.name, agent)

    workflow.add_edge(START, "pick_agent")
    workflow.add_conditional_edges(
        source="pick_agent",
        path=route_round_robin,
        path_map=[agent.name for agent in agent_pool],
    )
    # No checkpointer here: the team is invoked from inside a node of the main
    # graph, and the rotation counter is carried in the main graph's state.
    return workflow.compile(name=name)


agent_teams = {
    team_name: create_team(team_name, pool)
    for team_name, pool in agent_swarm_map.items()
}


# ---------------------------------------------------------------------------
# 7. Escalation agent (human handoff)
# ---------------------------------------------------------------------------

escalation_agent = create_react_agent(
    name="escalation_agent",
    model=llm,
    tools=[escalate_ticket, get_ticket_details, get_ticket_messages],
    prompt=SystemMessage(
        content=(
            "You are an escalation support agent at UdaHub. "
            "When a user's issue needs human support, or the user insists on talking to a "
            "real person, call escalate_ticket with the ticket ID from the conversation. "
            "After the escalation succeeds, tell the user: "
            "'A customer support representative will get in touch with you soon via email.' "
            "Do not claim success unless the tool confirms it."
        )
    ),
)


# ---------------------------------------------------------------------------
# 8. Main graph
# ---------------------------------------------------------------------------

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


def load_memory(state: SupportState) -> dict:
    """Recall long-term memory for this customer before doing anything else."""
    return {"preferences": load_preferences_text(state["customer_id"])}


def classify_urgency(state: SupportState) -> dict:
    result = urgency_detector.invoke({"query": state["query"]})
    return {"urgency": result["urgency"]}


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


workflow = StateGraph(SupportState)

workflow.add_node("load_memory", load_memory)
workflow.add_node("classify_urgency", classify_urgency)
workflow.add_node("handle_normal", handle_normal)
workflow.add_node("handle_urgent", handle_urgent)
workflow.add_node("handle_escalation", handle_escalation)

workflow.add_edge(START, "load_memory")
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


# ---------------------------------------------------------------------------
# 9. Entry points
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
    result = support_graph.invoke(
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


if __name__ == "__main__":
    # Demo values come from the seeded databases.
    DEMO_CUSTOMER = "a4ab87"  # external_user_id of Alice Kingsley
    DEMO_TICKET = "233314ae-815b-46aa-8466-c4163835b224"

    for demo_query in [
        "How do refunds work on my CultPass card?",                 # normal
        "My card was charged twice this morning, I need this fixed now!",  # urgent
        "I want to talk to a real human agent right now.",          # escalation
    ]:
        print(f"\n=== {demo_query}")
        outcome = run_support_query(demo_query, DEMO_CUSTOMER, DEMO_TICKET)
        print(outcome)

    # Uncomment for an interactive session:
    chat(DEMO_CUSTOMER, DEMO_TICKET)
