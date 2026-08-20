"""The escalation agent: the only agent allowed to hand a ticket to a human."""

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from agentic.config import llm
from agentic.tools import escalate_ticket, get_ticket_details, get_ticket_messages

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
