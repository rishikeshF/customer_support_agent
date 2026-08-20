"""Classification tools: how urgent is this ticket, and what is it about."""

from typing import Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agentic.config import llm


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
