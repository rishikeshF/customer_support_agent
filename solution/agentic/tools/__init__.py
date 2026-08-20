"""All tools the agents can call.

`agentic_tools` is the set handed to the domain experts. `escalate_ticket` is
deliberately left out of it: only the escalation agent may hand a ticket to a
human, so an expert cannot quietly escalate its way out of a hard question.
"""

from agentic.tools.classification import (
    DomainDataType,
    UrgencyDataType,
    domain_detector,
    urgency_detector,
)
from agentic.tools.customer import (
    get_account_user_by_external_id,
    get_user_reservations,
    get_user_subscription,
)
from agentic.tools.knowledge import (
    assess_knowledge_confidence,
    extract_citations,
    known_article_ids,
    search_rag_knowledge_base,
)
from agentic.tools.memory import (
    init_long_term_memory,
    load_history_text,
    load_preferences_text,
    recall_customer_preferences,
    recall_past_issues,
    remember_customer_preference,
    remember_resolved_issue,
)
from agentic.tools.tickets import (
    append_ticket_message,
    escalate_ticket,
    get_ticket_details,
    get_ticket_messages,
)

agentic_tools = [
    search_rag_knowledge_base,
    get_account_user_by_external_id,
    get_user_subscription,
    get_user_reservations,
    get_ticket_details,
    get_ticket_messages,
    remember_customer_preference,
    recall_customer_preferences,
    recall_past_issues,
]

__all__ = [
    "agentic_tools",
    "search_rag_knowledge_base",
    "assess_knowledge_confidence",
    "extract_citations",
    "known_article_ids",
    "get_account_user_by_external_id",
    "get_user_subscription",
    "get_user_reservations",
    "get_ticket_details",
    "get_ticket_messages",
    "append_ticket_message",
    "escalate_ticket",
    "remember_customer_preference",
    "recall_customer_preferences",
    "load_preferences_text",
    "remember_resolved_issue",
    "recall_past_issues",
    "load_history_text",
    "init_long_term_memory",
    "urgency_detector",
    "domain_detector",
    "UrgencyDataType",
    "DomainDataType",
]
