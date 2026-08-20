"""RAG tools: semantic search over the support knowledge base, and a check on
whether that knowledge base can actually answer a given question."""

from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agentic.config import llm, vectorstore


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


class KnowledgeConfidence(BaseModel):
    """How well the knowledge base covers a question."""

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 means the articles are irrelevant, 1 means they answer the question outright.",
    )
    answerable_by: Literal["knowledge_base", "customer_data", "neither"] = Field(
        default="neither",
        description="Where the answer would come from, if anywhere.",
    )
    reason: str = Field(default="", description="One short sentence explaining the score.")


def assess_knowledge_confidence(query: str) -> dict:
    """
    Score whether we can answer a question at all before an agent tries.

    A low score is the signal to escalate: if neither the knowledge base nor the
    customer's own records cover the question, no amount of agent effort will
    produce a grounded answer, so it belongs with a human.

    Account-specific questions ("what are my reservations") score through
    `answerable_by="customer_data"` even when no article matches, because the
    agents' database tools can answer those without a knowledge base article.
    """
    articles = search_rag_knowledge_base.invoke({"query": query, "top_n": 3})

    if not articles:
        return {
            "confidence": 0.0,
            "answerable_by": "neither",
            "reason": "The knowledge base returned no articles.",
            "articles": [],
        }

    rendered = "\n\n".join(
        f"[{a['article_id']}] {a['title']} ({a['category']})\n{a['snippet']}" for a in articles
    )

    messages = [
        SystemMessage(
            content=(
                "You decide whether a customer support question can be answered from the "
                "material available, before an agent attempts it.\n"
                "Score 'confidence' from 0 to 1 for how well the retrieved articles cover "
                "the question.\n"
                "Set 'answerable_by':\n"
                "- 'knowledge_base' if the articles genuinely answer it;\n"
                "- 'customer_data' if it is about this customer's own account, bookings, "
                "subscription or charges, which support tools can look up directly. Use a "
                "confidence of at least 0.7 in this case even if no article matches;\n"
                "- 'neither' if the question is outside both. Score this below 0.5.\n"
                "Be strict: a vaguely related article is not coverage."
            )
        ),
        HumanMessage(content=f"Question:\n{query}\n\nRetrieved articles:\n{rendered}"),
    ]

    result = llm.with_structured_output(KnowledgeConfidence).invoke(messages)
    return {
        "confidence": result.confidence,
        "answerable_by": result.answerable_by,
        "reason": result.reason,
        "articles": articles,
    }
