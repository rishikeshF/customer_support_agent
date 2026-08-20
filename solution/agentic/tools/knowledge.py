"""RAG tool: semantic search over the support knowledge base."""

from typing import Optional

from langchain_core.tools import tool

from agentic.config import vectorstore


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
