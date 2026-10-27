import logfire
from app.agents.state import AgentState
from app.config import settings, reload_retrieval_settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    limit / top_n come from settings (SEARCH_LIMIT, RERANK_TOP_N) — set via
    eval UI Apply → .env, or defaults in app/config.py.
    """
    reload_retrieval_settings()
    query = state["current_query"]
    search_limit = settings.SEARCH_LIMIT
    rerank_top_n = settings.RERANK_TOP_N
    # Never keep more docs than we retrieved
    if rerank_top_n > search_limit:
        rerank_top_n = search_limit

    with logfire.span(
        "🔍 Knowledge Retrieval",
        search_limit=search_limit,
        rerank_top_n=rerank_top_n,
    ):
        logfire.info(
            f"Searching Qdrant for: {query} "
            f"(limit={search_limit}, rerank_top_n={rerank_top_n})"
        )
        raw_results = search_enterprise_knowledge(query, limit=search_limit)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc["content"] for doc in raw_results]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(
                query, doc_contents, top_n=rerank_top_n
            )
            logfire.info(
                f"Reranking complete. Kept top {len(reranked_contents)} most relevant chunks."
            )

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
