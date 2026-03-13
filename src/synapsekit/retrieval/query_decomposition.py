"""Query Decomposition: break complex queries into sub-queries for better retrieval."""

from __future__ import annotations

from ..llm.base import BaseLLM
from .retriever import Retriever

_DECOMPOSE_PROMPT = """\
Break down the following complex question into {num_sub} simpler, \
self-contained sub-questions that together would help answer the original.

Return only the sub-questions, one per line, without numbering or bullets.

Question: {question}"""


class QueryDecompositionRetriever:
    """Query Decomposition: splits complex queries into simpler sub-queries,
    retrieves for each, and deduplicates the results.

    Usage::

        qdr = QueryDecompositionRetriever(retriever=retriever, llm=llm)
        results = await qdr.retrieve("Compare Python and Rust for ML", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        num_sub_queries: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._num_sub_queries = num_sub_queries

    async def _decompose(self, question: str) -> list[str]:
        """Use the LLM to decompose a question into sub-questions."""
        prompt = _DECOMPOSE_PROMPT.format(num_sub=self._num_sub_queries, question=question)
        response = await self._llm.generate(prompt)
        subs = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Always include the original query
        return [question, *subs[: self._num_sub_queries]]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Decompose query, retrieve for each sub-query, deduplicate."""
        sub_queries = await self._decompose(query)

        seen: set[str] = set()
        results: list[str] = []
        for sub_q in sub_queries:
            sub_results = await self._retriever.retrieve(
                sub_q, top_k=top_k, metadata_filter=metadata_filter
            )
            for text in sub_results:
                if text not in seen:
                    seen.add(text)
                    results.append(text)

        return results[:top_k]

    async def retrieve_with_sub_queries(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> tuple[list[str], list[str]]:
        """Retrieve and also return the generated sub-queries."""
        sub_queries = await self._decompose(query)

        seen: set[str] = set()
        results: list[str] = []
        for sub_q in sub_queries:
            sub_results = await self._retriever.retrieve(
                sub_q, top_k=top_k, metadata_filter=metadata_filter
            )
            for text in sub_results:
                if text not in seen:
                    seen.add(text)
                    results.append(text)

        return results[:top_k], sub_queries
