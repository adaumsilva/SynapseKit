"""Corrective RAG (CRAG): self-correcting retrieval with relevance grading."""

from __future__ import annotations

from ..llm.base import BaseLLM
from .retriever import Retriever

_GRADE_PROMPT = """\
You are a relevance grader. Given a user question and a retrieved document, \
determine if the document is relevant to answering the question.

Respond with exactly one word: "relevant" or "irrelevant".

Question: {question}

Document: {document}"""


class CRAGRetriever:
    """Corrective RAG: grades retrieved documents for relevance and falls back
    to web search or query rewriting when results are poor.

    Usage::

        crag = CRAGRetriever(retriever=retriever, llm=llm)
        results = await crag.retrieve("What is CRAG?", top_k=5)

    The retriever:
    1. Retrieves initial candidates
    2. Grades each for relevance using the LLM
    3. If too few are relevant, rewrites the query and retrieves again
    4. Returns only relevant documents
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        relevance_threshold: float = 0.5,
        max_retries: int = 1,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._relevance_threshold = relevance_threshold
        self._max_retries = max_retries

    async def _grade_document(self, question: str, document: str) -> bool:
        """Use the LLM to grade a document's relevance to the question."""
        prompt = _GRADE_PROMPT.format(question=question, document=document)
        response = await self._llm.generate(prompt)
        return response.strip().lower().startswith("relevant")

    async def _rewrite_query(self, question: str) -> str:
        """Use the LLM to rewrite the query for better retrieval."""
        prompt = (
            "Rewrite this question to be more specific and better suited for "
            "semantic search. Return only the rewritten question, nothing else.\n\n"
            f"Original question: {question}"
        )
        return (await self._llm.generate(prompt)).strip()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve with corrective grading and optional query rewriting."""
        current_query = query

        for _attempt in range(1 + self._max_retries):
            candidates = await self._retriever.retrieve(
                current_query, top_k=top_k * 2, metadata_filter=metadata_filter
            )

            if not candidates:
                if _attempt < self._max_retries:
                    current_query = await self._rewrite_query(current_query)
                    continue
                return []

            # Grade each candidate
            relevant = []
            for doc in candidates:
                if await self._grade_document(query, doc):
                    relevant.append(doc)
                    if len(relevant) >= top_k:
                        break

            # Check if enough relevant docs were found
            ratio = len(relevant) / len(candidates) if candidates else 0
            if ratio >= self._relevance_threshold or _attempt >= self._max_retries:
                return relevant[:top_k]

            # Not enough relevant results — rewrite and retry
            current_query = await self._rewrite_query(current_query)

        return []

    async def retrieve_with_grades(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> tuple[list[str], dict]:
        """Retrieve and return grading metadata for transparency."""
        candidates = await self._retriever.retrieve(
            query, top_k=top_k * 2, metadata_filter=metadata_filter
        )

        graded: list[dict] = []
        relevant: list[str] = []
        for doc in candidates:
            is_relevant = await self._grade_document(query, doc)
            graded.append({"text": doc, "relevant": is_relevant})
            if is_relevant:
                relevant.append(doc)

        return relevant[:top_k], {
            "total_candidates": len(candidates),
            "relevant_count": len(relevant),
            "grades": graded,
        }
