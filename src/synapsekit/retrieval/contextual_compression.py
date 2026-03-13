"""Contextual Compression: compress retrieved documents to only relevant parts."""

from __future__ import annotations

from ..llm.base import BaseLLM
from .retriever import Retriever

_COMPRESS_PROMPT = """\
Given the following question and document, extract only the parts of the \
document that are directly relevant to answering the question. If no part \
is relevant, respond with "NOT_RELEVANT".

Return only the relevant excerpt, nothing else.

Question: {question}

Document: {document}"""


class ContextualCompressionRetriever:
    """Contextual Compression: retrieves documents then compresses them to
    include only the parts relevant to the query.

    Usage::

        ccr = ContextualCompressionRetriever(retriever=retriever, llm=llm)
        results = await ccr.retrieve("What is attention?", top_k=5)
        # Returns compressed excerpts, not full documents
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        fetch_k: int = 10,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._fetch_k = fetch_k

    async def _compress(self, question: str, document: str) -> str | None:
        """Compress a document to only the parts relevant to the question."""
        prompt = _COMPRESS_PROMPT.format(question=question, document=document)
        response = (await self._llm.generate(prompt)).strip()
        if response.upper() == "NOT_RELEVANT" or not response:
            return None
        return response

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve, compress, and filter documents."""
        candidates = await self._retriever.retrieve(
            query, top_k=self._fetch_k, metadata_filter=metadata_filter
        )

        compressed: list[str] = []
        for doc in candidates:
            result = await self._compress(query, doc)
            if result is not None:
                compressed.append(result)
                if len(compressed) >= top_k:
                    break

        return compressed
