"""Ensemble Retrieval: fuse results from multiple retrievers."""

from __future__ import annotations

from .retriever import Retriever


class EnsembleRetriever:
    """Ensemble Retrieval: combines results from multiple retrievers using
    Reciprocal Rank Fusion (RRF) for better recall and diversity.

    Usage::

        ensemble = EnsembleRetriever(
            retrievers=[retriever_a, retriever_b],
            weights=[0.7, 0.3],
        )
        results = await ensemble.retrieve("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required.")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must have the same length as retrievers.")
        self._retrievers = retrievers
        self._weights = weights or [1.0] * len(retrievers)
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve from all retrievers and fuse with weighted RRF."""
        scores: dict[str, float] = {}

        for retriever, weight in zip(self._retrievers, self._weights, strict=True):
            results = await retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
            for rank, text in enumerate(results):
                scores[text] = scores.get(text, 0.0) + weight / (self._rrf_k + rank + 1)

        sorted_texts = sorted(scores, key=lambda t: scores[t], reverse=True)
        return sorted_texts[:top_k]
