"""Semantic vector retrieval for tool selection.

Encodes all tool descriptions into vector embeddings using
sentence-transformers (all-MiniLM-L6-v2) and stores them in a FAISS
index for fast similarity search.

This implements the "vector" routing mode from the research paper:
instead of stuffing every tool into the LLM prompt, only the top-k
most semantically relevant tools are retrieved for a given query.
"""

from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from model.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class ToolRetriever:
    """Vector-indexed tool retriever using sentence-transformers + FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.tools = TOOL_REGISTRY
        self.descriptions = [t["description"] for t in self.tools]

        # Encode all tool descriptions (normalized for cosine similarity)
        self.embeddings = self.encoder.encode(
            self.descriptions,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Build FAISS index (inner product = cosine similarity on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        logger.info(
            "ToolRetriever initialized | tools=%d | embedding_dim=%d",
            len(self.tools), dim,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant tools for a given query.

        Args:
            query: Natural-language user query.
            top_k: Maximum number of tools to return.

        Returns:
            List of dicts, each containing the tool dict and a similarity score.
        """
        query_vec = self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        k = min(top_k, len(self.tools))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "tool": self.tools[idx],
                    "score": float(score),
                })

        logger.info("Retrieved %d tools for query: '%s'", len(results), query[:80])
        return results

    def retrieve_tool_names(self, query: str, top_k: int = 5) -> list[str]:
        """Convenience method — return just the tool names."""
        return [r["tool"]["name"] for r in self.retrieve(query, top_k)]
