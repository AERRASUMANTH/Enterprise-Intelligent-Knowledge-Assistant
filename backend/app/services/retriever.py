"""Retriever: turns a question into the most relevant stored chunks.

Retrieval is semantic: the question is embedded with the same Sentence
Transformers model as the documents and ChromaDB returns the nearest chunks
by cosine distance, so different wording with the same meaning still matches.
"""

from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


def retrieve(
    question: str,
    top_k: int,
    embedder: EmbeddingService,
    store: VectorStore,
    max_distance: float | None = None,
) -> list[dict]:
    """Return up to `top_k` chunks ordered by relevance (smallest distance first).

    Chunks farther than `max_distance` (default: settings.retrieval_max_distance)
    are dropped as irrelevant. Returns an empty list when nothing is indexed.
    """
    if store.count() == 0:
        return []
    max_distance = settings.retrieval_max_distance if max_distance is None else max_distance
    query_embedding = embedder.embed_query(question)
    return [r for r in store.query(query_embedding, top_k) if r["distance"] <= max_distance]
