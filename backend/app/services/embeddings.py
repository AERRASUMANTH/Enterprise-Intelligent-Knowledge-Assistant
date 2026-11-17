"""Embedding service built on Sentence Transformers.

The model is loaded once on first use and then reused for every request.
The same model embeds both document chunks and user questions, which is
required for their vectors to be comparable.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (used for document chunks)."""
        if not texts:
            return []
        # Normalized vectors make cosine distance well-behaved (0 = identical).
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single question."""
        return self.embed_texts([query])[0]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Shared instance so the model is loaded only once per process."""
    return EmbeddingService(settings.embedding_model)
