"""ChromaDB vector store service.

Duplicate strategy: a document is identified by its filename.
- Re-uploading the same file with the same content is a no-op ("unchanged").
- Re-uploading a file with the same name but new content replaces all of its
  old chunks ("replaced"), so stale chunks never linger.
Chunk IDs are stable: "<filename>::<chunk_index>".
"""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.chunker import Chunk


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
        )

    # ---- Writing ----

    def get_document_hash(self, source: str) -> str | None:
        """Return the stored content hash for a document, or None if not indexed."""
        found = self._collection.get(where={"source": source}, limit=1, include=["metadatas"])
        if not found["ids"]:
            return None
        return found["metadatas"][0].get("content_hash")

    def delete_document(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]], content_hash: str
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        if not chunks:
            return

        metadatas = []
        for chunk in chunks:
            meta = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "content_hash": content_hash,
            }
            # ChromaDB metadata values cannot be None, so omit missing pages.
            if chunk.page is not None:
                meta["page"] = chunk.page
            metadatas.append(meta)

        self._collection.upsert(
            ids=[f"{c.source}::{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ---- Reading ----

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[dict]:
        """One entry per indexed document: filename, chunk count and pages."""
        found = self._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in found["metadatas"]:
            doc = docs.setdefault(meta["source"], {"filename": meta["source"], "chunks": 0, "pages": set()})
            doc["chunks"] += 1
            if meta.get("page") is not None:
                doc["pages"].add(meta["page"])
        return [
            {"filename": d["filename"], "chunks": d["chunks"], "pages": sorted(d["pages"])}
            for d in sorted(docs.values(), key=lambda d: d["filename"].lower())
        ]

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Return up to `top_k` nearest chunks, closest first."""
        total = self.count()
        if total == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": text,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "distance": distance,
            }
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    """Shared instance pointing at the configured persistent directory."""
    return VectorStore(settings.chroma_persist_dir, settings.chroma_collection_name)
