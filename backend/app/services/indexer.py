"""Indexing pipeline: load -> chunk -> embed -> store.

This is the single place that wires the RAG services together for ingestion,
so API routes (and later, agents) don't duplicate the steps.
"""

import hashlib
from dataclasses import dataclass

from app.config import settings
from app.services.chunker import chunk_pages
from app.services.document_loader import load_document
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass
class IndexResult:
    filename: str
    status: str        # "indexed", "replaced" or "unchanged"
    chunks_indexed: int


def index_document(
    filename: str,
    data: bytes,
    embedder: EmbeddingService,
    store: VectorStore,
) -> IndexResult:
    content_hash = hashlib.sha256(data).hexdigest()

    existing_hash = store.get_document_hash(filename)
    if existing_hash == content_hash:
        return IndexResult(filename=filename, status="unchanged", chunks_indexed=0)

    # Parse and chunk before touching the store, so a bad file never
    # deletes the previously indexed version.
    pages = load_document(filename, data)
    chunks = chunk_pages(pages, filename, settings.chunk_size, settings.chunk_overlap)
    embeddings = embedder.embed_texts([c.text for c in chunks])

    if existing_hash is not None:
        store.delete_document(filename)
    store.add_chunks(chunks, embeddings, content_hash)

    status = "replaced" if existing_hash is not None else "indexed"
    return IndexResult(filename=filename, status=status, chunks_indexed=len(chunks))
