"""Retrieval endpoints."""

from fastapi import APIRouter, Depends

from app.schemas import RetrievedChunk, RetrieveRequest, RetrieveResponse
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.retriever import retrieve
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    request: RetrieveRequest,
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: VectorStore = Depends(get_vector_store),
) -> RetrieveResponse:
    results = retrieve(request.question, request.top_k, embedder, store)
    return RetrieveResponse(
        question=request.question,
        results=[RetrievedChunk(**r) for r in results],
    )
