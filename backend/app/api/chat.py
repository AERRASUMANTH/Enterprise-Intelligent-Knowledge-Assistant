"""Chat endpoint: grounded question answering over the indexed documents."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import ChatMetadata, ChatRequest, ChatResponse, RetrievedChunk, SourceRef
from app.services.answer_generator import generate_answer
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.llm import LLMClient, LLMError, LLMNotConfiguredError, get_llm_client
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(tags=["chat"])


def require_llm_client() -> LLMClient:
    """Dependency that turns missing/placeholder Gemini settings into a clear 503 error."""
    try:
        return get_llm_client()
    except LLMNotConfiguredError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Gemini is not configured: {exc} Then restart the server.",
        ) from exc


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: VectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(require_llm_client),
) -> ChatResponse:
    try:
        result = generate_answer(request.question, request.top_k, embedder, store, llm)
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        evidence=[RetrievedChunk(**e) for e in result.evidence],
        metadata=ChatMetadata(
            model=result.model,
            top_k=request.top_k,
            retrieved_chunks=len(result.evidence),
            insufficient_context=result.insufficient_context,
        ),
        success=result.success,
    )
