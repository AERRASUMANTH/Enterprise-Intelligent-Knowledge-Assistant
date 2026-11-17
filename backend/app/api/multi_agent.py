"""Multi-agent endpoint: runs the LangGraph orchestrator."""

from fastapi import APIRouter, Depends

from app.agents.common import AgentDeps
from app.agents.orchestrator import run_multi_agent
from app.api.chat import require_llm_client
from app.schemas import ChatRequest, MultiAgentResponse
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(tags=["multi-agent"])


@router.post("/multi-agent", response_model=MultiAgentResponse)
def multi_agent(
    request: ChatRequest,
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: VectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(require_llm_client),
) -> MultiAgentResponse:
    deps = AgentDeps(embedder=embedder, store=store, llm=llm, top_k=request.top_k)
    return MultiAgentResponse(**run_multi_agent(request.question, deps))
