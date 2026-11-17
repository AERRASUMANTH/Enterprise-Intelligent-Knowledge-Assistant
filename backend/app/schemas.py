"""Request/response models for the API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class UploadResponse(BaseModel):
    status: Literal["indexed", "replaced", "unchanged"]
    filename: str
    chunks_indexed: int


class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    pages: list[int] = Field(description="Page numbers present (empty for DOCX/TXT)")


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class RetrieveRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, examples=["How was the previous server failure resolved?"]
    )
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    distance: float = Field(description="Cosine distance: 0 = identical, larger = less similar")


class RetrieveResponse(BaseModel):
    question: str
    results: list[RetrievedChunk]


# ---- Chat (grounded answer generation) ----

class ChatRequest(RetrieveRequest):
    """Same fields as retrieval: `question` plus optional `top_k`."""

    question: str = Field(
        ..., min_length=1, examples=["How was the previous server outage resolved?"]
    )


class SourceRef(BaseModel):
    filename: str
    page: int | None = None


class ChatMetadata(BaseModel):
    model: str | None
    top_k: int
    retrieved_chunks: int
    insufficient_context: bool


class AgentOutput(BaseModel):
    agent: str
    success: bool
    output: str
    insufficient_context: bool = False
    sources: list[SourceRef] = []
    evidence: list[RetrievedChunk] = []
    error: str | None = None


class RouteScoreOut(BaseModel):
    agent: str
    score: float = Field(description="Cosine similarity between the question and the agent's capabilities")
    selected: bool
    reason: str


class RoutingInfo(BaseModel):
    method: str
    confidence: float
    fallback_to_search: bool
    scores: list[RouteScoreOut]


class MultiAgentResponse(BaseModel):
    question: str
    selected_agents: list[str]
    routing: RoutingInfo
    agent_results: list[AgentOutput]
    final_answer: str
    sources: list[SourceRef]
    success: bool = Field(description="True only if every selected agent and the final synthesis succeeded")
    partial_success: bool = Field(description="True if an answer was produced but some step failed")
    status: Literal["complete", "partial", "failed"]
    synthesis_error: str | None = None
    model: str | None = None
    usage: dict[str, int] = Field(
        default_factory=dict, description="Gemini calls and tokens used for this request"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    evidence: list[RetrievedChunk]
    metadata: ChatMetadata
    success: bool = Field(description="True when a grounded answer was found in the documents")
