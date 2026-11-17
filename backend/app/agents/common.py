"""Shared building blocks for the agents.

Agents reuse the existing services (retriever, embeddings, vector store, Gemini
client) and the prompt/citation helpers from the answer generator.
"""

from dataclasses import dataclass, field

from app.services.answer_generator import (
    INSUFFICIENT_MARKER,
    extract_citations,
    unique_sources,
)
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMClient, LLMError
from app.services.retriever import retrieve
from app.services.vector_store import VectorStore

GROUNDING_RULES = (
    "Use ONLY facts in the numbered context; no outside knowledge or guessing. "
    "Cite passages as [1], [2]. Be concise. "
    f"If the context lacks the needed information, reply {INSUFFICIENT_MARKER} "
    "followed by one sentence on what is missing."
)


@dataclass
class AgentDeps:
    """Shared services handed to every agent (one Gemini client for all of them)."""
    embedder: EmbeddingService
    store: VectorStore
    llm: LLMClient
    top_k: int = 3
    usage: dict = field(default_factory=dict)  # Gemini calls/tokens for this request

    @property
    def max_evidence(self) -> int:
        """Cap on chunks sent to Gemini in one prompt."""
        return self.top_k + 2


@dataclass
class AgentResult:
    agent: str
    success: bool
    output: str
    insufficient_context: bool = False
    sources: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    error: str | None = None
    cited: list[dict] = field(default_factory=list)  # evidence the output actually cites


def search(deps: AgentDeps, query: str, top_k: int | None = None) -> list[dict]:
    return retrieve(query, top_k or deps.top_k, deps.embedder, deps.store)


def merge_evidence(*groups: list[dict]) -> list[dict]:
    """Combine chunk lists, dropping duplicates and keeping the best distance first."""
    seen, merged = set(), []
    for chunk in sorted((c for g in groups for c in g), key=lambda c: c["distance"]):
        key = (chunk["source"], chunk.get("page"), chunk["text"])
        if key not in seen:
            seen.add(key)
            merged.append(chunk)
    return merged


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        page = f", page {chunk['page']}" if chunk.get("page") is not None else ""
        blocks.append(f"[{number}] (source: {chunk['source']}{page})\n{chunk['text']}")
    return "\n\n".join(blocks)


def run_grounded_llm(
    agent: str, deps: AgentDeps, role: str, task: str, chunks: list[dict]
) -> AgentResult:
    """Ask Gemini to do `task` using only `chunks`, and package the result.

    Never invents an answer: empty evidence or a Gemini failure is reported as such.
    """
    if not chunks:
        return AgentResult(
            agent=agent, success=False, insufficient_context=True,
            output="No relevant information was found in the indexed documents.",
        )

    prompt = f"CONTEXT:\n{format_context(chunks)}\n\nTASK:\n{task}\n\nRESPONSE:"
    try:
        raw = deps.llm.generate(
            prompt, system_instruction=f"{role}\n{GROUNDING_RULES}", usage=deps.usage
        )
    except LLMError as exc:
        return AgentResult(
            agent=agent, success=False, output="", evidence=chunks, error=str(exc)
        )

    if raw.startswith(INSUFFICIENT_MARKER):
        detail = raw[len(INSUFFICIENT_MARKER):].lstrip(" :.-\n").rstrip()
        return AgentResult(
            agent=agent, success=False, insufficient_context=True, evidence=chunks,
            output=detail or "The documents do not contain enough information for this task.",
        )

    cited = extract_citations(raw, len(chunks))
    used = [chunks[i - 1] for i in cited] if cited else chunks
    return AgentResult(
        agent=agent, success=True, output=raw, sources=unique_sources(used),
        evidence=chunks, cited=used,
    )
