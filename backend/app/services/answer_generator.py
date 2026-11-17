"""Grounded answer generation: retrieve chunks, then ask Gemini to answer
using ONLY those chunks.

Gemini cites the chunks it used as [1], [2], ...; we turn those citations into
the list of sources, so every source in the response really backs the answer.
"""

import re
from dataclasses import dataclass, field

from app.services.embeddings import EmbeddingService
from app.services.llm import LLMClient
from app.services.retriever import retrieve
from app.services.vector_store import VectorStore

INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

NO_DOCUMENTS_ANSWER = (
    "I could not find any relevant information in the indexed documents. "
    "Please upload documents related to your question."
)
INSUFFICIENT_ANSWER = (
    "The available documents do not contain enough information to answer this question."
)

SYSTEM_INSTRUCTION = f"""You are an enterprise knowledge assistant.
Answer the user's question using ONLY the numbered context passages provided.

Rules:
- Use only facts stated in the context. Never use outside knowledge or guess.
- Cite the passages you used with their numbers in square brackets, e.g. [1] or [2][3].
- Be concise and factual.
- If the context does not contain the information needed to answer, reply with exactly
  {INSUFFICIENT_MARKER}
  followed by one short sentence saying what information is missing.
- If the context answers only part of the question, answer that part and clearly say
  which part the documents do not cover."""


@dataclass
class AnswerResult:
    answer: str
    success: bool                 # True when a grounded answer was produced
    insufficient_context: bool
    sources: list[dict] = field(default_factory=list)    # [{"filename", "page"}]
    evidence: list[dict] = field(default_factory=list)   # retrieved chunks
    model: str | None = None


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        page = f", page {chunk['page']}" if chunk.get("page") is not None else ""
        blocks.append(f"[{number}] (source: {chunk['source']}{page})\n{chunk['text']}")
    context = "\n\n".join(blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"


def extract_citations(answer: str, num_chunks: int) -> list[int]:
    """Return the valid 1-based passage numbers cited in the answer, in order."""
    cited = []
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in re.split(r"[,\s]+", group.strip()):
            if part.isdigit() and 1 <= int(part) <= num_chunks and int(part) not in cited:
                cited.append(int(part))
    return cited


def unique_sources(chunks: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for chunk in chunks:
        key = (chunk["source"], chunk.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"filename": chunk["source"], "page": chunk.get("page")})
    return sources


def generate_answer(
    question: str,
    top_k: int,
    embedder: EmbeddingService,
    store: VectorStore,
    llm: LLMClient,
) -> AnswerResult:
    """Answer a question from the indexed documents.

    Raises LLMError if Gemini cannot be reached; other outcomes are returned.
    """
    chunks = retrieve(question, top_k, embedder, store)

    # Nothing indexed: don't spend an LLM call.
    if not chunks:
        return AnswerResult(
            answer=NO_DOCUMENTS_ANSWER, success=False, insufficient_context=True, model=llm.model
        )

    raw = llm.generate(build_prompt(question, chunks), system_instruction=SYSTEM_INSTRUCTION)

    if raw.startswith(INSUFFICIENT_MARKER):
        # Drop separators right after the marker, but keep the sentence's own punctuation.
        detail = raw[len(INSUFFICIENT_MARKER):].lstrip(" :.-\n").rstrip()
        answer = f"{INSUFFICIENT_ANSWER} {detail}".strip() if detail else INSUFFICIENT_ANSWER
        return AnswerResult(
            answer=answer,
            success=False,
            insufficient_context=True,
            evidence=chunks,
            model=llm.model,
        )

    cited = extract_citations(raw, len(chunks))
    # If Gemini forgot to cite, fall back to all retrieved chunks as sources.
    used_chunks = [chunks[i - 1] for i in cited] if cited else chunks

    return AnswerResult(
        answer=raw,
        success=True,
        insufficient_context=False,
        sources=unique_sources(used_chunks),
        evidence=chunks,
        model=llm.model,
    )
