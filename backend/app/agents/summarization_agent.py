"""Summarization Agent: concise, source-referenced summaries via Gemini.

Accepts retrieved chunks (the normal case) or raw document text.
"""

from app.agents.common import AgentDeps, AgentResult, run_grounded_llm

NAME = "summarization"

ROLE = (
    "You are a summarization agent for enterprise documents. Produce a concise summary "
    "that keeps important facts (names, dates, numbers, decisions) and their citations. "
    "Summarize only the specific topic requested. If the context covers a different topic "
    "(e.g. another policy than the one asked about), reply with the insufficient-context "
    "marker and name the topics the documents do cover."
)


def run(question: str, deps: AgentDeps, chunks: list[dict]) -> AgentResult:
    task = (
        "Summarize the information in the context that is relevant to this request, "
        f"in at most 6 short bullet points:\n{question}"
    )
    return run_grounded_llm(NAME, deps, ROLE, task, chunks)


def summarize_text(text: str, deps: AgentDeps, source: str = "provided text") -> AgentResult:
    """Summarize arbitrary document text (not from the vector store)."""
    chunk = {"text": text, "source": source, "page": None, "distance": 0.0}
    task = "Summarize this document in at most 6 short bullet points."
    return run_grounded_llm(NAME, deps, ROLE, task, [chunk] if text.strip() else [])
