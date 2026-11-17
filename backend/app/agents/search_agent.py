"""Search Agent: finds relevant document chunks using the existing RAG retriever.

No LLM call: it returns the raw evidence with filename, page and distance.
"""

from app.agents.common import AgentDeps, AgentResult, search
from app.services.answer_generator import unique_sources

NAME = "search"


def run(question: str, deps: AgentDeps) -> AgentResult:
    chunks = search(deps, question)
    if not chunks:
        return AgentResult(
            agent=NAME, success=False, insufficient_context=True,
            output="No documents are indexed yet, so nothing could be searched.",
        )
    files = ", ".join(sorted({c["source"] for c in chunks}))
    return AgentResult(
        agent=NAME,
        success=True,
        output=f"Found {len(chunks)} relevant passage(s) in: {files}.",
        sources=unique_sources(chunks),
        evidence=chunks,
    )
