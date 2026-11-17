"""Compliance Checker Agent: compares a situation with retrieved company policy.

Provides guidance based on the documents only, never legal certainty.
"""

from app.agents.common import AgentDeps, AgentResult, merge_evidence, run_grounded_llm, search

NAME = "compliance"

DISCLAIMER = (
    "Note: this is an automated reading of the indexed company documents, not legal advice. "
    "Confirm with your compliance or HR team."
)

ROLE = (
    "You are a compliance-checker agent. Find the company policy requirements in the context "
    "that apply to the user's question or situation. List each relevant requirement with its "
    "citation, then state whether the situation appears compliant, non-compliant, or unclear "
    "based ONLY on the context. Never claim legal certainty."
)


def run(question: str, deps: AgentDeps, chunks: list[dict] | None = None) -> AgentResult:
    policy_chunks = search(deps, f"{question} policy rule requirement compliance")
    evidence = merge_evidence(chunks or [], policy_chunks)[: deps.max_evidence]
    task = f"Check this question or situation against the company policies:\n{question}"
    result = run_grounded_llm(NAME, deps, ROLE, task, evidence)
    if result.success:
        result.output = f"{result.output}\n\n{DISCLAIMER}"
    return result
