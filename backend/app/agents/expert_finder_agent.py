"""Expert Finder Agent: identifies employees by expertise, role and responsibilities
mentioned in the indexed documents. Never invents people or contact details.

There is no employee directory: people are found only in uploaded documents.
"""

from app.agents.common import AgentDeps, AgentResult, merge_evidence, run_grounded_llm, search

NAME = "expert_finder"

DATA_NOTE = (
    "Note: people are identified only from the uploaded enterprise documents; "
    "there is no employee directory, so others may also be relevant."
)

ROLE = (
    "You are an expert-finder agent. Identify people named in the context who could help "
    "with the request, judging by meaning rather than exact words: their stated role, job "
    "title, skills, responsibilities, task ownership, or involvement in related work "
    "(e.g. someone who fixed or owns a similar issue). For each person give: name, role, "
    "why they are relevant, and contact details ONLY if they appear in the context. "
    "Never invent people, roles or contact details."
)


def run(question: str, deps: AgentDeps, chunks: list[dict] | None = None) -> AgentResult:
    # Semantic search phrased around people, so passages naming owners,
    # roles and contacts rank higher even if they share no words with the question.
    people_query = (
        f"People, job titles, owners, responsibilities and contact details related to: {question}"
    )
    evidence = merge_evidence(chunks or [], search(deps, people_query))[: deps.max_evidence]
    task = f"Who are the relevant experts or responsible people for this request?\n{question}"
    result = run_grounded_llm(NAME, deps, ROLE, task, evidence)
    result.output = f"{result.output}\n\n{DATA_NOTE}".strip()
    return result
