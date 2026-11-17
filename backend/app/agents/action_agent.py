"""Action Agent: recommends practical next steps from other agents' findings.

Owners and deadlines are included only when the evidence states them.
It only recommends; it never executes external actions.
"""

from app.agents.common import AgentDeps, AgentResult, merge_evidence, run_grounded_llm

NAME = "action"

ROLE = (
    "You are an action-planning agent. Recommend practical next steps based on the context "
    "and the findings of other agents. For each action give: the action, the suggested owner, "
    "and the deadline. Include an owner or deadline ONLY if the context states it; otherwise "
    "write 'not specified'. Do not execute anything; only recommend."
)


def run(
    question: str, deps: AgentDeps, chunks: list[dict], findings: list[AgentResult]
) -> AgentResult:
    # Only the chunks other agents actually cited, not everything they saw.
    evidence = merge_evidence(chunks, *[f.cited for f in findings if f.success])[: deps.max_evidence]
    notes = "\n".join(
        f"- {f.agent}: {f.output}" for f in findings if f.success and f.agent != "search"
    )
    task = f"Recommend next steps for this request:\n{question}"
    if notes:
        task += f"\n\nFindings from other agents (use only if supported by the context):\n{notes}"
    return run_grounded_llm(NAME, deps, ROLE, task, evidence)
