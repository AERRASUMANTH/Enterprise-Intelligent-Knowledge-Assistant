"""LangGraph orchestrator for the five agents.

Graph:  START -> route -> search -> [selected specialists, fixed order] -> synthesize -> END

- `route` uses SEMANTIC routing (app.agents.router): the question is embedded and
  compared with descriptions of each agent's capabilities.
- The Search Agent always runs first; its evidence is shared with the others.
- Only the selected specialists run; the Action Agent runs last so it can use
  the other agents' findings.
- `synthesize` asks Gemini (existing client) to merge all findings into one
  de-duplicated answer with citations. If that call fails, the agents' own
  outputs are returned de-duplicated, never an invented answer.
"""

import operator
import re
from dataclasses import asdict
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents import (
    action_agent,
    compliance_agent,
    expert_finder_agent,
    search_agent,
    summarization_agent,
)
from app.agents.common import AgentDeps, AgentResult, merge_evidence, run_grounded_llm
from app.agents.router import SPECIALISTS, RoutingDecision, route
from app.services.answer_generator import unique_sources

SECTION_TITLES = {
    "summarization": "Summary",
    "expert_finder": "Relevant Experts",
    "compliance": "Compliance Check",
    "action": "Recommended Actions",
}

SYNTHESIS_ROLE = (
    "You are the final-answer writer for a team of enterprise assistant agents. Combine the "
    "agents' findings and the context into ONE coherent answer to the user's question. "
    "State each fact only once, merge overlapping findings, and organise the answer by the "
    "parts of the question (use short headings or bullets when helpful). Keep names, roles, "
    "owners, deadlines and policy requirements exactly as stated. Keep any notes the agents "
    "add about limitations (e.g. 'not legal advice', 'no employee directory')."
)


class GraphState(TypedDict):
    question: str
    routing: RoutingDecision | None
    results: Annotated[list[AgentResult], operator.add]
    synthesis: AgentResult | None


# ---- helpers ----

def _search_evidence(state: GraphState) -> list[dict]:
    return next((r.evidence for r in state["results"] if r.agent == "search"), [])


def _key(chunk: dict) -> tuple:
    return (chunk["source"], chunk.get("page"), chunk["text"])


def remap_citations(text: str, own: list[dict], merged: list[dict]) -> str:
    """Rewrite an agent's [n] citations so they point into the merged evidence list."""
    index = {_key(c): i + 1 for i, c in enumerate(merged)}

    def convert(match: re.Match) -> str:
        numbers = []
        for part in re.split(r"[,\s]+", match.group(1).strip()):
            if part.isdigit() and 1 <= int(part) <= len(own):
                new = index.get(_key(own[int(part) - 1]))
                if new and new not in numbers:
                    numbers.append(new)
        return "".join(f"[{n}]" for n in numbers)

    return re.sub(r"\[([\d,\s]+)\]", convert, text)


def _normalise(line: str) -> str:
    line = re.sub(r"\[[\d,\s]+\]", "", line.lower())
    return re.sub(r"[^a-z0-9 ]+", "", line).strip()


def dedupe_sections(sections: list[tuple[str, str]]) -> str:
    """Join (title, text) sections, dropping lines already stated in an earlier section."""
    seen: set[str] = set()
    parts = []
    for title, text in sections:
        kept = []
        for line in text.splitlines():
            norm = _normalise(line)
            if norm and norm in seen:
                continue
            if norm:
                seen.add(norm)
            kept.append(line)
        body = "\n".join(kept).strip()
        if body:
            parts.append(f"## {title}\n{body}")
    return "\n\n".join(parts)


# ---- graph ----

def build_graph(deps: AgentDeps):
    def route_node(state: GraphState) -> dict:
        return {"routing": route(state["question"], deps.embedder)}

    def search_node(state: GraphState) -> dict:
        return {"results": [search_agent.run(state["question"], deps)]}

    def summarization_node(state: GraphState) -> dict:
        return {"results": [summarization_agent.run(state["question"], deps, _search_evidence(state))]}

    def expert_node(state: GraphState) -> dict:
        return {"results": [expert_finder_agent.run(state["question"], deps, _search_evidence(state))]}

    def compliance_node(state: GraphState) -> dict:
        return {"results": [compliance_agent.run(state["question"], deps, _search_evidence(state))]}

    def action_node(state: GraphState) -> dict:
        findings = [r for r in state["results"] if r.agent != "search"]
        return {"results": [action_agent.run(state["question"], deps, _search_evidence(state), findings)]}

    def synthesize_node(state: GraphState) -> dict:
        return {"synthesis": synthesize(state["question"], state["results"], deps)}

    def next_step(state: GraphState) -> str:
        done = {r.agent for r in state["results"]}
        for name in SPECIALISTS:
            if name in state["routing"].selected_agents and name not in done:
                return name
        return "synthesize"

    graph = StateGraph(GraphState)
    graph.add_node("route", route_node)
    graph.add_node("search", search_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("expert_finder", expert_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("action", action_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "search")
    for node in ["search", *SPECIALISTS]:
        graph.add_conditional_edges(node, next_step, [*SPECIALISTS, "synthesize"])
    graph.add_edge("synthesize", END)
    return graph.compile()


def synthesize(question: str, results: list[AgentResult], deps: AgentDeps) -> AgentResult:
    """Produce the single final answer, calling Gemini only when it adds value.

    - one successful specialist: its answer IS the final answer (no extra call)
    - several: one Gemini call merges them, given only the chunks they cited
    - search only: one Gemini call answers from the retrieved chunks
    """
    specialists = [r for r in results if r.agent != "search"]
    succeeded = [r for r in specialists if r.success]

    # Specialists ran but none produced anything: nothing truthful to synthesize.
    if specialists and not succeeded:
        return AgentResult(agent="synthesis", success=False, output="")

    if len(succeeded) == 1:
        only = succeeded[0]
        return AgentResult(
            agent="synthesis", success=True, output=only.output,
            sources=only.sources, evidence=only.evidence, cited=only.cited,
        )

    if succeeded:
        merged = merge_evidence(*[r.cited for r in succeeded])
    else:
        merged = _search_evidence({"results": results})
    task = f"User question:\n{question}"
    if succeeded:
        findings = "\n\n".join(
            f"### {SECTION_TITLES[r.agent]} (from the {r.agent} agent)\n"
            f"{remap_citations(r.output, r.evidence, merged)}"
            for r in succeeded
        )
        task += (
            "\n\nAgent findings (citations refer to the context passages above):\n"
            f"{findings}\n\nWrite the final combined answer."
        )
    else:
        task += "\n\nAnswer the question directly from the context."
    return run_grounded_llm("synthesis", deps, SYNTHESIS_ROLE, task, merged)


def run_multi_agent(question: str, deps: AgentDeps) -> dict:
    state = build_graph(deps).invoke(
        {"question": question, "routing": None, "results": [], "synthesis": None}
    )
    routing: RoutingDecision = state["routing"]
    results: list[AgentResult] = state["results"]
    synthesis: AgentResult = state["synthesis"]
    specialists_ok = [r for r in results if r.agent != "search" and r.success]

    if synthesis.success:
        final_answer = synthesis.output
        sources = synthesis.sources
    elif specialists_ok:
        # Synthesis failed: fall back to the agents' real outputs, de-duplicated.
        final_answer = dedupe_sections([(SECTION_TITLES[r.agent], r.output) for r in specialists_ok])
        sources = unique_sources([
            {"source": s["filename"], "page": s["page"]} for r in specialists_ok for s in r.sources
        ])
    elif synthesis.insufficient_context or all(r.insufficient_context for r in results if not r.success):
        final_answer = "The available documents do not contain enough information to answer this question."
        # Add the agents' own explanation of what is missing / what is available.
        detail = synthesis.output or next(
            (r.output for r in results if r.agent != "search" and r.insufficient_context and r.output), ""
        )
        if detail:
            final_answer += f" {detail}"
        sources = []
    else:
        error = next((r.error for r in [*results, synthesis] if r.error), "unknown error")
        final_answer = f"No answer could be generated because Gemini failed: {error}"
        sources = []

    all_ok = all(r.success for r in results) and synthesis.success
    answered = synthesis.success or bool(specialists_ok)
    status = "complete" if all_ok else ("partial" if answered else "failed")

    return {
        "question": question,
        "selected_agents": routing.selected_agents,
        "routing": {
            "method": "semantic",
            "confidence": routing.confidence,
            "fallback_to_search": routing.fallback,
            "scores": [asdict(s) for s in routing.scores],
        },
        "agent_results": [asdict(r) for r in results],
        "final_answer": final_answer,
        "sources": sources,
        "success": all_ok,
        "partial_success": status == "partial",
        "status": status,
        "synthesis_error": synthesis.error,
        "model": deps.llm.model,
        "usage": {
            "gemini_calls": deps.usage.get("gemini_calls", 0),
            "prompt_tokens": deps.usage.get("prompt_tokens", 0),
            "output_tokens": deps.usage.get("output_tokens", 0),
        },
    }
