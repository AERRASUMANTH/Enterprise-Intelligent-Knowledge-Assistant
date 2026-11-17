"""Tests for the five agents, semantic routing and the LangGraph orchestrator (Gemini mocked)."""

import pytest

from app.agents import (
    action_agent,
    compliance_agent,
    expert_finder_agent,
    search_agent,
    summarization_agent,
)
from app.agents.common import AgentDeps
from app.agents.orchestrator import dedupe_sections, remap_citations, run_multi_agent
from app.agents.router import route
from app.api.chat import require_llm_client
from app.main import app
from app.services.answer_generator import INSUFFICIENT_MARKER
from app.services.indexer import index_document
from app.services.llm import LLMError
from tests.conftest import make_pdf
from tests.fake_llm import FakeLLM

DATASET = make_pdf([
    "Incident INC-1: an expired TLS certificate caused the outage. A new certificate was installed.",
    "Follow-up: add certificate expiry alerts. Owner: Marcus Odell, Network Engineer, due 2026-11-20.",
    "Travel policy: expenses over 500 dollars require manager approval before booking.",
])

REPLIES = {
    "summarization agent": "The outage was fixed by installing a new certificate [1].",
    "expert-finder": "Marcus Odell (Network Engineer) owns certificate alerts [1].",
    "compliance-checker": "Expenses over 500 dollars need manager approval [1].",
    "action-planning": "Add expiry alerts. Owner: Marcus Odell. Deadline: 2026-11-20 [1].",
    "final-answer writer": "Combined answer: a new certificate was installed [1].",
}


def replies(fail_for: str | None = None):
    """Mock Gemini: reply chosen by which agent is asking; optionally fail one agent."""
    def respond(prompt, system):
        for marker, text in REPLIES.items():
            if marker in system:
                if marker == fail_for:
                    raise LLMError("Gemini API error (503): overloaded")
                return text
        raise AssertionError("unexpected agent")
    return respond


@pytest.fixture
def deps(embedder, store):
    index_document("dataset.pdf", DATASET, embedder, store)
    return AgentDeps(embedder=embedder, store=store, llm=FakeLLM(replies()), top_k=3)


# ---- Individual agents ----

def test_search_agent_returns_evidence(deps):
    result = search_agent.run("How was the outage fixed?", deps)
    assert result.success
    assert result.evidence[0]["page"] == 1
    assert result.sources[0]["filename"] == "dataset.pdf"
    assert deps.llm.calls == []  # search never calls Gemini


def test_search_agent_empty_database(embedder, store):
    deps = AgentDeps(embedder=embedder, store=store, llm=FakeLLM("unused"))
    result = search_agent.run("anything", deps)
    assert not result.success and result.insufficient_context


def test_summarization_agent(deps):
    chunks = search_agent.run("outage", deps).evidence
    result = summarization_agent.run("Summarize the outage", deps, chunks)
    assert result.success
    assert "new certificate" in result.output
    assert result.sources


def test_summarize_raw_text(deps):
    result = summarization_agent.summarize_text("Some long report text.", deps, source="report.txt")
    assert result.success
    assert result.sources == [{"filename": "report.txt", "page": None}]


def test_expert_finder_agent_is_grounded_and_labelled(deps):
    result = expert_finder_agent.run("Who can help with certificate renewal problems?", deps)
    assert result.success
    assert "Marcus Odell" in result.output
    assert "no employee directory" in result.output
    assert "Never invent people" in deps.llm.calls[0]["system_instruction"]
    # The people-focused semantic search found the page naming the owner.
    assert any(e["page"] == 2 for e in result.evidence)


def test_expert_finder_reports_missing_info(deps):
    deps.llm = FakeLLM(f"{INSUFFICIENT_MARKER} No employee is named for payroll.")
    result = expert_finder_agent.run("Who handles payroll?", deps)
    assert not result.success and result.insufficient_context
    assert "payroll" in result.output


def test_compliance_agent_adds_disclaimer(deps):
    result = compliance_agent.run("Can I book a 700 dollar flight without approval?", deps)
    assert result.success
    assert "not legal advice" in result.output


def test_action_agent_uses_findings(deps):
    chunks = search_agent.run("follow-up actions", deps).evidence
    summary = summarization_agent.run("outage", deps, chunks)
    result = action_agent.run("What should we do next?", deps, chunks, [summary])
    assert result.success
    assert "summarization:" in deps.llm.calls[-1]["prompt"]


def test_agent_reports_gemini_failure_without_inventing(deps):
    deps.llm = FakeLLM(error=LLMError("Gemini API error (503): overloaded"))
    result = summarization_agent.run("outage", deps, search_agent.run("outage", deps).evidence)
    assert not result.success
    assert result.output == ""
    assert "503" in result.error


# ---- Semantic router (real embedding model, no keywords) ----

@pytest.mark.parametrize(
    "question, specialist",
    [
        # Two different phrasings per intent.
        ("Who can help me fix certificate renewal problems?", "expert_finder"),
        ("Which colleague has experience renewing TLS certificates?", "expert_finder"),
        ("What should we do to prevent another outage?", "action"),
        ("How can we stop this failure from recurring?", "action"),
        ("Are employees allowed to book expensive international flights?", "compliance"),
        ("Can I expense a business class seat?", "compliance"),
        ("Give me the key points from the sales strategy.", "summarization"),
        ("Recap the incident report for me", "summarization"),
        # "policy" alone must not force compliance: summarizing a policy is summarization.
        ("Summarize the remote work policy", "summarization"),
        ("Does working from another country break the remote work policy?", "compliance"),
    ],
)
def test_semantic_routing_single_intent(embedder, question, specialist):
    decision = route(question, embedder)
    assert decision.selected_agents == ["search", specialist]
    assert not decision.fallback


def test_semantic_routing_multiple_intents(embedder):
    decision = route(
        "Who should I contact about the travel expense policy, and do expenses need approval?", embedder
    )
    assert decision.selected_agents == ["search", "expert_finder", "compliance"]


@pytest.mark.parametrize(
    "question",
    [
        "Find information about the previous infrastructure failure.",
        "When is the datacenter migration due?",
        "What is the weather like?",
    ],
)
def test_low_confidence_falls_back_to_search_only(embedder, question):
    decision = route(question, embedder)
    assert decision.selected_agents == ["search"]
    assert decision.fallback
    assert "summarization" not in decision.selected_agents


def test_routing_explains_its_choice(embedder):
    decision = route("Who can help me fix certificate renewal problems?", embedder)
    scores = {s.agent: s for s in decision.scores}
    assert scores["expert_finder"].selected
    assert "similar to capability" in scores["expert_finder"].reason
    assert not scores["compliance"].selected
    assert 0 < decision.confidence <= 1


# ---- Final answer helpers ----

def test_remap_citations_points_into_merged_evidence():
    a = {"source": "a.pdf", "page": 1, "text": "A", "distance": 0.1}
    b = {"source": "b.pdf", "page": 2, "text": "B", "distance": 0.2}
    merged = [a, b]
    # The agent saw only [b], so its [1] is merged passage [2].
    assert remap_citations("Fact [1].", [b], merged) == "Fact [2]."
    assert remap_citations("Both [1, 2].", [b, a], merged) == "Both [2][1]."


def test_dedupe_sections_removes_repeated_facts():
    text = dedupe_sections([
        ("Summary", "The certificate expired [1].\nA new one was installed [2]."),
        ("Recommended Actions", "The certificate expired [3].\nAdd expiry alerts [2]."),
    ])
    assert text.count("certificate expired") == 1
    assert "Add expiry alerts" in text
    assert "## Recommended Actions" in text


# ---- Orchestrator (LangGraph) ----

def test_orchestrator_single_agent_uses_one_gemini_call(deps):
    q = "How can we stop this outage from recurring?"
    result = run_multi_agent(q, deps)

    assert result["selected_agents"] == ["search", "action"]
    assert [r["agent"] for r in result["agent_results"]] == ["search", "action"]
    assert result["routing"]["method"] == "semantic"
    assert result["success"] is True
    assert result["partial_success"] is False
    assert result["status"] == "complete"
    # One specialist: its answer is the final answer, no extra synthesis call.
    assert result["final_answer"] == REPLIES["action-planning"]
    assert len(deps.llm.calls) == 1
    assert result["usage"]["gemini_calls"] == 1
    assert result["sources"] and result["sources"][0]["filename"] == "dataset.pdf"


def test_orchestrator_search_only_uses_one_gemini_call(deps):
    result = run_multi_agent("Find information about the previous infrastructure failure.", deps)
    assert result["selected_agents"] == ["search"]
    assert result["routing"]["fallback_to_search"] is True
    assert result["status"] == "complete"
    assert result["sources"]
    assert len(deps.llm.calls) == 1  # only the final answer; search itself never calls Gemini
    assert deps.llm.calls[0]["prompt"].count("(source:") <= deps.top_k


def test_multi_intent_synthesis_gets_only_cited_chunks(deps):
    q = "Who should I contact about the travel expense policy, and do expenses need approval?"
    result = run_multi_agent(q, deps)

    assert result["selected_agents"] == ["search", "expert_finder", "compliance"]
    assert result["usage"]["gemini_calls"] == 3  # two specialists + one merge
    synthesis_prompt = deps.llm.calls[-1]["prompt"]
    assert "final-answer writer" in deps.llm.calls[-1]["system_instruction"]
    # Each specialist cited [1] only, so at most 2 distinct passages are re-sent.
    assert synthesis_prompt.count("(source:") <= 2
    for call in deps.llm.calls:
        assert call["prompt"].count("(source:") <= deps.max_evidence


def test_orchestrator_partial_failure(deps):
    deps.llm = FakeLLM(replies(fail_for="compliance-checker"))
    q = "Who should I contact about the travel expense policy, and do expenses need approval?"
    result = run_multi_agent(q, deps)

    by_name = {r["agent"]: r for r in result["agent_results"]}
    assert by_name["expert_finder"]["success"] is True
    assert by_name["compliance"]["success"] is False
    assert "503" in by_name["compliance"]["error"]
    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["status"] == "partial"


def test_orchestrator_synthesis_failure_uses_deduplicated_agent_outputs(deps):
    deps.llm = FakeLLM(replies(fail_for="final-answer writer"))
    q = "Who should I contact about the travel expense policy, and do expenses need approval?"
    result = run_multi_agent(q, deps)

    assert result["status"] == "partial"
    assert "503" in result["synthesis_error"]
    assert "## Relevant Experts" in result["final_answer"]
    assert "## Compliance Check" in result["final_answer"]
    assert result["sources"]  # sources preserved from the agents


def test_orchestrator_insufficient_context_is_honest(deps):
    deps.llm = FakeLLM(f"{INSUFFICIENT_MARKER} No onboarding policy; documents cover travel and incidents.")
    result = run_multi_agent("Summarize the onboarding policy", deps)

    assert result["status"] == "failed"
    assert result["sources"] == []
    assert "do not contain enough information" in result["final_answer"]
    assert "documents cover travel and incidents" in result["final_answer"]


def test_orchestrator_total_failure(deps):
    deps.llm = FakeLLM(error=LLMError("Gemini API error (503): overloaded"))
    result = run_multi_agent("Summarize the incident report", deps)

    assert result["success"] is False
    assert result["partial_success"] is False
    assert result["status"] == "failed"
    assert "Gemini failed" in result["final_answer"]
    assert result["sources"] == []


# ---- API ----

def test_multi_agent_endpoint(client):
    client.post("/documents/upload", files={"file": ("dataset.pdf", DATASET, "application/pdf")})
    app.dependency_overrides[require_llm_client] = lambda: FakeLLM(replies())

    response = client.post(
        "/multi-agent",
        json={"question": "Who should I contact about the travel expense policy, and do expenses need approval?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_agents"] == ["search", "expert_finder", "compliance"]
    assert body["status"] == "complete" and body["success"] is True
    assert body["routing"]["scores"][0]["agent"] == "search"
    assert body["sources"][0]["filename"] == "dataset.pdf"
    assert body["model"] == "fake-gemini"


def test_chat_endpoint_still_works(client):
    client.post("/documents/upload", files={"file": ("dataset.pdf", DATASET, "application/pdf")})
    app.dependency_overrides[require_llm_client] = lambda: FakeLLM("Installed a new certificate [1].")
    assert client.post("/chat", json={"question": "outage?"}).json()["success"] is True
