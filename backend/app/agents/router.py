"""Semantic agent router.

Each agent is described by a few example requests ("capability prototypes").
The question is embedded with the SAME Sentence Transformers model used for
retrieval and compared (cosine similarity) with every prototype. An agent's
score is its best-matching prototype.

- The whole question and each of its clauses ("... and what should we do next?")
  are scored separately. For each one, a specialist is selected when its score is
  >= `routing_threshold` AND within `routing_margin` of that clause's best agent.
  The union is used, so a multi-intent question selects several agents while weak
  side-matches are ignored.
- The Search Agent always runs, because it supplies the evidence; when no
  specialist is confident, Search alone is the safe fallback.
"""

import re
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.services.embeddings import EmbeddingService

SPECIALISTS = ["summarization", "expert_finder", "compliance", "action"]

CAPABILITIES: dict[str, list[str]] = {
    "search": [
        "Find information about a topic in the documents",
        "What happened during the incident?",
        "Look up details about a project or event",
        "What does the report say about this?",
        "Show me documents related to this subject",
    ],
    "summarization": [
        "Summarize this document",
        "Give me the key points",
        "Provide a brief overview of the report",
        "What are the main takeaways and highlights?",
        "Condense this information into a short summary",
        "Summarize this policy",
        "Give me the key points of this policy document",
        "Give me an overview of these guidelines",
    ],
    "expert_finder": [
        "Who can help me with this problem?",
        "Who is the expert on this topic?",
        "Which employee is responsible for this?",
        "Who should I contact about this issue?",
        "Find a person with skills and experience in this area",
        "Who owns this system or process?",
        "Which engineer knows the most about this technology?",
        "Who is in charge of this project?",
        "Which colleague has experience with this?",
    ],
    "compliance": [
        "Is this allowed under company policy?",
        "Does this violate any company rules?",
        "Are employees permitted to do this?",
        "What are the policy requirements and approvals needed?",
        "Is this compliant with the regulations?",
        "Can I expense this purchase?",
        "Is this within the allowed spending limits?",
    ],
    "action": [
        "What should we do next?",
        "What steps should we take to prevent this from happening again?",
        "Recommend follow-up actions with owners and deadlines",
        "Create an action plan to resolve this",
        "How do we avoid this problem in the future?",
    ],
}

_prototype_cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}


@dataclass
class RouteScore:
    agent: str
    score: float
    selected: bool
    reason: str


@dataclass
class RoutingDecision:
    selected_agents: list[str]
    confidence: float          # best specialist score (0..1)
    fallback: bool             # True when no specialist was confident
    scores: list[RouteScore]


def _prototypes(embedder: EmbeddingService):
    """Embed the capability prototypes once per embedding model."""
    key = embedder.model_name
    if key not in _prototype_cache:
        agents, texts = [], []
        for agent, examples in CAPABILITIES.items():
            agents += [agent] * len(examples)
            texts += examples
        _prototype_cache[key] = (agents, texts, np.array(embedder.embed_texts(texts)))
    return _prototype_cache[key]


def split_clauses(question: str) -> list[str]:
    """Split a multi-part question into clauses (only reasonably long ones)."""
    parts = re.split(r"[?;]|,?\s+\band\b\s+|,?\s+\balso\b\s+", question)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def route(
    question: str,
    embedder: EmbeddingService,
    threshold: float | None = None,
    margin: float | None = None,
) -> RoutingDecision:
    threshold = settings.routing_threshold if threshold is None else threshold
    margin = settings.routing_margin if margin is None else margin
    agents, texts, proto = _prototypes(embedder)

    queries = [question] + [c for c in split_clauses(question) if c != question.strip()]
    q = np.array(embedder.embed_texts(queries))
    sims = q @ proto.T  # vectors are normalized -> cosine similarity

    best: dict[str, tuple[float, str, str]] = {}   # overall best per agent
    chosen: set[str] = set()
    for qi, query in enumerate(queries):
        per_agent: dict[str, tuple[float, str]] = {}
        for pi, agent in enumerate(agents):
            s = float(sims[qi, pi])
            if agent not in per_agent or s > per_agent[agent][0]:
                per_agent[agent] = (s, texts[pi])
            if agent not in best or s > best[agent][0]:
                best[agent] = (s, texts[pi], query)
        top = max(per_agent[a][0] for a in SPECIALISTS)
        chosen |= {
            a for a in SPECIALISTS
            if per_agent[a][0] >= threshold and per_agent[a][0] >= top - margin
        }

    selected = [a for a in SPECIALISTS if a in chosen]
    confidence = max(best[a][0] for a in SPECIALISTS)

    scores = []
    for agent in ["search", *SPECIALISTS]:
        s, proto_text, query = best[agent]
        if agent == "search":
            reason = "Always runs: retrieves the evidence used by every other agent."
            is_selected = True
        elif agent in selected:
            reason = f"'{query}' is similar to capability '{proto_text}' ({s:.2f})."
            is_selected = True
        elif s < threshold:
            reason = f"Best match '{proto_text}' scored {s:.2f}, below threshold {threshold:.2f}."
            is_selected = False
        else:
            reason = f"Best match '{proto_text}' scored {s:.2f}, but another agent fit the intent better."
            is_selected = False
        scores.append(RouteScore(agent, round(s, 3), is_selected, reason))

    return RoutingDecision(
        selected_agents=["search", *selected],
        confidence=round(confidence, 3),
        fallback=not selected,
        scores=scores,
    )
