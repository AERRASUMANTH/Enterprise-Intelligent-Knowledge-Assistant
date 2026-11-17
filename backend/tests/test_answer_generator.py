"""Unit tests for grounded answer generation and the Gemini client wrapper (mocked)."""

import pytest
from google.genai import errors as genai_errors

from app.services.answer_generator import (
    INSUFFICIENT_MARKER,
    NO_DOCUMENTS_ANSWER,
    SYSTEM_INSTRUCTION,
    extract_citations,
    generate_answer,
)
from app.services.indexer import index_document
from app.services.llm import LLMClient, LLMError, LLMNotConfiguredError
from tests.conftest import make_pdf
from tests.fake_llm import FakeLLM

INCIDENT_PDF = make_pdf([
    "Cafeteria menu for the week",
    "The outage was caused by an expired TLS certificate and fixed by installing a new one",
])


@pytest.fixture
def indexed_store(embedder, store):
    index_document("incident.pdf", INCIDENT_PDF, embedder, store)
    return store


def test_grounded_answer_uses_cited_sources(embedder, indexed_store):
    # The incident text is on page 2 and should be retrieved first -> passage [1].
    llm = FakeLLM("It was fixed by installing a new TLS certificate [1].")
    result = generate_answer("How was the outage fixed?", 2, embedder, indexed_store, llm)

    assert result.success is True
    assert result.insufficient_context is False
    assert result.answer == "It was fixed by installing a new TLS certificate [1]."
    assert result.sources == [{"filename": "incident.pdf", "page": 2}]
    assert len(result.evidence) == 2
    assert result.model == "fake-gemini"


def test_prompt_contains_context_and_grounding_rules(embedder, indexed_store):
    llm = FakeLLM("answer [1]")
    generate_answer("How was the outage fixed?", 2, embedder, indexed_store, llm)

    call = llm.calls[0]
    assert "expired TLS certificate" in call["prompt"]
    assert "source: incident.pdf, page 2" in call["prompt"]
    assert "How was the outage fixed?" in call["prompt"]
    assert call["system_instruction"] == SYSTEM_INSTRUCTION
    assert "ONLY" in SYSTEM_INSTRUCTION and INSUFFICIENT_MARKER in SYSTEM_INSTRUCTION


def test_uncited_answer_falls_back_to_all_retrieved_sources(embedder, indexed_store):
    llm = FakeLLM("A new certificate was installed.")
    result = generate_answer("How was the outage fixed?", 2, embedder, indexed_store, llm)

    assert result.success is True
    assert {s["page"] for s in result.sources} == {1, 2}


def test_insufficient_context(embedder, indexed_store):
    # Related to the documents (so chunks are retrieved) but not answerable from them.
    llm = FakeLLM(f"{INSUFFICIENT_MARKER} The documents do not mention the CEO's salary.")
    result = generate_answer("How much did the new TLS certificate cost?", 2, embedder, indexed_store, llm)

    assert result.success is False
    assert result.insufficient_context is True
    assert result.sources == []
    assert "do not contain enough information" in result.answer
    assert result.answer.endswith("do not mention the CEO's salary.")
    assert result.evidence  # evidence is still returned for transparency
    assert all(e["page"] == 2 for e in result.evidence)  # unrelated cafeteria page filtered out


def test_empty_database_skips_llm_call(embedder, store):
    llm = FakeLLM("should not be used")
    result = generate_answer("Anything?", 3, embedder, store, llm)

    assert result.success is False
    assert result.answer == NO_DOCUMENTS_ANSWER
    assert llm.calls == []


def test_llm_error_propagates(embedder, indexed_store):
    llm = FakeLLM(error=LLMError("quota exceeded"))
    with pytest.raises(LLMError):
        generate_answer("How was the outage fixed?", 2, embedder, indexed_store, llm)


@pytest.mark.parametrize(
    "text, n, expected",
    [
        ("Fixed [1].", 3, [1]),
        ("See [2][1] and [2].", 3, [2, 1]),
        ("Combined [1, 3].", 3, [1, 3]),
        ("Out of range [7] and [0].", 3, []),
        ("No citations.", 3, []),
    ],
)
def test_extract_citations(text, n, expected):
    assert extract_citations(text, n) == expected


# ---- LLMClient (Gemini wrapper), with the SDK call mocked ----

def test_missing_api_key_raises():
    with pytest.raises(LLMNotConfiguredError):
        LLMClient(api_key="", model="gemini-test")


def test_placeholder_api_key_raises():
    with pytest.raises(LLMNotConfiguredError, match="GEMINI_API_KEY"):
        LLMClient(api_key="your-gemini-api-key-here", model="gemini-test")


@pytest.mark.parametrize("model", ["", "   ", "your_model_name", "YOUR-MODEL"])
def test_missing_or_placeholder_model_raises(model):
    with pytest.raises(LLMNotConfiguredError, match="GEMINI_MODEL"):
        LLMClient(api_key="fake-key-for-tests", model=model)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def _client_with(monkeypatch, behaviour, max_retries=0):
    client = LLMClient(api_key="fake-key-for-tests", model="gemini-test", max_retries=max_retries)
    client.sleeps = []
    client._sleep = client.sleeps.append  # record backoff instead of waiting
    monkeypatch.setattr(client._client.models, "generate_content", behaviour)
    return client


def _api_error(code):
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "message": f"error {code}", "status": "X"}})


def _failing_then_ok(codes):
    """generate_content stub: raise each code in turn, then succeed."""
    calls = []

    def fake_generate(**kwargs):
        calls.append(1)
        if len(calls) <= len(codes):
            raise _api_error(codes[len(calls) - 1])
        return _FakeResponse("recovered")

    return fake_generate, calls


@pytest.mark.parametrize("code", [429, 503, 504])
def test_temporary_errors_are_retried(monkeypatch, code):
    stub, calls = _failing_then_ok([code, code])
    client = _client_with(monkeypatch, stub, max_retries=2)

    assert client.generate("prompt") == "recovered"
    assert len(calls) == 3
    assert client.sleeps == [1.0, 2.0]  # exponential backoff


def test_retries_give_up_after_max(monkeypatch):
    stub, calls = _failing_then_ok([503, 503, 503])
    client = _client_with(monkeypatch, stub, max_retries=2)

    with pytest.raises(LLMError, match="503"):
        client.generate("prompt")
    assert len(calls) == 3  # 1 attempt + 2 retries


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_errors_are_not_retried(monkeypatch, code):
    stub, calls = _failing_then_ok([code])
    client = _client_with(monkeypatch, stub, max_retries=2)

    with pytest.raises(LLMError):
        client.generate("prompt")
    assert len(calls) == 1
    assert client.sleeps == []


def test_generate_returns_text(monkeypatch):
    captured = {}

    def fake_generate(model, contents, config):
        captured.update(model=model, contents=contents, config=config)
        return _FakeResponse("  hello  ")

    client = _client_with(monkeypatch, fake_generate)
    assert client.generate("prompt", system_instruction="rules") == "hello"
    assert captured["model"] == "gemini-test"
    assert captured["config"].system_instruction == "rules"
    assert captured["config"].max_output_tokens == 1024  # output-token cap is always sent


def test_usage_is_accumulated(monkeypatch):
    class Meta:
        prompt_token_count = 120
        candidates_token_count = 30

    class Resp:
        text = "ok"
        usage_metadata = Meta()

    client = _client_with(monkeypatch, lambda **kwargs: Resp())
    usage = {}
    client.generate("a", usage=usage)
    client.generate("b", usage=usage)
    assert usage == {"gemini_calls": 2, "prompt_tokens": 240, "output_tokens": 60}


def test_api_error_is_wrapped(monkeypatch):
    def fake_generate(**kwargs):
        raise genai_errors.ClientError(
            400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}
        )

    client = _client_with(monkeypatch, fake_generate)
    with pytest.raises(LLMError, match="API key not valid"):
        client.generate("prompt")


def test_unknown_model_error_points_to_setting(monkeypatch):
    def fake_generate(**kwargs):
        raise genai_errors.ClientError(
            404, {"error": {"code": 404, "message": "models/gemini-test is not found", "status": "NOT_FOUND"}}
        )

    client = _client_with(monkeypatch, fake_generate)
    with pytest.raises(LLMError, match="Check GEMINI_MODEL"):
        client.generate("prompt")


@pytest.mark.parametrize("code", [503, 504])
def test_overloaded_error_is_explained(monkeypatch, code):
    def fake_generate(**kwargs):
        raise genai_errors.ServerError(
            code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}}
        )

    client = _client_with(monkeypatch, fake_generate)
    with pytest.raises(LLMError, match="temporarily unavailable or overloaded"):
        client.generate("prompt")


def test_error_message_never_contains_api_key(monkeypatch):
    def fake_generate(**kwargs):
        raise genai_errors.ClientError(
            400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}
        )

    client = _client_with(monkeypatch, fake_generate)
    with pytest.raises(LLMError) as info:
        client.generate("prompt")
    assert "fake-key-for-tests" not in str(info.value)
    assert "Check GEMINI_API_KEY" in str(info.value)


def test_network_error_is_wrapped(monkeypatch):
    def fake_generate(**kwargs):
        raise ConnectionError("no internet")

    client = _client_with(monkeypatch, fake_generate)
    with pytest.raises(LLMError, match="Could not reach Gemini"):
        client.generate("prompt")


def test_empty_response_is_an_error(monkeypatch):
    client = _client_with(monkeypatch, lambda **kwargs: _FakeResponse(None))
    with pytest.raises(LLMError, match="empty response"):
        client.generate("prompt")
