"""API tests for POST /chat with Gemini mocked."""

import pytest

from app.api import chat as chat_module
from app.api.chat import require_llm_client
from app.main import app
from app.services.answer_generator import INSUFFICIENT_MARKER
from app.services.llm import LLMError, LLMNotConfiguredError
from tests.conftest import make_pdf
from tests.fake_llm import FakeLLM


@pytest.fixture
def use_llm():
    """Install a FakeLLM for the /chat endpoint."""
    def _install(llm):
        app.dependency_overrides[require_llm_client] = lambda: llm
        return llm
    return _install


def upload_incident(client):
    pdf = make_pdf([
        "Office parking rules",
        "Root cause: expired TLS certificate. Resolved by installing a new certificate.",
    ])
    client.post("/documents/upload", files={"file": ("incident_report.pdf", pdf, "application/pdf")})


def test_chat_grounded_answer(client, use_llm):
    upload_incident(client)
    use_llm(FakeLLM("It was resolved by installing a new certificate [1]."))

    response = client.post("/chat", json={"question": "How was the previous server outage resolved?"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["answer"] == "It was resolved by installing a new certificate [1]."
    # Sources come from the actual retrieval result, not hardcoded values.
    assert body["sources"] == [{"filename": "incident_report.pdf", "page": 2}]
    assert body["evidence"][0]["source"] == "incident_report.pdf"
    assert "certificate" in body["evidence"][0]["text"]
    assert body["metadata"]["model"] == "fake-gemini"
    assert body["metadata"]["insufficient_context"] is False
    assert body["metadata"]["retrieved_chunks"] == len(body["evidence"])


def test_chat_respects_top_k(client, use_llm):
    upload_incident(client)
    use_llm(FakeLLM("answer [1]"))
    body = client.post("/chat", json={"question": "outage", "top_k": 1}).json()
    assert body["metadata"]["top_k"] == 1
    assert len(body["evidence"]) == 1


def test_chat_insufficient_context(client, use_llm):
    upload_incident(client)
    use_llm(FakeLLM(f"{INSUFFICIENT_MARKER} No salary information is present."))

    body = client.post("/chat", json={"question": "How much did the new TLS certificate cost?"}).json()

    assert body["success"] is False
    assert body["metadata"]["insufficient_context"] is True
    assert body["sources"] == []


def test_chat_empty_database(client, use_llm):
    llm = use_llm(FakeLLM("unused"))
    response = client.post("/chat", json={"question": "Anything?"})

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["evidence"] == []
    assert llm.calls == []


def test_chat_gemini_failure_returns_502(client, use_llm):
    upload_incident(client)
    use_llm(FakeLLM(error=LLMError("Gemini API error (429): quota exceeded")))

    response = client.post("/chat", json={"question": "outage"})
    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]


def test_chat_missing_api_key_returns_503(client, monkeypatch):
    def no_key():
        raise LLMNotConfiguredError("GEMINI_API_KEY is not set.")

    monkeypatch.setattr(chat_module, "get_llm_client", no_key)
    response = client.post("/chat", json={"question": "outage"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_chat_placeholder_model_returns_503(client, monkeypatch):
    from app.services.llm import LLMClient

    monkeypatch.setattr(
        chat_module, "get_llm_client",
        lambda: LLMClient(api_key="fake-key-for-tests", model="your_model_name"),
    )
    response = client.post("/chat", json={"question": "outage"})

    assert response.status_code == 503
    assert "GEMINI_MODEL" in response.json()["detail"]
    assert "fake-key-for-tests" not in response.text


def test_chat_validation(client, use_llm):
    use_llm(FakeLLM("unused"))
    assert client.post("/chat", json={"question": "  "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
