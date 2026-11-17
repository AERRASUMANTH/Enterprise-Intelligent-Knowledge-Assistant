"""Integration test against the REAL index in chroma_data/.

It only queries data that was uploaded earlier: it never uploads, writes or
deletes anything. It is skipped automatically if the dataset isn't indexed.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.vector_store import VectorStore, get_vector_store

DATASET = "Enterprise Knowledge Dataset.pdf"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_store() -> VectorStore:
    # Check the folder first so a fresh clone doesn't create an empty database.
    if not settings.chroma_persist_dir.exists():
        pytest.skip("No local chroma_data/ folder; upload the dataset first.")
    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection_name)
    if store.get_document_hash(DATASET) is None:
        pytest.skip(f"'{DATASET}' is not indexed; upload it first.")
    return store


@pytest.fixture
def real_client(embedder, real_store):
    from app.services.embeddings import get_embedding_service

    app.dependency_overrides[get_embedding_service] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: real_store
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_retrieve_from_indexed_dataset(real_client):
    question = "How was the previous server failure resolved?"
    response = real_client.post("/rag/retrieve", json={"question": question, "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == question

    results = body["results"]
    assert len(results) == 3
    assert all(r["source"] == DATASET for r in results)
    assert all(isinstance(r["page"], int) for r in results)  # PDF -> real page numbers

    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)  # most relevant first
    assert all(0 <= d <= 2 for d in distances)  # valid cosine distance range

    # The best match should be the IT incident (resolved via an expired certificate).
    assert "certificate" in results[0]["text"].lower()
