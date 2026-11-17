from app.config import settings
from tests.conftest import make_docx, make_pdf


def upload(client, filename, data, content_type="application/octet-stream"):
    return client.post("/documents/upload", files={"file": (filename, data, content_type)})


def test_health_still_works(client):
    assert client.get("/health").json()["status"] == "ok"


def test_list_documents(client):
    assert client.get("/documents").json() == {"documents": [], "total_documents": 0, "total_chunks": 0}
    upload(client, "a.pdf", make_pdf(["Page one", "Page two"]))
    upload(client, "b.txt", b"hello")

    body = client.get("/documents").json()
    assert body["total_documents"] == 2 and body["total_chunks"] == 3
    assert body["documents"][0] == {"filename": "a.pdf", "chunks": 2, "pages": [1, 2]}
    assert body["documents"][1] == {"filename": "b.txt", "chunks": 1, "pages": []}


def test_cors_allows_frontend_origin(client):
    response = client.options(
        "/chat",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_unexpected_errors_return_json(client, store, monkeypatch):
    def boom():
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(store, "list_documents", boom)
    no_raise = client.__class__(client.app, raise_server_exceptions=False)
    response = no_raise.get("/documents")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error. See server logs."}
    assert "secret internal detail" not in response.text


def test_upload_txt(client):
    response = upload(client, "incident.txt", b"The server failure was resolved by replacing the power supply.")
    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "filename": "incident.txt", "chunks_indexed": 1}


def test_upload_pdf_and_docx(client):
    assert upload(client, "a.pdf", make_pdf(["Page one", "Page two"])).json()["chunks_indexed"] == 2
    assert upload(client, "b.docx", make_docx(["Hello from Word"])).json()["chunks_indexed"] == 1


def test_duplicate_upload(client):
    upload(client, "incident.txt", b"Some incident text.")
    response = upload(client, "incident.txt", b"Some incident text.")
    assert response.json()["status"] == "unchanged"


def test_upload_strips_client_path(client):
    response = upload(client, "C:\\Users\\me\\notes.txt", b"hello")
    assert response.json()["filename"] == "notes.txt"


def test_unsupported_type_rejected(client):
    assert upload(client, "photo.png", b"\x89PNG").status_code == 415


def test_empty_file_rejected(client):
    assert upload(client, "empty.txt", b"").status_code == 400


def test_unreadable_file_rejected(client):
    assert upload(client, "broken.pdf", b"not really a pdf").status_code == 422


def test_oversized_file_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    assert upload(client, "big.txt", b"a" * (1024 * 1024 + 1)).status_code == 413


def test_retrieve_on_empty_database(client):
    response = client.post("/rag/retrieve", json={"question": "Anything?", "top_k": 3})
    assert response.status_code == 200
    assert response.json() == {"question": "Anything?", "results": []}


def test_upload_then_retrieve(client):
    upload(client, "incident_report.pdf", make_pdf([
        "Quarterly budget overview",
        "The server failure was resolved by restoring from the nightly backup",
    ]))
    upload(client, "menu.txt", b"Friday lunch is pizza.")

    response = client.post(
        "/rag/retrieve",
        json={"question": "How was the previous server failure resolved?", "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How was the previous server failure resolved?"
    assert 1 <= len(body["results"]) <= 2
    # The unrelated lunch menu is filtered out by the relevance threshold.
    assert "menu.txt" not in {r["source"] for r in body["results"]}

    top = body["results"][0]
    assert top["source"] == "incident_report.pdf"
    assert top["page"] == 2
    assert "backup" in top["text"]
    assert isinstance(top["distance"], float)


def test_retrieve_default_top_k(client):
    response = client.post("/rag/retrieve", json={"question": "hello"})
    assert response.status_code == 200


def test_retrieve_validation(client):
    assert client.post("/rag/retrieve", json={"question": "   "}).status_code == 422
    assert client.post("/rag/retrieve", json={"question": "hi", "top_k": 0}).status_code == 422
    too_many = settings.max_top_k + 1
    assert client.post("/rag/retrieve", json={"question": "hi", "top_k": too_many}).status_code == 422
