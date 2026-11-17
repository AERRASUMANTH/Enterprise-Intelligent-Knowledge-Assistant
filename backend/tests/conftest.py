"""Shared test fixtures.

Every test that touches ChromaDB gets its own temporary directory, so tests
never read or modify the real `chroma_data/` folder.
"""

import io

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store


@pytest.fixture(scope="session")
def embedder() -> EmbeddingService:
    """Load the real embedding model once for the whole test session."""
    return EmbeddingService(settings.embedding_model)


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "test_collection")


@pytest.fixture
def client(embedder, store):
    app.dependency_overrides[get_embedding_service] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---- Sample document builders ----

def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of text per page."""
    n = len(pages)
    font_id = 3 + 2 * n
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * i} 0 R" for i in range(n)), n
        ),
    ]
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def make_docx(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
