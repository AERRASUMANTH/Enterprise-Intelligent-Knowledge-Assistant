"""Document upload endpoints."""

from pathlib import PurePath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import settings
from app.schemas import DocumentInfo, DocumentListResponse, UploadResponse
from app.services.document_loader import (
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    validate_file_type,
)
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.indexer import index_document
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(store: VectorStore = Depends(get_vector_store)) -> DocumentListResponse:
    """List the documents currently indexed in ChromaDB."""
    docs = [DocumentInfo(**d) for d in store.list_documents()]
    return DocumentListResponse(
        documents=docs, total_documents=len(docs), total_chunks=sum(d.chunks for d in docs)
    )


# A plain `def` (not async) lets FastAPI run this CPU-heavy work in a thread
# pool, so the server stays responsive while a document is being embedded.
@router.post("/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(..., description="A PDF, DOCX or TXT file"),
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: VectorStore = Depends(get_vector_store),
) -> UploadResponse:
    # Keep only the base name, in case the client sent a full path.
    filename = PurePath((file.filename or "").replace("\\", "/")).name
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File must have a filename.")

    try:
        validate_file_type(filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)  # read one extra byte to detect oversize
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds the {settings.max_upload_size_mb} MB limit.",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    try:
        result = index_document(filename, data, embedder, store)
    except (EmptyDocumentError, UnreadableDocumentError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    return UploadResponse(
        status=result.status, filename=result.filename, chunks_indexed=result.chunks_indexed
    )
