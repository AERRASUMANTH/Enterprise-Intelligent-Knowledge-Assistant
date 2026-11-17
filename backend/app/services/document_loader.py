"""Document loader: extracts text from PDF, DOCX and TXT files.

Each document becomes a list of `PageText` items. PDFs keep their real page
numbers. DOCX and TXT files have no reliable page concept, so `page` is None.
"""

import io
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentError(Exception):
    """Base error for document loading problems."""


class UnsupportedFileTypeError(DocumentError):
    """The file extension is not one we can read."""


class EmptyDocumentError(DocumentError):
    """The file was read but contains no extractable text."""


class UnreadableDocumentError(DocumentError):
    """The file is corrupt or could not be parsed."""


@dataclass
class PageText:
    text: str
    page: int | None  # 1-based page number, or None when not available


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_file_type(filename: str) -> str:
    """Return the lowercase extension, or raise if the type is unsupported."""
    ext = get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or '(none)'}'. Allowed types: {allowed}"
        )
    return ext


def _load_pdf(data: bytes) -> list[PageText]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        pages.append(PageText(text=page.extract_text() or "", page=number))
    return pages


def _load_docx(data: bytes) -> list[PageText]:
    doc = DocxDocument(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    # Include table contents, which are not part of doc.paragraphs.
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return [PageText(text="\n".join(parts), page=None)]


def _load_txt(data: bytes) -> list[PageText]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")  # never fails; fine for legacy text files
    return [PageText(text=text, page=None)]


_LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}


def load_document(filename: str, data: bytes) -> list[PageText]:
    """Extract text from a document given its filename and raw bytes.

    Returns only pages that contain text. Raises a DocumentError subclass on
    unsupported, unreadable or empty documents.
    """
    ext = validate_file_type(filename)

    try:
        pages = _LOADERS[ext](data)
    except Exception as exc:  # parsers raise many different error types
        raise UnreadableDocumentError(f"Could not read '{filename}': {exc}") from exc

    pages = [PageText(text=p.text.strip(), page=p.page) for p in pages]
    pages = [p for p in pages if p.text]
    if not pages:
        raise EmptyDocumentError(f"No readable text found in '{filename}'.")
    return pages


def load_document_from_path(path: str | Path) -> list[PageText]:
    """Convenience wrapper for loading a file from disk."""
    path = Path(path)
    return load_document(path.name, path.read_bytes())
