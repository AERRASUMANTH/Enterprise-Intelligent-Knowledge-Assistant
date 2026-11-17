"""Text chunker: splits page text into overlapping chunks.

Chunks never span two pages, so every chunk keeps an exact page number.
Chunk boundaries prefer whitespace so words are not cut in half.
"""

from dataclasses import dataclass

from app.services.document_loader import PageText


@dataclass
class Chunk:
    text: str
    source: str           # original filename
    page: int | None      # page number, if known
    chunk_index: int      # position of the chunk within the whole document


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one string into overlapping pieces of at most `chunk_size` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Move the end back to the last whitespace, unless we're at the end
        # or that would make the chunk too small.
        if end < len(text):
            space = text.rfind(" ", start, end)
            newline = text.rfind("\n", start, end)
            cut = max(space, newline)
            if cut > start + chunk_size // 2:
                end = cut

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= len(text):
            break
        # Step forward, keeping about `chunk_overlap` characters of context,
        # and always make progress.
        next_start = max(end - chunk_overlap, start + 1)
        # If that lands mid-word, move forward to the start of the next word.
        if not text[next_start - 1].isspace():
            boundary = min(
                (i for i in (text.find(" ", next_start, end), text.find("\n", next_start, end)) if i != -1),
                default=-1,
            )
            if boundary != -1:
                next_start = boundary + 1
        start = next_start

    return pieces


def chunk_pages(
    pages: list[PageText], source: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every page of a document, keeping filename and page metadata."""
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, source=source, page=page.page, chunk_index=len(chunks))
            )
    return chunks
