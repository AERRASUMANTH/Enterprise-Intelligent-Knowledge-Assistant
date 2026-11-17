import pytest

from app.services.document_loader import (
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    load_document,
)
from tests.conftest import make_docx, make_pdf


def test_load_txt():
    pages = load_document("notes.txt", "Server rebooted at 9am.".encode("utf-8"))
    assert len(pages) == 1
    assert pages[0].text == "Server rebooted at 9am."
    assert pages[0].page is None


def test_load_txt_non_utf8():
    pages = load_document("legacy.txt", "Café".encode("latin-1"))
    assert pages[0].text == "Café"


def test_load_pdf_keeps_page_numbers():
    data = make_pdf(["First page about networking", "Second page about storage"])
    pages = load_document("report.pdf", data)

    assert [p.page for p in pages] == [1, 2]
    assert "networking" in pages[0].text
    assert "storage" in pages[1].text


def test_load_docx():
    data = make_docx(["Quarterly compliance review", "All controls passed."])
    pages = load_document("review.docx", data)

    assert "Quarterly compliance review" in pages[0].text
    assert "All controls passed." in pages[0].text


def test_extension_is_case_insensitive():
    assert load_document("NOTES.TXT", b"hello")[0].text == "hello"


@pytest.mark.parametrize("filename", ["image.png", "script.exe", "no_extension"])
def test_unsupported_types_rejected(filename):
    with pytest.raises(UnsupportedFileTypeError):
        load_document(filename, b"data")


def test_empty_txt_rejected():
    with pytest.raises(EmptyDocumentError):
        load_document("empty.txt", b"   \n  ")


def test_corrupt_pdf_rejected():
    with pytest.raises(UnreadableDocumentError):
        load_document("broken.pdf", b"this is not a pdf")


def test_corrupt_docx_rejected():
    with pytest.raises(UnreadableDocumentError):
        load_document("broken.docx", b"this is not a docx")
