import fitz
import pytest

from talkai.ingestion import DocumentError, load_document, split_into_chunks


def make_pdf(pages: list[str]) -> bytes:
    pdf = fitz.open()
    for text in pages:
        pdf.new_page().insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def test_loads_txt():
    doc = load_document(b"Hello world.\n\n\n\nSecond paragraph.", "a.txt")
    assert doc.pages[0].page is None
    assert "Second paragraph." in doc.full_text


def test_rejects_unsupported_extension():
    with pytest.raises(DocumentError, match="Unsupported"):
        load_document(b"data", "notes.docx")


def test_rejects_empty_file_and_blank_text():
    with pytest.raises(DocumentError, match="empty"):
        load_document(b"", "a.txt")
    with pytest.raises(DocumentError, match="No readable text"):
        load_document(b"   \n  ", "a.txt")


def test_rejects_corrupt_pdf():
    with pytest.raises(DocumentError):
        load_document(b"this is not a pdf", "fake.pdf")


def test_pdf_keeps_page_numbers():
    doc = load_document(make_pdf(["Page one text", "Page two text"]), "r.pdf")
    assert [p.page for p in doc.pages] == [1, 2]
    chunks = split_into_chunks(doc)
    assert {c.page for c in chunks} == {1, 2}


def test_long_text_is_split_with_overlap():
    doc = load_document(("Sentence about Python. " * 200).encode(), "long.txt")
    chunks = split_into_chunks(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
