"""Document ingestion for Talk.AI: validation, text extraction and chunking.

Supports PDF (via PyMuPDF) and TXT files. Page numbers are kept for PDFs so
answers can cite where information came from.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_MB = 15


class DocumentError(Exception):
    """Raised for problems with the uploaded document (shown to the user)."""


@dataclass
class PageText:
    page: Optional[int]  # 1-based page number; None for TXT files
    text: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: Optional[int]


@dataclass
class Document:
    filename: str
    doc_id: str  # SHA-256 of the file bytes, used to cache the vector index
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def _clean(text: str) -> str:
    """Normalise whitespace while keeping line breaks (useful for resumes)."""
    text = text.replace("\x00", "").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> list[PageText]:
    import fitz  # PyMuPDF; imported lazily so TXT-only use stays light

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentError(
            "This PDF could not be opened. It may be corrupted or not a real PDF."
        ) from exc
    if pdf.needs_pass:
        raise DocumentError("This PDF is password-protected. Remove the password and try again.")
    pages = []
    for index, page in enumerate(pdf, start=1):
        text = _clean(page.get_text("text"))
        if text:
            pages.append(PageText(page=index, text=text))
    pdf.close()
    return pages


def _extract_txt(data: bytes) -> list[PageText]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")  # permissive fallback for old files
    text = _clean(text)
    return [PageText(page=None, text=text)] if text else []


def load_document(data: bytes, filename: str) -> Document:
    """Validate an upload and return its extracted text."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type '{ext or 'unknown'}'. Upload a PDF or TXT file.")
    if not data:
        raise DocumentError("The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise DocumentError(f"The file is larger than {MAX_FILE_MB} MB. Upload a smaller file.")

    pages = _extract_pdf(data) if ext == ".pdf" else _extract_txt(data)
    if not pages:
        raise DocumentError(
            "No readable text was found. If this is a scanned PDF, it needs OCR first "
            "(Talk.AI reads text-based PDFs only)."
        )
    return Document(filename=filename, doc_id=hashlib.sha256(data).hexdigest(), pages=pages)


def split_into_chunks(doc: Document, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    """Split each page into overlapping chunks, remembering the source page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in doc.pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(chunk_id=f"c{len(chunks)}", text=piece, page=page.page))
    return chunks
