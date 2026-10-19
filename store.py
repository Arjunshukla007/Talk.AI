"""Persistent vector store (ChromaDB) plus a small JSON registry of uploaded documents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb

from talkai.config import Settings
from talkai.embeddings import build_embedding_function
from talkai.ingestion import Document, PageText, split_into_chunks


@dataclass
class Hit:
    chunk_id: str
    text: str
    page: Optional[int]
    doc_id: str
    filename: str
    score: float = 0.0  # 0-1 relevance (set by the retrieval layer)


def _where(doc_ids: Optional[list[str]]):
    if not doc_ids:
        return None
    return {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}


class VectorStore:
    def __init__(self, settings: Settings, embedding_function=None):
        self.settings = settings
        self.root = Path(settings.chroma_dir)
        (self.root / "documents").mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.root))
        ef = embedding_function or build_embedding_function(settings)
        # One collection per embedding backend, because vector sizes differ between models.
        self.collection = client.get_or_create_collection(
            name=f"talkai_{settings.resolved_embedding}", embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )

    # ---- documents ----
    def _registry(self, doc_id: str) -> Path:
        return self.root / "documents" / f"{doc_id}.json"

    def has_document(self, doc_id: str) -> bool:
        return bool(self.collection.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add_document(self, doc: Document) -> int:
        chunks = split_into_chunks(doc, self.settings.chunk_size, self.settings.chunk_overlap)
        if not self.has_document(doc.doc_id):
            for i in range(0, len(chunks), 64):  # batch to respect embedding API limits
                batch = chunks[i : i + 64]
                self.collection.add(
                    ids=[f"{doc.doc_id[:16]}_{c.chunk_id}" for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[{"doc_id": doc.doc_id, "filename": doc.filename, "page": c.page or 0} for c in batch],
                )
        self._registry(doc.doc_id).write_text(
            json.dumps({
                "doc_id": doc.doc_id, "filename": doc.filename, "chunks": len(chunks),
                "pages": [{"page": p.page, "text": p.text} for p in doc.pages],
            }),
            encoding="utf-8",
        )
        return len(chunks)

    def get_document(self, doc_id: str) -> Optional[Document]:
        path = self._registry(doc_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Document(raw["filename"], raw["doc_id"], [PageText(p["page"], p["text"]) for p in raw["pages"]])

    def list_documents(self) -> list[dict]:
        docs = []
        for path in sorted((self.root / "documents").glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if self.has_document(raw["doc_id"]):
                words = sum(len(p["text"].split()) for p in raw["pages"])
                docs.append({"doc_id": raw["doc_id"], "filename": raw["filename"], "chunks": raw["chunks"],
                             "pages": len(raw["pages"]) if raw["pages"][0]["page"] else None, "words": words})
        return docs

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
        self._registry(doc_id).unlink(missing_ok=True)

    # ---- search ----
    @staticmethod
    def _to_hit(cid: str, text: str, meta: dict, score: float = 0.0) -> Hit:
        return Hit(cid, text, meta.get("page") or None, meta["doc_id"], meta["filename"], score)

    def vector_search(self, query: str, k: int, doc_ids: Optional[list[str]] = None) -> list[Hit]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, total), where=_where(doc_ids))
        return [
            self._to_hit(i, t, m, round(1 - d, 4))
            for i, t, m, d in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def all_chunks(self, doc_ids: Optional[list[str]] = None) -> list[Hit]:
        res = self.collection.get(where=_where(doc_ids), include=["documents", "metadatas"])
        return [self._to_hit(i, t, m) for i, t, m in zip(res["ids"], res["documents"], res["metadatas"])]
