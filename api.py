"""FastAPI backend for Talk.AI. Run: uvicorn api:app --reload  (docs at /docs)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from talkai import analysis
from talkai.config import ConfigError, Settings
from talkai.factory import build_service
from talkai.ingestion import DocumentError
from talkai.llm import LLMError
from talkai.rag import RAGService

app = FastAPI(title="Talk.AI API", version="1.0.0",
              description="Retrieval-augmented question answering over resumes and PDFs.")


@lru_cache
def get_service() -> RAGService:
    return build_service(Settings.from_env())


# Map domain errors to clear HTTP responses instead of stack traces.
@app.exception_handler(DocumentError)
async def _doc_error(_: Request, exc: DocumentError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ConfigError)
async def _config_error(_: Request, exc: ConfigError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    doc_ids: Optional[list[str]] = None
    history: list[Message] = []


class DocRequest(BaseModel):
    doc_id: str


class MatchRequest(BaseModel):
    doc_id: str
    job_description: str = Field(min_length=30)


def _text(svc: RAGService, doc_id: str) -> str:
    doc = svc.store.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found. Upload it first.")
    return doc.full_text


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...), svc: RAGService = Depends(get_service)):
    return svc.ingest(await file.read(), file.filename or "upload")


@app.get("/documents")
def documents(svc: RAGService = Depends(get_service)):
    return svc.store.list_documents()


@app.delete("/documents/{doc_id}", status_code=204)
def delete(doc_id: str, svc: RAGService = Depends(get_service)):
    svc.store.delete_document(doc_id)


@app.post("/ask")
def ask(req: AskRequest, svc: RAGService = Depends(get_service)):
    ans = svc.ask(req.question, req.doc_ids, [m.model_dump() for m in req.history])
    return {
        "answer": ans.text, "found": ans.found, "latency_ms": ans.latency_ms, "search_query": ans.search_query,
        "sources": [{"filename": s.filename, "page": s.page, "text": s.text, "relevance": s.score} for s in ans.sources],
    }


@app.post("/summary")
def summary(req: DocRequest, svc: RAGService = Depends(get_service)):
    return {"summary": analysis.summarize(svc.llm, _text(svc, req.doc_id))}


@app.post("/resume")
def resume(req: DocRequest, svc: RAGService = Depends(get_service)):
    return analysis.extract_resume(svc.llm, _text(svc, req.doc_id))


@app.post("/match")
def match(req: MatchRequest, svc: RAGService = Depends(get_service)):
    return analysis.match_job(svc.llm, _text(svc, req.doc_id), req.job_description)
