"""Builds the service objects. Kept in one place so the UI, API and tests share wiring."""
from __future__ import annotations

from talkai.config import Settings
from talkai.llm import build_llm
from talkai.rag import RAGService
from talkai.store import VectorStore


def build_service(settings: Settings) -> RAGService:
    settings.validate_llm()
    return RAGService(VectorStore(settings), build_llm(settings), settings)
