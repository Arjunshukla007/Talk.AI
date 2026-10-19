import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from talkai.config import Settings  # noqa: E402
from talkai.rag import RAGService  # noqa: E402
from talkai.store import VectorStore  # noqa: E402

SAMPLES = Path(__file__).parent.parent / "sample_data"


class FakeLLM:
    """Scripted LLM so tests never call a paid API. `reply` may be a string or a function(system, user)."""

    def __init__(self, reply="OK [1]"):
        self.reply, self.calls = reply, []

    def _answer(self, system, user):
        self.calls.append((system, user))
        return self.reply(system, user) if callable(self.reply) else self.reply

    def complete(self, system, user, max_tokens=900):
        return self._answer(system, user)

    def stream(self, system, user, max_tokens=900):
        yield from re.findall(r"\S+\s*", self._answer(system, user))  # token-like pieces


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), openai_api_key="test-key", embedding_backend="local",
                   chroma_dir=str(tmp_path / "db"), chunk_size=200, chunk_overlap=30)


@pytest.fixture
def store(settings):
    return VectorStore(settings)


@pytest.fixture
def make_service(settings, store):
    def _make(reply="OK [1]"):
        return RAGService(store, FakeLLM(reply), settings)
    return _make


@pytest.fixture
def loaded(make_service):
    """A service with the sample resume and job description already indexed."""
    svc = make_service()
    ids = {}
    for name in ("sample_resume.txt", "sample_job_description.txt"):
        ids[name] = svc.ingest((SAMPLES / name).read_bytes(), name)["doc_id"]
    return svc, ids
