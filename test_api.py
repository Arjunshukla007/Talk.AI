import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from talkai.config import ConfigError
from tests.conftest import SAMPLES


@pytest.fixture
def client(make_service):
    svc = make_service("Nova Labs [1]")
    api.app.dependency_overrides[api.get_service] = lambda: svc
    yield TestClient(api.app), svc
    api.app.dependency_overrides.clear()


def upload(c, name="sample_resume.txt", data=None):
    return c.post("/documents", files={"file": (name, data if data is not None else (SAMPLES / name).read_bytes())})


def test_health(client):
    assert client[0].get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete(client):
    c, _ = client
    doc = upload(c)
    assert doc.status_code == 201 and doc.json()["chunks"] > 0
    assert len(c.get("/documents").json()) == 1
    res = c.post("/ask", json={"question": "Where did they intern?", "doc_ids": [doc.json()["doc_id"]]}).json()
    assert res["found"] and res["sources"][0]["filename"] == "sample_resume.txt"
    assert c.delete(f"/documents/{doc.json()['doc_id']}").status_code == 204
    assert c.get("/documents").json() == []


def test_bad_uploads_return_400(client):
    c, _ = client
    assert upload(c, "x.docx", b"data").status_code == 400
    assert upload(c, "empty.txt", b"").status_code == 400


def test_unknown_document_is_404(client):
    assert client[0].post("/summary", json={"doc_id": "nope"}).status_code == 404


def test_resume_and_match_endpoints(client):
    c, svc = client
    doc_id = upload(c).json()["doc_id"]
    svc.llm.reply = json.dumps({"skills": ["Python"], "score": 72, "summary": "ok"})
    assert c.post("/resume", json={"doc_id": doc_id}).json()["skills"] == ["Python"]
    jd = (SAMPLES / "sample_job_description.txt").read_text()
    assert c.post("/match", json={"doc_id": doc_id, "job_description": jd}).json()["score"] == 72
    assert c.post("/match", json={"doc_id": doc_id, "job_description": "short"}).status_code == 422


def test_missing_key_is_503(client):
    c, svc = client

    def boom(*a, **k):
        raise ConfigError("OPENAI_API_KEY is missing.")

    svc.ask = boom
    res = c.post("/ask", json={"question": "hello there"})
    assert res.status_code == 503 and "OPENAI_API_KEY" in res.json()["detail"]
