"""Smoke test of the Streamlit UI with a scripted LLM (no network, no API key)."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import talkai.factory as factory
from tests.conftest import FakeLLM

APP = str(Path(__file__).parent.parent / "app.py")


def scripted(system, user):
    if "recruiter" in system:
        return json.dumps({"score": 64, "summary": "Decent fit.", "matched_skills": ["Python"], "missing_skills": ["Docker"]})
    if "Extract data" in system:
        return json.dumps({"name": "Aarav", "skills": ["Python"], "education": [], "experience": [], "projects": []})
    if "standalone" in system:
        return "candidate internship"
    if "Summarise" in system:
        return "A short summary."
    return "They interned at Nova Labs [1]."


@pytest.fixture
def app(monkeypatch, tmp_path):
    for k, v in {"OPENAI_API_KEY": "test", "EMBEDDING_BACKEND": "local", "CHROMA_DIR": str(tmp_path / "db")}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(factory, "build_llm", lambda s: FakeLLM(scripted))
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.sidebar.button if b.label == "Load sample files").click().run()
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_chat_with_sources_and_follow_up(app):
    app.chat_input[0].set_value("Where did they intern?").run()
    app.chat_input[0].set_value("and for how long?").run()
    assert not app.exception and len(app.chat_message) == 4
    assert "Nova Labs" in app.chat_message[1].markdown[0].value


def test_analysis_tabs_run(app):
    click(app, "Summarise")
    click(app, "Extract profile")
    app.text_area[0].set_value("Junior GenAI engineer. Requires Python, LangChain, Docker and AWS experience.").run()
    click(app, "Check match")
    assert not app.exception and not app.error
    assert app.metric[0].value == "64/100"


def test_missing_key_shows_setup_help(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "db"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
