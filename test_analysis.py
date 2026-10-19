import json

import pytest

from talkai import analysis
from talkai.llm import LLMError
from tests.conftest import FakeLLM


def test_resume_extraction_validates_and_cleans():
    payload = {"name": "Aarav", "skills": ["Python", " "], "education": ["B.Tech"],
               "experience": [{"role": "Intern", "company": "Nova Labs"}], "projects": []}
    profile = analysis.extract_resume(FakeLLM("```json\n" + json.dumps(payload) + "\n```"), "resume text")
    assert profile.skills == ["Python"] and profile.experience == ["Intern - Nova Labs"]


def test_invalid_json_is_repaired_once():
    replies = iter(["not json", '{"score": 80, "summary": "Good fit"}'])
    report = analysis.match_job(FakeLLM(lambda s, u: next(replies)), "resume", "job")
    assert report.score == 80


def test_persistently_invalid_json_raises_clear_error():
    with pytest.raises(LLMError, match="unexpected format"):
        analysis.extract_resume(FakeLLM("still not json"), "resume")


def test_match_score_must_be_in_range():
    with pytest.raises(LLMError):
        analysis.match_job(FakeLLM('{"score": 140}'), "r", "j")


def test_long_documents_are_summarised_in_sections():
    llm = FakeLLM("summary")
    out = analysis.summarize(llm, "word " * 10000)  # ~50k chars -> several parts
    assert len(llm.calls) > 2 and "summarised in sections" in out
