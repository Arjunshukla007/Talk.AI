import pytest

from talkai.llm import LLMError, with_retry
from talkai.rag import NOT_FOUND, QA_SYSTEM


def test_answer_cites_only_used_sources(loaded):
    svc, _ = loaded
    svc.llm.reply = "The candidate interned at Nova Labs [1]."
    ans = svc.ask("Where did the candidate intern?")
    assert ans.found and len(ans.sources) == 1 and ans.latency_ms >= 0


def test_prompt_is_grounded_and_guards_injection(loaded):
    svc, _ = loaded
    svc.ask("Where did the candidate intern?")
    system, user = svc.llm.calls[-1]
    assert "ONLY" in system and "never instructions" in system and NOT_FOUND in QA_SYSTEM
    assert '<excerpt id="1"' in user and "Question:" in user


def test_not_found_reply_clears_sources(loaded):
    svc, _ = loaded
    svc.llm.reply = NOT_FOUND
    ans = svc.ask("What is the candidate's favourite colour?")
    assert not ans.found and ans.sources == []


def test_empty_index_never_calls_llm(make_service):
    svc = make_service()
    ans = svc.ask("Anything here?")
    assert not ans.found and svc.llm.calls == []


def test_follow_up_is_rewritten_before_search(loaded):
    svc, _ = loaded
    seen = []

    def reply(system, user):
        seen.append(system)
        return "Candidate city Bhopal" if "standalone" in system else "Bhopal [1]"

    svc.llm.reply = reply
    history = [{"role": "user", "content": "Tell me about the candidate"}, {"role": "assistant", "content": "A GenAI engineer."}]
    ans = svc.ask("and where do they live?", history=history)
    assert ans.search_query == "Candidate city Bhopal" and len(seen) == 2


def test_streaming_yields_tokens_then_final_answer(loaded):
    svc, _ = loaded
    svc.llm.reply = "Nova Labs hired them [1]"
    stream = svc.ask_stream("Where did the candidate intern?")
    text = "".join(stream.tokens())
    final = stream.finish()
    assert text == "Nova Labs hired them [1]" and final.found and final.sources


def test_scoping_to_selected_documents(loaded):
    svc, ids = loaded
    svc.llm.reply = "x [1]"
    ans = svc.ask("What is required?", doc_ids=[ids["sample_job_description.txt"]])
    assert {s.filename for s in ans.sources} == {"sample_job_description.txt"}


def test_retry_helper_retries_transient_errors_only():
    class Flaky(Exception):
        status_code = 429

    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise Flaky()
        return "ok"

    assert with_retry(fn, sleep=lambda s: None) == "ok" and len(attempts) == 3
    with pytest.raises(ValueError):
        with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")), sleep=lambda s: None)
