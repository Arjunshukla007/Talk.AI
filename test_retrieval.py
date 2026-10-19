from talkai.retrieval import bm25_search, hybrid_search


def test_documents_are_persisted_and_listed(loaded, settings):
    svc, ids = loaded
    names = {d["filename"] for d in svc.store.list_documents()}
    assert names == {"sample_resume.txt", "sample_job_description.txt"}
    from talkai.store import VectorStore
    assert len(VectorStore(settings).list_documents()) == 2  # survives a restart


def test_ingest_is_idempotent(loaded):
    svc, ids = loaded
    before = svc.store.collection.count()
    again = svc.ingest((__import__("pathlib").Path("sample_data/sample_resume.txt")).read_bytes(), "sample_resume.txt")
    assert again["already_indexed"] and svc.store.collection.count() == before


def test_bm25_finds_exact_rare_term(loaded):
    svc, _ = loaded
    top = bm25_search(svc.store, "Kubernetes", 3)
    assert top and "Kubernetes" in top[0].text


def test_hybrid_filters_by_document(loaded):
    svc, ids = loaded
    hits = hybrid_search(svc.store, "What skills are required?", 5, [ids["sample_job_description.txt"]])
    assert hits and {h.filename for h in hits} == {"sample_job_description.txt"}
    assert all(0 <= h.score <= 1 for h in hits)


def test_hybrid_answers_across_documents(loaded):
    svc, _ = loaded
    hits = hybrid_search(svc.store, "Which city is the candidate based in?", 3)
    assert any("Bhopal" in h.text for h in hits)


def test_delete_document_removes_chunks(loaded):
    svc, ids = loaded
    svc.store.delete_document(ids["sample_resume.txt"])
    assert {d["filename"] for d in svc.store.list_documents()} == {"sample_job_description.txt"}
    assert svc.store.get_document(ids["sample_resume.txt"]) is None
