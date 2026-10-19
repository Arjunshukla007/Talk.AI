from dataclasses import replace

from evaluation.run_eval import DOCS, load_dataset, retrieval_metrics
from talkai.ingestion import load_document


def test_hybrid_retrieval_is_competitive(store):
    for p in DOCS:
        store.add_document(load_document(p.read_bytes(), p.name))
    m = retrieval_metrics(store, load_dataset(), k=3)
    assert m["hybrid"]["hit@3"] >= max(m["vector"]["hit@3"], m["bm25"]["hit@3"]) - 0.01
    assert m["hybrid"]["hit@3"] >= 0.7
