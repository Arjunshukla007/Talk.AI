"""Evaluate Talk.AI retrieval (and optionally answers) on a small labelled dataset.

Retrieval metrics need no API key:   python -m evaluation.run_eval --embedding local
Answer metrics (uses your LLM key):  python -m evaluation.run_eval --embedding auto --llm

hit@k = share of questions where a retrieved chunk contains the expected text.
MRR   = mean of 1/rank of the first correct chunk (higher is better).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from talkai.config import Settings
from talkai.rag import NOT_FOUND, RAGService
from talkai.retrieval import bm25_search, hybrid_search
from talkai.store import VectorStore

ROOT = Path(__file__).parent
DOCS = [ROOT.parent / "sample_data" / "sample_resume.txt", ROOT.parent / "sample_data" / "sample_job_description.txt"]


def load_dataset(path: Path = ROOT / "dataset.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_metrics(store: VectorStore, dataset: list[dict], k: int = 3) -> dict[str, dict[str, float]]:
    methods = {
        "vector": lambda q: store.vector_search(q, k),
        "bm25": lambda q: bm25_search(store, q, k),
        "hybrid": lambda q: hybrid_search(store, q, k),
    }
    rows = [d for d in dataset if d["answerable"]]
    results = {}
    for name, search in methods.items():
        hits_at_k, rr = 0, 0.0
        for item in rows:
            texts = [h.text.lower() for h in search(item["question"])]
            rank = next((i for i, t in enumerate(texts, 1) if item["expected"].lower() in t), None)
            hits_at_k += rank is not None
            rr += 1 / rank if rank else 0
        results[name] = {f"hit@{k}": round(hits_at_k / len(rows), 3), "mrr": round(rr / len(rows), 3)}
    return results


def answer_metrics(service: RAGService, dataset: list[dict]) -> dict[str, float]:
    correct = refused_ok = n_ans = n_unans = 0
    for item in dataset:
        ans = service.ask(item["question"])
        if item["answerable"]:
            n_ans += 1
            correct += item["expected"].lower() in ans.text.lower() and ans.found
        else:
            n_unans += 1
            refused_ok += not ans.found
    return {"answer_contains_expected": round(correct / n_ans, 3), "correct_refusals": round(refused_ok / n_unans, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="local", help="local (offline), chroma or openai")
    parser.add_argument("--llm", action="store_true", help="also grade end-to-end answers (needs an API key)")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    settings = replace(Settings.from_env(), embedding_backend=args.embedding, chroma_dir=tempfile.mkdtemp(), chunk_size=200,
                       chunk_overlap=30)
    store = VectorStore(settings)
    for path in DOCS:
        from talkai.ingestion import load_document
        store.add_document(load_document(path.read_bytes(), path.name))
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} questions, {store.collection.count()} chunks, embedding={args.embedding}\n")
    print(f"{'method':8} {'hit@' + str(args.k):>7} {'mrr':>6}")
    for name, m in retrieval_metrics(store, dataset, args.k).items():
        print(f"{name:8} {m[f'hit@{args.k}']:>7} {m['mrr']:>6}")
    if args.llm:
        from talkai.llm import build_llm
        service = RAGService(store, build_llm(settings), settings)
        print("\nEnd-to-end:", answer_metrics(service, dataset))
        print(f"(a correct refusal means the reply was: {NOT_FOUND!r})")


if __name__ == "__main__":
    main()
