"""Tokenizer and embedding functions for ChromaDB."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import chromadb

from talkai.config import Settings

_TOKEN = re.compile(r"[a-z0-9+#.]{2,}")
_STOP = set(
    "the a an and or of to in on for with is are was were be by at as it this that from what which who whom "
    "did does do how where when why has have he she his her they their i you me my about tell give list".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase keyword tokens without stop words (used by BM25 and local embeddings)."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".")
        if tok and tok not in _STOP:
            out.append(tok)
    return out


class HashEmbedding(chromadb.EmbeddingFunction):
    """Deterministic offline embedding (hashed unigrams + bigrams). Keyword-aware, not semantic.

    Used for unit tests and the offline evaluation run; production uses a neural model.
    """

    DIM = 512

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "talkai_hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()

    def __call__(self, input):  # noqa: A002 - name required by ChromaDB
        return [self._embed(t) for t in input]

    def _embed(self, text: str) -> list[float]:
        words = tokenize(text)
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vec = [0.0] * self.DIM
        for feat, n in Counter(feats).items():
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            vec[h % self.DIM] += (1 + math.log(n)) * (1 if (h >> 64) % 2 == 0 else -1)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def build_embedding_function(settings: Settings):
    backend = settings.resolved_embedding
    if backend == "openai":
        from chromadb.utils import embedding_functions

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key, model_name=settings.embedding_model
        )
    if backend == "chroma":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()  # local ONNX MiniLM, downloaded on first use
    if backend == "local":
        return HashEmbedding()
    raise ValueError(f"Unknown embedding backend '{backend}'.")
