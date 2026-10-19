"""Central configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = ("openai", "anthropic")


class ConfigError(Exception):
    """Missing or invalid configuration (shown to the user)."""


@dataclass(frozen=True)
class Settings:
    provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-5-5"
    # auto = OpenAI embeddings when an OpenAI key exists, else Chroma's local MiniLM model.
    # local = offline keyword-hash embedding, used by tests and the evaluation script.
    embedding_backend: str = "auto"
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "chroma_db"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        g = lambda k, d="": os.getenv(k, d).strip()  # noqa: E731
        return cls(
            provider=g("LLM_PROVIDER", "openai").lower(),
            openai_api_key=g("OPENAI_API_KEY"),
            anthropic_api_key=g("ANTHROPIC_API_KEY"),
            openai_model=g("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_model=g("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            embedding_backend=g("EMBEDDING_BACKEND", "auto").lower(),
            embedding_model=g("EMBEDDING_MODEL", "text-embedding-3-small"),
            chroma_dir=g("CHROMA_DIR", "chroma_db"),
        )

    @property
    def resolved_embedding(self) -> str:
        if self.embedding_backend == "auto":
            return "openai" if self.openai_api_key else "chroma"
        return self.embedding_backend

    def validate_llm(self) -> None:
        if self.provider not in PROVIDERS:
            raise ConfigError(f"Unknown LLM_PROVIDER '{self.provider}'. Use one of: {', '.join(PROVIDERS)}.")
        if self.provider == "openai" and not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is missing. Add it to your .env file or the sidebar.")
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is missing. Add it to your .env file or the sidebar.")
        if self.resolved_embedding == "openai" and not self.openai_api_key:
            raise ConfigError("EMBEDDING_BACKEND=openai needs OPENAI_API_KEY.")
