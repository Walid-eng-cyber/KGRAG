"""Local embeddings via Ollama (bge-m3). Shared by entity resolution and the
Phase 2 vector index so both embed text the same way."""
from __future__ import annotations

import ollama

import config


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector (list[float]) per input."""
    if not texts:
        return []
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    return client.embed(model=config.EMBED_MODEL, input=texts)["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
