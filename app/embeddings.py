from __future__ import annotations

from typing import List
from openai import OpenAI

from .config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


def embed_text(text: str) -> List[float]:
    client = _get_client()
    kwargs = {}
    if settings.embedding_model.startswith("text-embedding-3"):
        kwargs["dimensions"] = settings.embedding_dimensions
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=[text],
        **kwargs,
    )
    return resp.data[0].embedding
