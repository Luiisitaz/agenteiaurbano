from __future__ import annotations

from typing import Dict, List

from .database import SessionLocal
from .embeddings import embed_text
from .tools import search_similar_reports
from .config import settings


def rag_search(query: str, limit: int | None = None) -> List[Dict]:
    embedding = embed_text(query)
    max_results = limit or settings.max_results
    with SessionLocal() as db:
        return search_similar_reports(db, embedding, limit=max_results)
