import logging
from typing import List, Dict
#import numpy as np

from backend.ingestion.embeddings import EmbeddingService
from backend.config import settings

log = logging.getLogger(__name__)

_embed_svc = None


def _get_embed_svc():
    global _embed_svc
    if _embed_svc is None:
        _embed_svc = EmbeddingService()
    return _embed_svc


def search_chunks(query: str, top_k: int = None, user_id: str = None) -> List[Dict]:
    """
    Query ChromaDB, normalize scores, re-rank results.
    Returns list of chunk dicts with metadata and scores.
    """
    k = top_k or settings.top_k
    svc = _get_embed_svc()

    where_filter = {"user_id": user_id} if user_id else None
    results = svc.query(query, top_k=k * 2, where_filter=where_filter)

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # chroma returns cosine distance, convert to similarity
    scores = [1.0 - d for d in distances]

    # normalize to 0-1 range
    scores = _normalize_scores(scores)

    # build result list
    chunks = []
    for i in range(len(ids)):
        chunks.append({
            "embedding_id": ids[i],
            "content": docs[i],
            "metadata": metas[i],
            "score": scores[i],
        })

    # re-rank: boost tables and images slightly, they carry denser info
    chunks = _rerank(chunks)

    return chunks[:k]


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _rerank(chunks: List[Dict]) -> List[Dict]:
    """Simple re-ranking: slight boost for tables/images, then sort by adjusted score."""
    TYPE_BOOST = {"table": 0.05, "image_caption": 0.03, "text": 0.0}

    for chunk in chunks:
        content_type = chunk["metadata"].get("content_type", "text")
        boost = TYPE_BOOST.get(content_type, 0.0)
        chunk["score"] = min(chunk["score"] + boost, 1.0)

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks
