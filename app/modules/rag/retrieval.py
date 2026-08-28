from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.database import get_qdrant_client
from app.core.embeddings import embed_texts
from app.modules.rag.exceptions import RagError
from app.modules.rag.schemas import SourceChunk

RRF_K = 60  # standard Reciprocal Rank Fusion constant


def _point_to_source_chunk(point, score: float) -> SourceChunk:
    # `score` is passed explicitly rather than read off `point` because
    # the two callers get it from different places: vector_search's
    # points are Qdrant ScoredPoints (have .score); bm25_search's points
    # come from client.scroll(), which returns plain Records with NO
    # score attribute at all - the BM25 score has to come from bm25's own
    # output instead.
    return SourceChunk(
        text=point.payload.get("text", ""),
        heading=point.payload.get("heading"),
        source=point.payload.get("source"),
        score=float(score),  # bm25's scores are numpy.float64, not plain float
    )


def vector_search(question: str, top_k: int, collection_name: str | None = None) -> list[SourceChunk]:
    """Semantic search: embed the question, ask Qdrant for the closest
    stored vectors. This is the plain, Week 3 retrieval method - good at
    matching meaning, but can miss exact terms (codes, IDs, names) that
    don't carry much semantic weight of their own.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name

    try:
        [query_vector] = embed_texts([question])
        results = get_qdrant_client().search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
        )
    except Exception as exc:
        raise RagError(f"Vector search failed: {exc}") from exc

    return [_point_to_source_chunk(point, point.score) for point in results]


def _load_corpus(collection_name: str) -> list:
    """Fetch every point in the collection for BM25 to index.

    Scrolling the whole collection at query time isn't how you'd do this
    at scale, but it's simple and transparent for a teaching app of this
    size - the point is understanding BM25 + RRF, not building a
    production search engine.
    """
    client = get_qdrant_client()
    points = []
    next_offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if next_offset is None:
            break

    return points


def bm25_search(question: str, top_k: int, collection_name: str | None = None) -> list[SourceChunk]:
    """Keyword search: BM25 ranks chunks by exact term overlap with the
    question. Catches things vector search often misses - exact codes,
    IDs, names - at the cost of missing paraphrases and synonyms.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name

    try:
        points = _load_corpus(name)
        if not points:
            return []

        tokenized_corpus = [point.payload.get("text", "").lower().split() for point in points]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(question.lower().split())

        ranked = sorted(zip(points, scores), key=lambda pair: pair[1], reverse=True)
    except Exception as exc:
        raise RagError(f"Keyword search failed: {exc}") from exc

    return [_point_to_source_chunk(point, score) for point, score in ranked[:top_k]]


def hybrid_search(question: str, top_k: int, collection_name: str | None = None) -> list[SourceChunk]:
    """Hybrid search: combine vector search and BM25 via Reciprocal Rank
    Fusion (RRF), so the final ranking benefits from both - meaning-based
    matches AND exact-term matches - without needing their raw scores to
    be on comparable scales (RRF only uses each result's RANK in its own
    list, not its score).

    RRF score for a chunk = sum, over every ranked list it appears in, of
    1 / (RRF_K + rank). A chunk ranked highly in either list scores well;
    one ranked highly in BOTH scores best.
    """
    # Pull a slightly larger candidate pool from each method than top_k,
    # so fusion has more to work with than just each method's top pick.
    candidate_pool = max(top_k * 3, 10)

    vector_results = vector_search(question, candidate_pool, collection_name)
    bm25_results = bm25_search(question, candidate_pool, collection_name)

    scores: dict[str, float] = {}
    chunks_by_key: dict[str, SourceChunk] = {}

    for ranked_list in (vector_results, bm25_results):
        for rank, chunk in enumerate(ranked_list):
            key = f"{chunk.source}:{chunk.text}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunks_by_key.setdefault(key, chunk)

    # .score on the returned chunks is now the RRF score, not a raw cosine
    # similarity or BM25 score - the two aren't on the same scale, so this
    # is the only score that means anything post-fusion.
    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        chunks_by_key[key].model_copy(update={"score": scores[key]})
        for key in fused_keys[:top_k]
    ]
