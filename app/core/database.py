from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return the cached, process-wide Qdrant client instance."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def check_connection() -> bool:
    """Ping Qdrant to confirm the client can reach the server.

    Returns True if Qdrant responded, False otherwise. Never raises -
    callers (e.g. the health module) decide how to report a failure.
    """
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# WEEK 2 ADDITION: a real piece of RAG infrastructure to hand to a tool
#
# `get_collection_info` is used by Week 2's tool-calling demo (see
# `chat/tools.py`) - it lets the Groq model ask a real question about the
# actual vector store instead of a made-up example. Week 3 will build the
# real `search_documents` tool once there's something to search.
# ---------------------------------------------------------------------------


def get_collection_info(collection_name: str | None = None) -> dict:
    """Report whether a Qdrant collection exists and how many vectors it holds.

    Falls back to the configured default collection when no name is given.
    Never raises - a missing collection is a normal, expected state before
    Week 3's ingestion pipeline has run, not an error.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name

    try:
        info = get_qdrant_client().get_collection(name)
        return {
            "collection_name": name,
            "exists": True,
            "points_count": info.points_count,
        }
    except Exception:
        return {
            "collection_name": name,
            "exists": False,
            "points_count": 0,
        }
