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
