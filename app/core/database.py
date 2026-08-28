from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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


def ensure_collection(vector_size: int, collection_name: str | None = None) -> None:
    """Create the configured Qdrant collection if it doesn't exist yet.

    Safe to call before every ingest - does nothing if the collection is
    already there.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name
    client = get_qdrant_client()

    try:
        client.get_collection(name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
