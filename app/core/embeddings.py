from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Return the cached, process-wide embedding model instance."""
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of embedding vectors, in order."""
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embedding_dimension() -> int:
    """The length of each vector this model produces - Qdrant's collection
    has to be created with this exact size (see core/database.py).
    """
    return get_embedding_model().get_sentence_embedding_dimension()
