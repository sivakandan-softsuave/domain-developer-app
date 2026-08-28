from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration the RAG app needs, in one place."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Embeddings (local, via sentence-transformers - free, no API key) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Generation (Groq) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Vector store (Qdrant) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "domain_app_docs"

    # --- Chunking defaults (used from Week 3 onward) ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Trace logging (used from Week 5 onward) ---
    traces_file: str = "data/traces.jsonl"


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide Settings instance."""
    return Settings()
