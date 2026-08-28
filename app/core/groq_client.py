from functools import lru_cache

from groq import Groq

from app.core.config import get_settings


@lru_cache
def get_groq_client() -> Groq:
    """Return the cached, process-wide Groq client instance."""
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)
