# ---------------------------------------------------------------------------
# WEEK 2 CONCEPT: A second client factory
#
# Same idea as `core/database.py`, applied to Groq instead of Qdrant: one
# cached factory, built from `Settings`, so every module that needs to talk
# to Groq imports a ready-to-use client instead of constructing one (and
# re-reading the API key) inline. Two providers, one consistent pattern.
# ---------------------------------------------------------------------------

from functools import lru_cache

from groq import Groq

from app.core.config import get_settings


@lru_cache
def get_groq_client() -> Groq:
    """Return the cached, process-wide Groq client instance."""
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)
