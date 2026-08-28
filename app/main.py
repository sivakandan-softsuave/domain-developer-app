from fastapi import FastAPI

from app.core.config import get_settings
from app.core.database import check_connection
from app.modules.chat.exceptions import ChatProviderError, chat_provider_error_handler
from app.modules.chat.router import router as chat_router
from app.modules.rag.exceptions import RagError, rag_error_handler
from app.modules.rag.router import router as rag_router

app = FastAPI(
    title="Domain RAG App",
    description="A RAG application built with Qdrant, Groq and local sentence-transformers embeddings.",
    version="0.1.0",
)

app.include_router(chat_router)
app.include_router(rag_router)
app.add_exception_handler(ChatProviderError, chat_provider_error_handler)
app.add_exception_handler(RagError, rag_error_handler)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "qdrant_connected": check_connection(),
        "embedding_model": settings.embedding_model,
        "groq_model": settings.groq_model,
    }
