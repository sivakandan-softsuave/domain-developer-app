from fastapi import APIRouter

from app.core.config import get_settings
from app.modules.rag.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.modules.rag.service import answer_question, ingest_document

router = APIRouter(tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store a document so /query can retrieve it later."""
    chunks_created = ingest_document(request.text, request.source)
    settings = get_settings()
    return IngestResponse(chunks_created=chunks_created, collection_name=settings.qdrant_collection_name)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using retrieval-augmented generation over
    whatever has been ingested so far."""
    answer, sources = answer_question(request.question, request.top_k)
    return QueryResponse(answer=answer, sources=sources)
