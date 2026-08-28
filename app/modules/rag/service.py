import uuid

from qdrant_client.models import PointStruct

from app.core.config import get_settings
from app.core.database import ensure_collection, get_qdrant_client
from app.core.embeddings import embed_texts, embedding_dimension
from app.modules.rag.chunker import chunk_text
from app.modules.rag.exceptions import RagError
from app.modules.rag.llm import generate_answer
from app.modules.rag.retrieval import hybrid_search
from app.modules.rag.schemas import DebugQueryResponse, SourceChunk


def ingest_document(text: str, source: str | None, collection_name: str | None = None) -> int:
    """Chunk, embed, and store a document in Qdrant. Returns the number of
    chunks created. `collection_name` defaults to the configured
    collection - overridable so the eval script can ingest into an
    isolated collection instead of the real app's.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection_name

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RagError("No content left to ingest after chunking.")

    try:
        vectors = embed_texts([chunk.text for chunk in chunks])
    except Exception as exc:
        raise RagError(f"Embedding failed: {exc}") from exc

    try:
        ensure_collection(embedding_dimension(), name)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk.text, "heading": chunk.heading, "source": source},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        get_qdrant_client().upsert(collection_name=name, points=points)
    except Exception as exc:
        raise RagError(f"Storing chunks in Qdrant failed: {exc}") from exc

    return len(chunks)


def _retrieve_and_answer(question: str, top_k: int) -> tuple[str, list[SourceChunk]]:
    """Shared by answer_question() and inspect_query(): retrieve via
    hybrid search, then generate an answer grounded in what was found.
    """
    sources = hybrid_search(question, top_k)

    if not sources:
        return "I don't have any ingested documents to answer from yet.", sources

    context = "\n\n---\n\n".join(source.text for source in sources)
    answer = generate_answer(question, context)
    return answer, sources


def answer_question(question: str, top_k: int) -> tuple[str, list[SourceChunk]]:
    """Answer a question using retrieval-augmented generation. Returns
    (answer, sources)."""
    return _retrieve_and_answer(question, top_k)


def inspect_query(question: str, top_k: int) -> DebugQueryResponse:
    """The inspection view: question, retrieved chunks, and answer
    together - so you can tell whether a wrong answer came from a
    retrieval failure (wrong chunks) or a generation failure (right
    chunks, bad answer).
    """
    answer, sources = _retrieve_and_answer(question, top_k)
    return DebugQueryResponse(question=question, retrieved=sources, answer=answer)
