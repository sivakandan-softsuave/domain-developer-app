import uuid

from qdrant_client.models import PointStruct

from app.core.config import get_settings
from app.core.database import ensure_collection, get_qdrant_client
from app.core.embeddings import embed_texts, embedding_dimension
from app.modules.rag.chunker import chunk_text
from app.modules.rag.exceptions import RagError
from app.modules.rag.llm import generate_answer
from app.modules.rag.schemas import SourceChunk


def ingest_document(text: str, source: str | None) -> int:
    """Chunk, embed, and store a document in Qdrant. Returns the number of
    chunks created.
    """
    settings = get_settings()

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RagError("No content left to ingest after chunking.")

    try:
        vectors = embed_texts([chunk.text for chunk in chunks])
    except Exception as exc:
        raise RagError(f"Embedding failed: {exc}") from exc

    try:
        ensure_collection(embedding_dimension())
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk.text, "heading": chunk.heading, "source": source},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        get_qdrant_client().upsert(collection_name=settings.qdrant_collection_name, points=points)
    except Exception as exc:
        raise RagError(f"Storing chunks in Qdrant failed: {exc}") from exc

    return len(chunks)


def answer_question(question: str, top_k: int) -> tuple[str, list[SourceChunk]]:
    """Embed the question, retrieve the closest chunks from Qdrant, and
    generate an answer grounded in them. Returns (answer, sources).
    """
    settings = get_settings()

    try:
        [query_vector] = embed_texts([question])
    except Exception as exc:
        raise RagError(f"Embedding failed: {exc}") from exc

    try:
        results = get_qdrant_client().search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
    except Exception as exc:
        raise RagError(f"Qdrant search failed: {exc}") from exc

    sources = [
        SourceChunk(
            text=point.payload.get("text", ""),
            heading=point.payload.get("heading"),
            source=point.payload.get("source"),
            score=point.score,
        )
        for point in results
    ]

    if not sources:
        return "I don't have any ingested documents to answer from yet.", sources

    context = "\n\n---\n\n".join(source.text for source in sources)
    answer = generate_answer(question, context)
    return answer, sources
