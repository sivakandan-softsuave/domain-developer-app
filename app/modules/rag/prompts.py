RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the provided context. If the context doesn't contain the answer, "
    "say you don't know - never make something up."
)


def build_user_prompt(question: str, context: str) -> str:
    """Combine the retrieved context and the question into the one message
    sent to the model."""
    return f"Context:\n{context}\n\nQuestion: {question}"
