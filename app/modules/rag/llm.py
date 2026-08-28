from groq import APIError

from app.core.config import get_settings
from app.core.groq_client import get_groq_client
from app.modules.rag.exceptions import RagError
from app.modules.rag.prompts import RAG_SYSTEM_PROMPT, build_user_prompt


def generate_answer(question: str, context: str) -> str:
    """Ask Groq to answer the question using only the given context."""
    settings = get_settings()

    try:
        response = get_groq_client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
        )
    except APIError as exc:
        raise RagError(f"Groq API call failed: {exc}") from exc

    return response.choices[0].message.content
