from groq import APIError

from app.core.config import get_settings
from app.core.groq_client import get_groq_client
from app.modules.chat.exceptions import ChatProviderError
from app.modules.chat.prompts import SYSTEM_PROMPT


def generate_reply(message: str) -> str:
    """Ask Groq to reply to the user's message."""
    settings = get_settings()

    try:
        response = get_groq_client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )
    except APIError as exc:
        raise ChatProviderError(f"Groq API call failed: {exc}") from exc

    return response.choices[0].message.content
