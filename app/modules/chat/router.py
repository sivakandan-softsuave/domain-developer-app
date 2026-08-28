from fastapi import APIRouter

from app.modules.chat.llm import generate_reply
from app.modules.chat.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Plain prompting: send a message, get a Groq-generated reply."""
    reply = generate_reply(request.message)
    return ChatResponse(reply=reply)
