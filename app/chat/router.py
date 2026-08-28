from fastapi import APIRouter

from app.chat.schemas import ChatRequest, ChatResponse, ToolChatResponse
from app.chat.service import generate_reply, generate_reply_with_tools

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Plain prompting: send a message, get a Groq-generated reply."""
    reply = generate_reply(request.message)
    return ChatResponse(reply=reply)


@router.post("/tools", response_model=ToolChatResponse)
def chat_with_tools(request: ChatRequest) -> ToolChatResponse:
    """Tool-calling: same idea, but the model may call get_collection_info
    on the real Qdrant store before answering.
    """
    reply, tool_used = generate_reply_with_tools(request.message)
    return ToolChatResponse(reply=reply, tool_used=tool_used)
