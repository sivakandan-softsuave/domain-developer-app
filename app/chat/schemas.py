from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message to send to the model.",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be blank")
        return stripped


class ChatResponse(BaseModel):
    reply: str


class ToolChatResponse(BaseModel):
    reply: str
    tool_used: str | None = None  # None means the model answered without needing a tool
