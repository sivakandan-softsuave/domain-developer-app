from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest.")
    source: str | None = Field(
        default=None,
        description="Optional label for where this text came from (a filename, URL, or title).",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text cannot be blank")
        return stripped


class IngestResponse(BaseModel):
    chunks_created: int
    collection_name: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10, description="How many chunks to retrieve.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question cannot be blank")
        return stripped


class SourceChunk(BaseModel):
    text: str
    heading: str | None
    source: str | None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
