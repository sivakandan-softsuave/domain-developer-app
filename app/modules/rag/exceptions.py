from fastapi import Request
from fastapi.responses import JSONResponse


class RagError(Exception):
    """Raised whenever ingesting a document or answering a query fails,
    for any reason."""

    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


async def rag_error_handler(request: Request, exc: RagError) -> JSONResponse:
    """Registered on the app in main.py."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
