from fastapi import Request
from fastapi.responses import JSONResponse


class ChatProviderError(Exception):
    """Raised whenever talking to Groq fails, for any reason - rate limit,
    network failure, invalid key, unexpected response, an unknown tool the
    model asked for, etc. `service.py` doesn't need to know anything about
    HTTP; it just raises this with a human-readable message, and the
    handler below decides the status code.
    """

    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


async def chat_provider_error_handler(request: Request, exc: ChatProviderError) -> JSONResponse:
    """Registered on the app in main.py. Converts ChatProviderError into a
    clean JSON body instead of an unhandled-exception 500.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
