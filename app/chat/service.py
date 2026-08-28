import json

from groq import APIError

from app.chat.exceptions import ChatProviderError
from app.chat.tools import AVAILABLE_TOOLS, TOOL_SCHEMA
from app.core.config import get_settings
from app.core.groq_client import get_groq_client

SYSTEM_PROMPT = "You are a helpful assistant for a demo RAG teaching app."


def generate_reply(message: str) -> str:
    """Plain prompting: one request, one response, no tools involved."""
    client = get_groq_client()
    settings = get_settings()

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )
    except APIError as exc:
        raise ChatProviderError(f"Groq API call failed: {exc}") from exc

    return response.choices[0].message.content


def generate_reply_with_tools(message: str) -> tuple[str, str | None]:
    """Tool-calling: the model may ask for `get_collection_info` before
    answering. This is a fixed two-step exchange (not a general loop),
    since there's only one tool and it never needs to chain into another.

    Returns (reply_text, name_of_tool_used_or_None).
    """
    client = get_groq_client()
    settings = get_settings()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    try:
        first_response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            tools=TOOL_SCHEMA,
            tool_choice="auto",
        )
    except APIError as exc:
        raise ChatProviderError(f"Groq API call failed: {exc}") from exc

    response_message = first_response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        # The model answered directly - it decided it didn't need the tool.
        return response_message.content, None

    # The model wants a tool run. Echo its request back into the
    # conversation (the API requires this), then append one "tool" message
    # per result before asking for the final answer.
    messages.append(response_message.model_dump(exclude_unset=True))

    tool_used = None
    for call in tool_calls:
        tool_name = call.function.name
        tool_args = json.loads(call.function.arguments or "{}")
        tool_function = AVAILABLE_TOOLS.get(tool_name)

        if tool_function is None:
            raise ChatProviderError(f"Model requested unknown tool: {tool_name}")

        result = tool_function(**tool_args)
        tool_used = tool_name

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(result),
            }
        )

    try:
        second_response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
        )
    except APIError as exc:
        raise ChatProviderError(f"Groq API call failed: {exc}") from exc

    return second_response.choices[0].message.content, tool_used
