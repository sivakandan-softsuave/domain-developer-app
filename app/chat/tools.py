from app.core.database import get_collection_info


def get_collection_info_tool(collection_name: str | None = None) -> dict:
    """Thin wrapper so the tool's Python signature matches its JSON schema
    exactly - the model calls this by name, with whatever arguments (if
    any) it decided to pass.
    """
    return get_collection_info(collection_name)


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_collection_info",
            "description": (
                "Check the app's Qdrant vector store: whether the document "
                "collection exists yet and how many vectors it currently holds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": (
                            "Name of the collection to check. Optional - if "
                            "omitted, the app's default collection is used."
                        ),
                    }
                },
                "required": [],
            },
        },
    }
]

AVAILABLE_TOOLS = {"get_collection_info": get_collection_info_tool}
