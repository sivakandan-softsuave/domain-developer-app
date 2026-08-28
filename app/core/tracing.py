import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings


def log_trace(question: str, retrieved: list[dict], answer: str) -> str:
    """Append one complete trace to the traces file: the question, exactly
    what was fetched, and what was answered - enough to replay this
    request later. Returns the trace's id.

    A JSON-lines file (one JSON object per line) rather than a database:
    simple to append to, simple to read back with `json.loads` per line,
    no extra infrastructure for what's meant to be read by a person, not
    queried at scale.
    """
    settings = get_settings()
    trace = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved": retrieved,
        "answer": answer,
    }

    path = Path(settings.traces_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")

    return trace["id"]
