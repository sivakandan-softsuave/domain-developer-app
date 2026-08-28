"""
Standalone dev script - NOT part of the FastAPI app.

Draws a genuinely random sample of real traces (see app/core/tracing.py)
for the Week 5 error-analysis process, and writes them into a CSV you fill
in by hand.

Why random, not "the 20 most recent" or "20 I remember being weird": a
curated sample only shows you problems you already suspected. A random
sample is the only way to find the ones you didn't.

This script only SAMPLES - it deliberately leaves `note`, `category`, and
`severity` blank. Fill them in yourself, in that order:
  1. Read every row and write ONE HONEST SENTENCE in `note` - what (if
     anything) went wrong. Do this for all rows before you look at
     patterns across them. Deciding categories first biases what you
     notice while reading.
  2. Only after every `note` is written, group similar notes and give
     each group a short `category` name a stranger would understand.
  3. Rate `severity` 1-5 (how much this failure would actually hurt a
     real user) for each row.

Then run scripts/rank_error_taxonomy.py to get the frequency x severity
ranking and a suggested fix target.

Run from the project root:
    python scripts/sample_traces.py [sample_size]
(sample_size defaults to 20)
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

REVIEW_FILE = Path("data/error_analysis_sample.csv")
DEFAULT_SAMPLE_SIZE = 20


def load_traces(traces_file: Path) -> list[dict]:
    if not traces_file.exists():
        return []

    traces = []
    with traces_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def summarize_retrieved(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(nothing retrieved)"
    parts = [f"{r.get('source') or '?'} (score={r.get('score', 0):.3f})" for r in retrieved]
    return "; ".join(parts)


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE_SIZE

    settings = get_settings()
    traces = load_traces(Path(settings.traces_file))

    if not traces:
        print(
            f"No traces found in '{settings.traces_file}'. Run the app and make some "
            "real /query calls first - this script has nothing to sample yet."
        )
        return

    sample = random.sample(traces, min(sample_size, len(traces)))
    print(f"Sampled {len(sample)} of {len(traces)} total traces (randomly, not curated).")

    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trace_id", "question", "retrieved_summary", "answer", "note", "category", "severity"])
        for trace in sample:
            writer.writerow(
                [
                    trace["id"],
                    trace["question"],
                    summarize_retrieved(trace["retrieved"]),
                    trace["answer"],
                    "",
                    "",
                    "",
                ]
            )

    print(f"Wrote {REVIEW_FILE} - open it and fill in 'note' for every row first.")


if __name__ == "__main__":
    main()
