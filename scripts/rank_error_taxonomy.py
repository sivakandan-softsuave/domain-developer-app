"""
Standalone dev script - NOT part of the FastAPI app.

Reads the CSV that scripts/sample_traces.py wrote (and that you've since
filled in `note`/`category`/`severity` for by hand), groups the read
traces into their named categories, ranks each category by
frequency x average severity, and names the top-ranked one as the
suggested fix target.

This script cannot do the actual error analysis for you - it only tallies
categories and severities YOU decided on after reading every trace
yourself. If you haven't filled in the CSV yet, there's nothing real for
this to compute.

Run from the project root, after filling in data/error_analysis_sample.csv:
    python scripts/rank_error_taxonomy.py
"""

import csv
from pathlib import Path

REVIEW_FILE = Path("data/error_analysis_sample.csv")


def main() -> None:
    if not REVIEW_FILE.exists():
        print(f"'{REVIEW_FILE}' doesn't exist yet - run scripts/sample_traces.py first.")
        return

    with REVIEW_FILE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def is_labeled(row: dict) -> bool:
        return bool(row["category"].strip()) and bool(row["severity"].strip())

    complete = [r for r in rows if is_labeled(r)]
    incomplete = [r for r in rows if not is_labeled(r)]

    if incomplete:
        print(
            f"{len(incomplete)} of {len(rows)} rows are missing 'category' or "
            "'severity' - skipping them. Fill in every row for an honest ranking."
        )

    if not complete:
        print("No fully-labeled rows yet - nothing to rank.")
        return

    by_category: dict[str, list[int]] = {}
    for row in complete:
        category = row["category"].strip()
        try:
            severity = int(row["severity"])
        except ValueError:
            print(f"Skipping row with non-numeric severity: {row['severity']!r}")
            continue
        by_category.setdefault(category, []).append(severity)

    ranked = sorted(
        by_category.items(),
        key=lambda item: len(item[1]) * (sum(item[1]) / len(item[1])),
        reverse=True,
    )

    print(f"\nRanked error taxonomy ({len(complete)} labeled traces):\n")
    print(f"{'category':<30} {'count':>6} {'avg severity':>13} {'count x severity':>18}")
    for category, severities in ranked:
        count = len(severities)
        avg_severity = sum(severities) / count
        score = count * avg_severity
        print(f"{category:<30} {count:>6} {avg_severity:>13.1f} {score:>18.1f}")

    top_category = ranked[0][0]
    print(f"\nSuggested fix target: '{top_category}' - highest frequency x severity.")
    print("Before fixing it, write down a specific prediction of what should improve.")


if __name__ == "__main__":
    main()
