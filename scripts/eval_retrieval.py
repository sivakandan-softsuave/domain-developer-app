"""
Standalone dev script - NOT part of the FastAPI app, not imported by it.

Proves Week 4's "one improvement" (hybrid search) with a number: ingests a
tiny fixture corpus into an isolated Qdrant collection (separate from
whatever QDRANT_COLLECTION_NAME points at, so this never touches real
data), runs a small labeled eval set through both vector-only search
("before") and hybrid search ("after") at top_k=3, and prints hit-rate@3
for each.

One of the eval questions is just the bare code "ERR-4032" - a case
vector search tends to struggle with (a short code carries little
semantic meaning) and BM25 nails trivially (exact term match). That's the
concrete "before vs after" this script is meant to demonstrate.

Needs a real Qdrant instance reachable via the app's .env. Does NOT need
GROQ_API_KEY - this only exercises retrieval, never generation.

Run from the project root:
    python scripts/eval_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import get_qdrant_client  # noqa: E402
from app.modules.rag.retrieval import hybrid_search, vector_search  # noqa: E402
from app.modules.rag.service import ingest_document  # noqa: E402

EVAL_COLLECTION = "domain_app_docs_eval"
TOP_K = 3

FIXTURE_DOCS = [
    (
        "refund-policy.md",
        "# Refund Policy\n\n"
        "Refunds are issued within 14 days of purchase if the item is "
        "unused and in its original packaging.",
    ),
    (
        "error-codes.md",
        "# Error Codes\n\n"
        "## ERR-4032\n\n"
        "ERR-4032 means the payment gateway rejected the transaction due "
        "to an expired card. Ask the customer to update their card "
        "details and retry.\n\n"
        "## ERR-5001\n\n"
        "ERR-5001 indicates a temporary server outage. Retry the request "
        "after a few minutes.",
    ),
    (
        "shipping-policy.md",
        "# Shipping Policy\n\n"
        "Standard shipping takes 3-5 business days. Express shipping "
        "takes 1-2 business days and costs an extra $10.",
    ),
    (
        "account-deletion.md",
        "# Account Deletion\n\n"
        "Users can delete their account permanently from Settings > "
        "Privacy > Delete Account. This action cannot be undone.",
    ),
]

EVAL_SET = [
    ("How long do refunds take?", "refund-policy.md"),
    ("How fast is express shipping?", "shipping-policy.md"),
    ("How do I permanently delete my account?", "account-deletion.md"),
    ("ERR-4032", "error-codes.md"),
    ("What should I do about ERR-5001?", "error-codes.md"),
    ("What happens if I return a used item?", "refund-policy.md"),
]


def reset_eval_collection() -> None:
    """Start from a clean slate each run, so re-running this script
    doesn't accumulate duplicate points."""
    try:
        get_qdrant_client().delete_collection(EVAL_COLLECTION)
    except Exception:
        pass  # didn't exist yet - fine


def hit_rate_at_k(search_fn) -> float:
    hits = 0
    for question, expected_source in EVAL_SET:
        results = search_fn(question, TOP_K, EVAL_COLLECTION)
        retrieved_sources = {result.source for result in results}
        if expected_source in retrieved_sources:
            hits += 1
    return hits / len(EVAL_SET)


def main() -> None:
    reset_eval_collection()

    print(f"Ingesting {len(FIXTURE_DOCS)} fixture documents into '{EVAL_COLLECTION}'...")
    for source, text in FIXTURE_DOCS:
        chunks_created = ingest_document(text, source, collection_name=EVAL_COLLECTION)
        print(f"  {source}: {chunks_created} chunk(s)")

    print(f"\nRunning {len(EVAL_SET)} eval questions at top_k={TOP_K}...\n")

    vector_hit_rate = hit_rate_at_k(vector_search)
    hybrid_hit_rate = hit_rate_at_k(hybrid_search)

    print(f"Vector-only hit-rate@{TOP_K}: {vector_hit_rate:.0%}")
    print(f"Hybrid       hit-rate@{TOP_K}: {hybrid_hit_rate:.0%}")

    delta = hybrid_hit_rate - vector_hit_rate
    if delta > 0:
        print(f"\nHybrid search improved hit-rate@{TOP_K} by {delta:.0%} on this eval set.")
    elif delta < 0:
        print(f"\nHybrid search made hit-rate@{TOP_K} worse by {abs(delta):.0%} on this eval set.")
    else:
        print(f"\nNo change in hit-rate@{TOP_K} on this eval set.")


if __name__ == "__main__":
    main()
