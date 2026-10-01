"""Explicit gate/full execution; old experiments remain frozen and read-only."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot.stateful_run import cache_usage
from llm_committee.pivot.strong_run import run as run_journaled
from scripts.forced_feedback_public_history import (
    OUTPUT,
    FeedbackGraph,
    FeedbackMockProvider,
    prepare,
    verify_preservation,
)
from scripts.scale_public_history import freeze, read


def enrich(report, output):
    groups = Counter()
    with sqlite3.connect(f"file:{output}/requests.sqlite3?mode=ro", uri=True) as db:
        ledger = list(db.execute("SELECT status,COUNT(*),SUM(charge) FROM calls GROUP BY status"))
        for (raw,) in db.execute("SELECT request FROM calls WHERE status='completed'"):
            groups[json.loads(raw)["purpose"]] += 1
    records = [read(p) for p in (output / "questions").glob("*.json")]
    cases = [c for r in records for c in r["cases"]]
    received = {"completed", "received", "invalid"}
    return {
        **report,
        "completed_cases": len(cases),
        "successful_cases": sum(c["status"] == "completed" for c in cases),
        "completed_new_calls_by_purpose": dict(groups),
        "reused_measurements": sum(len(r["reused_measurements"]) for r in records),
        "cost_accounting": {
            "received_response_estimate_usd": sum(cost for s, _, cost in ledger if s in received),
            "unresolved_reservations_usd": sum(cost for s, _, cost in ledger if s not in received),
            "unresolved_attempts": sum(n for s, n, _ in ledger if s not in received),
        },
        "cache_usage": cache_usage(output),
    }


def run(
    output=OUTPUT, *, mock=False, questions=50, max_inflight=64, workers=8, provider_factory=None, token_count=None
):
    output = Path(output).resolve()
    manifest, contexts = prepare(mock=mock)
    verify_preservation(manifest)
    freeze(output / "implementation-hashes.json", manifest["implementation_files"])
    result = run_journaled(
        manifest,
        contexts,
        output,
        question_limit=questions,
        request_limit=max_inflight,
        workers=workers,
        graph_type=FeedbackGraph,
        mock_provider_type=FeedbackMockProvider,
        provider_factory=provider_factory,
        token_count=token_count,
        event_namespace="public_history_forced_feedback",
        report_enricher=enrich,
    )
    verify_preservation(manifest)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run(
        args.output, mock=args.mock, questions=args.questions, max_inflight=args.max_inflight, workers=args.workers
    )
    print(json.dumps(result), flush=True)
    if result["status"] not in ("completed", "preflight_completed", "completed_with_failures"):
        raise SystemExit(1)
