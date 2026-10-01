"""Execute the frozen three-member A/B/E graph with bounded retries and resume."""

import argparse
from collections import Counter
from pathlib import Path

from .models import ROSTERS
from .stateful_run import cache_usage
from .strong_run import run as run_journaled
from .triadic_study import TriadicGraph, TriadicMockProvider, prepare


def run(manifest, contexts, output, **kwargs):
    def enrich(report, directory):
        import json

        records = [json.loads(p.read_text()) for p in (directory / "questions").glob("*.json")]
        return {
            **report,
            "cache_usage": cache_usage(directory),
            "quality_comparison_statuses": dict(Counter(r["status"] for q in records for r in q["E2"])),
        }

    return run_journaled(
        manifest,
        contexts,
        output,
        graph_type=TriadicGraph,
        mock_provider_type=TriadicMockProvider,
        event_namespace="triadic_ABE",
        token_count=lambda model, text: len(text.split()),
        report_enricher=enrich,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roster", choices=tuple(ROSTERS), default="mixed_family")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest, contexts = prepare(args.plan, roster=args.roster, mock=args.mock)
    report = run(
        manifest,
        contexts,
        args.output,
        question_limit=args.questions,
        request_limit=args.max_inflight,
        workers=args.workers,
    )
    if report["status"] not in ("completed", "completed_with_failures", "preflight_completed"):
        raise SystemExit(1)
