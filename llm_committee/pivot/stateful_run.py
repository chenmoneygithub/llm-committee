"""Run the integrated-position protocol in a fresh, immutable output directory."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .models import ROSTERS
from .stateful_study import PROTOCOL_PROMPTS, QUESTION_COUNT, StatefulGraph, StatefulMockProvider, prepare
from .strong_run import run as run_journaled


def cache_usage(output):
    """Actual reported cache usage, including received failed attempts; no assumed hits."""
    groups = {}
    with sqlite3.connect(f"file:{Path(output).resolve() / 'requests.sqlite3'}?mode=ro", uri=True) as db:
        for request, response in db.execute("SELECT request,response FROM calls WHERE response IS NOT NULL"):
            req, value = json.loads(request), json.loads(response)
            key = (req["model"], req["purpose"])
            group = groups.setdefault(
                key,
                {
                    "model": key[0],
                    "purpose": key[1],
                    "responses": 0,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "responses_with_unknown_usage": 0,
                    "unknown_token_fields": dict.fromkeys(("input_tokens", "cached_tokens", "output_tokens"), 0),
                },
            )
            group["responses"] += 1
            unknown = False
            for name in ("input_tokens", "cached_tokens", "output_tokens"):
                if type(value.get(name)) is int and value[name] >= 0:
                    group[name] += value[name]
                else:
                    group["unknown_token_fields"][name] += 1
                    unknown = True
            group["responses_with_unknown_usage"] += unknown
    return [
        {
            **g,
            "cached_input_percent": (
                None
                if g["responses_with_unknown_usage"]
                else 100 * g["cached_tokens"] / g["input_tokens"] if g["input_tokens"] else 0
            ),
            "token_total_scope": "Sum of reported nonnegative integer fields only; missing fields remain unknown",
        }
        for _, g in sorted(groups.items())
    ]


def run(manifest, contexts, output, **kwargs):
    if manifest["protocol_version"] not in PROTOCOL_PROMPTS or manifest["kind"] not in (
        "offline_mock",
        "paid_stateful_dyadic_pilot",
    ):
        raise ValueError("Stateful runner cannot reuse an archived debate protocol")

    def summarize(report, directory):
        return {
            **report,
            "cache_usage": cache_usage(directory),
            "cache_usage_note": "Provider-reported usage, not a prediction; mock counts are synthetic. Includes received retry attempts.",
        }

    return run_journaled(
        manifest,
        contexts,
        output,
        graph_type=StatefulGraph,
        mock_provider_type=StatefulMockProvider,
        event_namespace="stateful_study",
        report_enricher=summarize,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Offline fixtures only; no endpoints or credentials")
    mode.add_argument("--live", action="store_true", help="Explicitly start paid model calls")
    parser.add_argument("--roster", choices=tuple(ROSTERS), default="mixed_family")
    parser.add_argument(
        "--public-history",
        action="store_true",
        help="Record positions but never reinject their private fields into debate/D1",
    )
    parser.add_argument("--questions", type=int, default=QUESTION_COUNT)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest, contexts = prepare(args.plan, mock=args.mock, roster=args.roster, public_history=args.public_history)
    report = run(
        manifest,
        contexts,
        args.output,
        question_limit=args.questions,
        request_limit=args.max_inflight,
        workers=args.workers,
    )
    if report["status"] not in ("completed", "preflight_completed", "completed_with_failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
