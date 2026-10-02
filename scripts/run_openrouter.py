"""Run the existing GPT debate protocol through OpenRouter.

Prepare and mocks are offline. --live is the sole path to billed requests.
Only OPENROUTER_API_KEY is used, including for the Gemini text judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from llm_committee.pivot.models import PilotConfig, canonical, digest
from llm_committee.pivot.openrouter_provider import BASE_URL, MODELS, OpenRouterProvider
from llm_committee.pivot.stateful_run import run as run_dyadic
from llm_committee.pivot.stateful_study import StatefulGraph, StatefulMockProvider, prepare as prepare_dyadic
from llm_committee.pivot.triadic_run import run as run_triadic
from llm_committee.pivot.triadic_study import TriadicGraph, TriadicMockProvider, prepare as prepare_triadic

ROOT = Path(__file__).resolve().parents[1]
VERSION = "openrouter-single-assignment-2026-10-02-v1"


def prepare(*, members=2, questions=10, roster="same_family", mock=False):
    if members not in (2, 3) or questions < 1 or roster not in ("same_family", "same_model"):
        raise ValueError("Select a positive question count, GPT roster, and 2- or 3-member routing")
    design_file = ROOT / "docs" / (
        "turn-tone-dyadic-shared-plan-2026-09-26.json" if members == 2
        else "turn-tone-triadic-shared-plan-2026-09-28.json"
    )
    if members == 2:
        manifest, contexts = prepare_dyadic(design_file, roster=roster, public_history=True, mock=mock)
    else:
        manifest, contexts = prepare_triadic(design_file, roster=roster, mock=mock)
    if questions > len(manifest["plans"]):
        raise ValueError("Question count exceeds the frozen plan; do not silently truncate")
    manifest["plans"] = manifest["plans"][:questions]
    contexts = {p["question_id"]: contexts[p["question_id"]] for p in manifest["plans"]}
    manifest["source"]["contexts_sha256"] = digest(contexts)
    manifest["config"] = asdict(replace(PilotConfig(**manifest["config"]), closed_provider="openrouter"))
    manifest["design"].update(
        active_arms=["original"],
        fresh_debate_both_assignments=False,
        independent_question_count=questions,
        E_calibration_cases=0,
    )
    # Calibration batteries remain separate supplementary experiments.
    if members == 3:
        for plan in manifest["plans"]:
            plan["calibration_arm"] = None
    manifest["execution"].update(
        max_inflight_requests=32,
        question_workers=8,
        openrouter_transport_retries="Known HTTP connection/timeouts only; at most two additional identical attempts; unknown bills retained",
    )
    manifest["openrouter_run"] = {
        "version": VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "member_count": members,
        "question_selection": "First requested questions in the frozen original plan; no output-based selection",
        "active_tone_assignment": "original",
        "tone_unit": "turn, not global",
        "endpoint": BASE_URL,
        "model_mapping": MODELS,
        "reuse_previous_generations": False,
        "private_position_feedback": False,
        "scope": "A/B/C and E1, no D or separate E2 follow-up" if members == 2 else "A/B/E1/E2, no C/D or calibration battery",
    }
    graph_type = StatefulGraph if members == 2 else TriadicGraph
    graphs = [graph_type(contexts[p["question_id"]], p, manifest) for p in manifest["plans"]]
    counts = Counter(t.purpose for g in graphs for t in g.tasks.values())
    manifest["planned_counts"] = {
        "questions": questions,
        "tone_assignments": 1,
        "branch_paths": sum(len(g.route.leaves) for g in graphs),
        "formal_replies": counts["debate"],
        "initial_answers": counts["initial"],
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
        "D_calls": 0,
    }
    return json.loads(canonical(manifest)), contexts


class OpenRouterMock:
    """Synthetic response fixture; never passed to a live run."""
    def __init__(self, members):
        self.base = StatefulMockProvider() if members == 2 else TriadicMockProvider()

    def generate(self, request):
        value = self.base.generate(request)
        return replace(value, raw={"provider": "openrouter", "mock": True, "response": {
            "id": "synthetic-" + digest(request.key)[:16], "usage": {
                "cost": 0, "prompt_tokens": value.input_tokens, "completion_tokens": value.output_tokens,
            }
        }})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, choices=(2, 3), default=2)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--roster", choices=("same_family", "same_model"), default="same_family")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--mock", action="store_true")
    parser.add_argument("--question-limit", type=int, help="Dispatch only this many frozen questions as an initial gate")
    parser.add_argument("--max-inflight", type=int, default=32)
    args = parser.parse_args()
    manifest, contexts = prepare(members=args.members, questions=args.questions, roster=args.roster, mock=args.mock)
    print(canonical({"scope": manifest["openrouter_run"], "planned_counts": manifest["planned_counts"]}), flush=True)
    if not (args.live or args.mock):
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = run_dyadic if args.members == 2 else run_triadic
    options = {"provider_factory": (lambda: OpenRouterMock(args.members)) if args.mock else OpenRouterProvider}
    if args.members == 2:
        options["token_count"] = lambda model, text: len(text.split())  # No D/token-matched filler in this roster.
    report = runner(manifest, contexts, output, question_limit=args.question_limit or args.questions,
                    request_limit=args.max_inflight, workers=8, **options)
    if report["status"] not in ("completed", "completed_with_failures", "preflight_completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
