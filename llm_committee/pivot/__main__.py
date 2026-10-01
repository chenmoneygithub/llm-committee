"""CLI: offline by default; live execution requires an explicit spending policy."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from .budget import estimate_budget
from .continuation import load_checkpoint
from .models import ROSTERS, PilotConfig, canonical
from .providers import LiveProviders, MockProvider
from .questions import import_screened_archive, load_questions, validate_input_review
from .runner import expected_counts, manifest_for, require_executable, run_pilot
from .storage import Journal, RunBlocked


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("study-plan", "study-dry-run", "study-run"):
        command = commands.add_parser(name, help="Approved 60-question mixed-family study")
        command.add_argument("--bank", type=Path, required=True)
        command.add_argument("--workers", type=int, default=3)
        command.add_argument(
            "--max-inflight-requests",
            type=int,
            help="Enable dependency-ready branch/tone/measurement scheduling with one global request cap (1–64)",
        )
        command.add_argument("--continue-from", type=Path)
        command.add_argument(
            "--retry-format-failures",
            action="store_true",
            help="At most two identical retries for malformed output or known connection/timeouts; isolate only after exhaustion",
        )
        command.add_argument(
            "--reuse-diagnostic-retry",
            type=Path,
            help="Import an already paid diagnostic retry after validating its provenance",
        )
        command.add_argument(
            "--reconcile-identical-json-duplicates",
            action="store_true",
            help="With --continue-from, accept identical duplicate JSON fields in saved responses; no regeneration",
        )
        if name != "study-plan":
            command.add_argument("--output", type=Path, required=True)
        budget = command.add_mutually_exclusive_group(required=name == "study-run")
        budget.add_argument("--no-budget-limit", action="store_true", help="Explicitly run without a spending stop")
        budget.add_argument("--approve-spend-usd", type=float)
    reanalyze = commands.add_parser("reanalyze-topk", help="Recompute saved D reads with zero-fill; no API calls")
    reanalyze.add_argument("--source", type=Path, required=True)
    reanalyze.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("import-questions", help="Reuse archived screening; no inference")
    prepare.add_argument("--screening", type=Path, required=True)
    prepare.add_argument("--records", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--count", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=20260925)
    prepare.add_argument(
        "--indices", type=int, nargs="+", help="Explicit eligible archive indices, not effect-based selection"
    )
    prepare.add_argument(
        "--exclude-fingerprints",
        type=Path,
        help="Optional explicit exclusions; pilot questions are NOT automatically excluded from the main study",
    )
    for name in ("plan", "estimate", "dry-run", "run"):
        command = commands.add_parser(name)
        command.add_argument("--questions", type=Path, required=True)
        command.add_argument("--seed", type=int, default=20260925)
        command.add_argument("--roster", choices=list(ROSTERS), default="mixed_family")
        command.add_argument(
            "--judge-model",
            choices=["gemini-3.8-flash"],
            default=None,
            help="Explicit selection; omitted means no B/C-text/E judging",
        )
        command.add_argument("--debate-effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
        command.add_argument("--cache", action="store_true", help="Enable explicit prompt-cache breakpoints")
        command.add_argument("--closed-provider", choices=["databricks", "direct"], default="databricks")
        command.add_argument("--databricks-profile", default="un", help="Existing workspace OAuth profile")
        if name not in ("plan", "estimate"):
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--continue-from", type=Path, help="Exact-request checkpoint reuse in a NEW directory")
            command.add_argument(
                "--reconcile-databricks-usage",
                action="store_true",
                help="With --continue-from, reparse saved Gemini usage against total tokens; no generation",
            )
            command.add_argument(
                "--record-probability-failures",
                action="store_true",
                help="Engineering diagnostic: retain score mismatches as missing, continue independent work; never retries",
            )
        if name == "run":
            command.add_argument(
                "--approve-spend-usd",
                type=float,
                required=True,
                help="Explicit live API spend authorization for this run, e.g. 30",
            )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command.startswith("study-"):
        from .study import prepare_study, run_study

        try:
            prepared = prepare_study(
                args.bank,
                workers=args.workers,
                mock=args.command != "study-run",
                no_budget_limit=args.no_budget_limit,
                cap_usd=args.approve_spend_usd,
                checkpoint=args.continue_from,
                reconcile_identical_json_duplicates=args.reconcile_identical_json_duplicates,
                max_inflight_requests=args.max_inflight_requests,
                retry_format_failures=args.retry_format_failures,
                diagnostic_retry=args.reuse_diagnostic_retry,
            )
            if args.command == "study-plan":
                print(
                    json.dumps(
                        {
                            "counts": expected_counts(prepared[2]),
                            "budget": estimate_budget(prepared[2]),
                            "bank_sha256": prepared[2]["study"]["question_bank"]["sha256"],
                            "api_calls": 0,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            result = run_study(*prepared, args.output)
            return 0 if result["status"] in ("completed", "completed_with_failures") else 2
        except (RunBlocked, ValueError, ImportError) as exc:
            print(canonical({"status": "blocked", "reason": str(exc)}), flush=True)
            return 2
    if args.command == "reanalyze-topk":
        from .reanalysis import reanalyze_topk
        from .tinker_provider import make_renderer

        tokenizers = {}

        def token_count(model, text):
            if model not in tokenizers:
                tokenizers[model] = make_renderer(model, "none")[0]
            return len(tokenizers[model].encode(text, add_special_tokens=False))

        result = reanalyze_topk(args.source, args.output, token_counter=token_count)
        print(canonical({"status": result["status"], "api_calls": 0, "output": str(args.output.resolve())}))
        return 0
    if args.command == "import-questions":
        excluded = set(json.loads(args.exclude_fingerprints.read_text())) if args.exclude_fingerprints else set()
        document = import_screened_archive(
            args.screening,
            args.records,
            count=args.count,
            seed=args.seed,
            excluded_fingerprints=excluded,
            indices=tuple(args.indices) if args.indices is not None else None,
        )
        write_new(args.output, document)
        print(
            canonical(
                {
                    "status": "questions_prepared_for_pilot",
                    "count": len(document["questions"]),
                    "path": str(args.output.resolve()),
                    "paid_calls": 0,
                }
            )
        )
        return 0
    questions = load_questions(args.questions)
    if not 1 <= len(questions) <= 10:
        raise ValueError("This pilot entry point accepts 1–10 questions only")
    config = PilotConfig(
        roster=args.roster,
        seed=args.seed,
        judge_model=args.judge_model,
        debate_effort=args.debate_effort,
        cache=args.cache,
        closed_provider=args.closed_provider,
        databricks_profile=args.databricks_profile,
    )
    mock = args.command != "run"
    manifest = manifest_for(questions, config, mock=mock)
    if args.command in ("plan", "estimate"):
        value = (
            {"counts": expected_counts(manifest), "manifest": manifest}
            if args.command == "plan"
            else estimate_budget(manifest)
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    try:
        require_executable(config)
        if not mock:
            validate_input_review(args.questions, questions)
        if args.record_probability_failures:
            manifest["execution"]["record_probability_failures"] = True
        checkpoint = None
        if args.reconcile_databricks_usage and not args.continue_from:
            raise ValueError("Offline usage reconciliation requires --continue-from")
        if args.continue_from:
            if args.continue_from.resolve() == args.output.resolve():
                raise ValueError("Continuation requires a new directory; source is read-only")
            checkpoint = load_checkpoint(
                args.continue_from, manifest, reconcile_databricks_usage=args.reconcile_databricks_usage
            )
            manifest["continuation"] = checkpoint[0]
    except (RunBlocked, ValueError) as exc:
        print(canonical({"status": "blocked", "reason": str(exc), "api_spend_usd": 0}))
        return 2  # Before creating outputs, clients, reading keys, or dispatching requests.
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Output directory contains a different manifest; choose a new directory")
    else:
        write_new(path, manifest)
    snapshot = args.output / "implementation-start.tar.gz"
    if not snapshot.exists():
        with tarfile.open(snapshot, "x:gz") as archive:
            for source in sorted(Path(__file__).parent.glob("*.py")):
                archive.add(source, arcname=f"llm_committee/pivot/{source.name}")
    cap = 1000.0 if mock else args.approve_spend_usd
    progress = None if mock else lambda event: print(canonical(event), flush=True)
    journal = Journal(args.output / "requests.sqlite3", manifest, cap, progress=progress)
    provider = None
    session_status = "interrupted"
    try:
        if checkpoint is not None:
            journal.import_checkpoint(*checkpoint)
        try:
            # Credentials are only accessed on the explicit live `run` path.
            provider = (
                MockProvider()
                if mock
                else LiveProviders(
                    judge=bool(config.judge_model),
                    mixed=config.roster == "mixed_family",
                    debate_effort=config.debate_effort,
                    closed_provider=config.closed_provider,
                    databricks_profile=config.databricks_profile,
                )
            )
            result = run_pilot(questions, config, manifest, journal, provider)
            session_status = "success" if result["status"] == "completed" else "errored"
            code = 0 if result["status"] == "completed" else 2
        except (RunBlocked, ImportError, ValueError) as exc:
            session_status = "errored"
            result = {
                "kind": manifest["kind"],
                "status": "blocked",
                "reason": str(exc),
                "charged_or_reserved_usd": journal.charged_usd,
            }
            code = 2
        # These are generated run artifacts. The journal remains the durable source of truth.
        (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (args.output / "calls.json").write_text(json.dumps(journal.audit(), ensure_ascii=False, indent=2) + "\n")
        print(
            canonical(
                {
                    "status": result["status"],
                    "kind": manifest["kind"],
                    "counts": expected_counts(manifest),
                    "output": str(args.output.resolve()),
                    "api_spend_usd": 0 if mock else journal.charged_usd,
                    "prior_continuation_cost_usd": manifest.get("continuation", {}).get(
                        "prior_charged_or_reserved_usd", 0.0
                    ),
                    "measurement_failures": len(result.get("measurement_failures", [])),
                    "reason": result.get("reason"),
                }
            )
        )
        return code
    finally:
        journal.close()
        if not mock and provider is not None:
            try:
                provider.close(status=session_status)
            except Exception as exc:
                # A cleanup failure must not replace the saved pilot outcome or its exit code.
                print(canonical({"event": "cleanup_failed", "error_type": type(exc).__name__}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
