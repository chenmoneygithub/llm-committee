"""Run independent paired feedback forks, reusing bounded scheduling and durable retries.

python -m llm_committee.pivot.forced_feedback_run --source RUN --output OUT --mock
Replace --mock with --live for the explicitly authorized supplementary study.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import tarfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .failures import LocalTaskFailure, retry_statistics
from .forced_feedback import FeedbackGraph, prepare
from .models import canonical, digest
from .providers import LiveProviders, MockProvider
from .reanalysis import OfflineProvider
from .scheduling import run_ready_pool
from .storage import Journal, RunBlocked, RunStopped
from .study import atomic_json, native_token_count

UTC = timezone.utc  # noqa: UP017 -- keep compatibility with the shared Python 3.10 contracts


def run(
    manifest, contexts, output, *, case_limit=150, request_limit=64, workers=8, provider_factory=None, token_count=None
):
    if not 1 <= case_limit <= 150 or not 1 <= request_limit <= 64 or not 1 <= workers <= 8:
        raise ValueError("Invalid case/concurrency limit")
    if digest(contexts) != manifest["source"]["contexts_sha256"]:
        raise ValueError("Source context snapshot changed")
    output = output.resolve()
    source = Path(manifest["source"]["directory"])
    if output == source or source in output.parents:
        raise ValueError("Supplement must not write inside its archived source")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunBlocked("Another supplementary process owns this directory") from exc
        for name, value in (("manifest.json", manifest), ("source-contexts.json", contexts)):
            path = output / name
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise RunBlocked(f"Frozen {name} differs; no silent overwrite")
            else:
                atomic_json(path, value)
        snapshot = output / "implementation-start.tar.gz"
        if not snapshot.exists():
            with tarfile.open(snapshot, "x:gz") as archive:
                for path in sorted(Path(__file__).parent.glob("*.py")):
                    archive.add(path, arcname=f"llm_committee/pivot/{path.name}")
        return _run(manifest, contexts, output, case_limit, request_limit, workers, provider_factory, token_count)


def _run(manifest, contexts, output, limit, request_limit, workers, provider_factory, token_count):
    stop, log_lock = threading.Event(), threading.Lock()
    started = time.monotonic()

    def emit(event):
        with log_lock:
            print(canonical({"time_utc": datetime.now(UTC).isoformat(), **event}), flush=True)

    root = Journal(output / "requests.sqlite3", manifest, None, progress=emit)
    mock = manifest["kind"] == "offline_mock"
    if token_count is None:
        token_count = MockProvider().token_count if mock else native_token_count()
    if provider_factory is None:
        provider_factory = (
            MockProvider
            if mock
            else lambda: LiveProviders(
                judge=True,
                mixed=True,
                debate_effort=manifest["config"]["debate_effort"],
                closed_provider=manifest["config"]["closed_provider"],
                databricks_profile=manifest["config"]["databricks_profile"],
            )
        )
    selected = manifest["cases"][:limit]
    qids = list(dict.fromkeys(c["question_id"] for c in selected))
    graphs = {
        qid: FeedbackGraph(contexts[qid], [c for c in selected if c["question_id"] == qid], manifest) for qid in qids
    }
    completed, fatal, stats = {}, {}, {}
    local, providers, providers_lock = threading.local(), [], threading.Lock()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop.set())

    def capture_cases():
        for graph in graphs.values():
            for case in graph.cases:
                if case["id"] in completed:
                    continue
                prefix = graph.key(case, "")
                keys = {k for k in graph.tasks if k.startswith(prefix)}
                finished = graph.values.keys() | graph.failed.keys() | graph.blocked.keys()
                if not keys <= finished:
                    continue
                record = graph.case_report(case, token_count)
                atomic_json(output / "cases" / f"{case['id']}.json", record)
                completed[case["id"]] = record["status"]
                emit(
                    {
                        "event": "case_completed",
                        "case_id": case["id"],
                        "status": record["status"],
                        "completed_cases": len(completed),
                        "target_cases": limit,
                    }
                )

    def save(status):
        capture_cases()
        rows = root.db.execute("SELECT key,request,status,charge FROM calls").fetchall()
        failed = {k: v for graph in graphs.values() for k, v in graph.failed.items()}
        report = {
            "kind": manifest["kind"],
            "protocol_version": manifest["protocol_version"],
            "status": status,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_this_process_seconds": round(time.monotonic() - started, 2),
            "target_this_phase": limit,
            "planned_counts": manifest["planned_counts"],
            "completed_cases": len(completed),
            "successful_cases": sum(s == "completed" for s in completed.values()),
            "case_statuses": completed.copy(),
            "call_status_counts": dict(Counter(row[2] for row in rows)),
            "charged_or_reserved_usd": sum(row[3] for row in rows),
            "cost_note": "Token-based estimate, including retries/reservations; not a provider invoice",
            "max_inflight_requests": request_limit,
            "question_workers": workers,
            "task_failures": failed,
            "fatal_errors": fatal.copy(),
            **stats,
            **retry_statistics([(k, req, status) for k, req, status, cost in rows], failed),
        }
        atomic_json(output / "progress.json", report)
        return report

    def execute(request, parse):
        journal = None
        try:
            if stop.is_set():
                raise RunStopped("Stopped before dispatch")
            if not hasattr(local, "provider"):
                local.provider = provider_factory()
                with providers_lock:
                    providers.append(local.provider)
            journal = Journal(output / "requests.sqlite3", manifest, None, progress=emit, should_stop=stop.is_set)
            return journal.call(request, local.provider, parse)
        except LocalTaskFailure:
            raise
        except Exception:
            stop.set()
            raise
        finally:
            if journal:
                journal.close()

    def on_error(graph, exc):
        if isinstance(exc, LocalTaskFailure):
            emit({"event": "task_missing", "question_id": graph.question.id, **exc.document()})
        else:
            fatal[graph.question.id] = {"type": type(exc).__name__, "reason": str(exc)}
            emit({"event": "fatal_error", "question_id": graph.question.id, **fatal[graph.question.id]})

    last_save = 0.0

    def progress(current):
        nonlocal last_save
        stats.update(current)
        if time.monotonic() - last_save >= 1 or stop.is_set():
            save("stopping" if stop.is_set() else "running")
            last_save = time.monotonic()

    try:
        root.should_stop = lambda: True
        for graph in graphs.values():
            graph.restore(root, OfflineProvider(token_count))
        root.should_stop = stop.is_set
        save("running")
        emit(
            {"event": "supplement_started", "cases": limit, "total_cases": 150, "max_inflight_requests": request_limit}
        )
        stats.update(
            run_ready_pool(
                (g for g in graphs.values() if not g.complete),
                question_limit=workers,
                request_limit=request_limit,
                token_count=token_count,
                execute=execute,
                stop=stop,
                on_start=lambda g: emit({"event": "question_started", "question_id": g.question.id}),
                on_done=lambda g: capture_cases(),
                on_error=on_error,
                on_progress=progress,
            )
        )
        capture_cases()
        status = (
            "blocked"
            if fatal
            else "stopped"
            if len(completed) < limit
            else "completed_with_failures"
            if any(s != "completed" for s in completed.values())
            else "preflight_completed"
            if limit < 150
            else "completed"
        )
        report = save(status)
        atomic_json(output / "report.json", report)
        emit(
            {
                "event": "supplement_finished",
                "status": status,
                "completed_cases": len(completed),
                "charged_or_reserved_usd": report["charged_or_reserved_usd"],
            }
        )
        return report
    except BaseException as exc:
        stop.set()
        fatal["run"] = {"type": type(exc).__name__, "reason": str(exc)}
        atomic_json(output / "report.json", save("blocked"))
        raise
    finally:

        def close(provider):
            if hasattr(provider, "close"):
                try:
                    provider.close(status="interrupted" if stop.is_set() else "success")
                except Exception as exc:
                    emit({"event": "cleanup_failed", "type": type(exc).__name__})

        with ThreadPoolExecutor(max_workers=min(8, len(providers)) or 1) as pool:
            list(pool.map(close, providers))
        root.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--cases", type=int, default=150, help="First presampled cases; 6 is the engineering gate")
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest, contexts = prepare(args.source, mock=args.mock)
    report = run(
        manifest, contexts, args.output, case_limit=args.cases, request_limit=args.max_inflight, workers=args.workers
    )
    if report["status"] not in ("completed", "preflight_completed", "completed_with_failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
