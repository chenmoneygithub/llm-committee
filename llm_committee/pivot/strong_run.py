"""Run the fresh strongly/leaning pilot with both continuations in one frozen journal."""

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
from datetime import UTC, datetime
from pathlib import Path

from .failures import LocalTaskFailure, retry_statistics
from .models import canonical, digest
from .providers import LiveProviders
from .reanalysis import OfflineProvider
from .scheduling import run_ready_pool
from .storage import Journal, RunBlocked, RunStopped
from .strong_study import QUESTION_COUNT, StrongGraph, StrongMockProvider, prepare
from .study import atomic_json, native_token_count


def run(
    manifest,
    contexts,
    output,
    *,
    question_limit=QUESTION_COUNT,
    request_limit=64,
    workers=8,
    provider_factory=None,
    token_count=None,
    graph_type=StrongGraph,
    mock_provider_type=StrongMockProvider,
    event_namespace="strong_study",
    report_enricher=None,
):
    if manifest.get("analysis_only"):
        raise ValueError("Merged report views cannot dispatch or resume model calls")
    if not 1 <= question_limit <= len(manifest["plans"]) or not 1 <= request_limit <= 64 or not 1 <= workers <= 8:
        raise ValueError("Invalid question/concurrency limit")
    if digest(contexts) != manifest["source"]["contexts_sha256"]:
        raise ValueError("Frozen initial contexts changed")
    output = output.resolve()
    source = Path(manifest["source"]["directory"])
    if output == source or source in output.parents:
        raise ValueError("Never write inside the archived source")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunBlocked("Another process owns this dyadic run") from exc
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
        return _run(
            manifest,
            contexts,
            output,
            question_limit,
            request_limit,
            workers,
            provider_factory,
            token_count,
            graph_type,
            mock_provider_type,
            event_namespace,
            report_enricher,
        )


def _run(
    manifest,
    contexts,
    output,
    limit,
    request_limit,
    workers,
    provider_factory,
    token_count,
    graph_type,
    mock_provider_type,
    event_namespace,
    report_enricher,
):
    stop, log_lock = threading.Event(), threading.Lock()
    started = time.monotonic()

    def emit(event):
        with log_lock:
            print(canonical({"time_utc": datetime.now(UTC).isoformat(), **event}), flush=True)

    root = Journal(output / "requests.sqlite3", manifest, None, progress=emit)
    mock = manifest["kind"] == "offline_mock"
    if token_count is None:
        token_count = mock_provider_type().token_count if mock else native_token_count()
    if provider_factory is None:
        provider_factory = (
            mock_provider_type
            if mock
            else lambda: LiveProviders(
                judge=True,
                mixed=manifest["config"]["roster"] == "mixed_family",
                debate_effort=manifest["config"]["debate_effort"],
                closed_provider=manifest["config"]["closed_provider"],
                databricks_profile=manifest["config"]["databricks_profile"],
            )
        )
    plans = manifest["plans"][:limit]
    graphs = {p["question_id"]: graph_type(contexts[p["question_id"]], p, manifest) for p in plans}
    completed, fatal, stats = {}, {}, {}
    local, providers, providers_lock = threading.local(), [], threading.Lock()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop.set())

    def capture(graph):
        if graph.question.id in completed or not graph.complete:
            return
        record = graph.report(token_count)
        atomic_json(output / "questions" / f"{graph.question.id}.json", record)
        completed[graph.question.id] = record
        emit(
            {
                "event": "question_completed",
                "question_id": graph.question.id,
                "status": record["status"],
                "completed_questions": len(completed),
                "target_questions": limit,
            }
        )

    def save(status):
        rows = root.db.execute("SELECT key,request,status,charge FROM calls").fetchall()
        failures = {k: v for graph in graphs.values() for k, v in graph.failed.items()}
        trajectories = [t for record in completed.values() for t in record["trajectories"]]
        report = {
            "kind": manifest["kind"],
            "protocol_version": manifest["protocol_version"],
            "status": status,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_this_process_seconds": round(time.monotonic() - started, 2),
            "target_this_phase": limit,
            "planned_counts": manifest["planned_counts"],
            "completed_questions": len(completed),
            "successful_questions": sum(r["status"] == "completed" for r in completed.values()),
            "question_statuses": {qid: r["status"] for qid, r in completed.items()},
            "trajectory_statuses": dict(Counter(t["status"] for t in trajectories)),
            "call_status_counts": dict(Counter(row[2] for row in rows)),
            "charged_or_reserved_usd": sum(row[3] for row in rows),
            "cost_note": "Token-based estimate including retries/reservations, not a provider invoice",
            "max_inflight_requests": request_limit,
            "question_workers": workers,
            "task_failures": failures,
            "fatal_errors": fatal.copy(),
            **stats,
            **retry_statistics([(k, req, s) for k, req, s, cost in rows], failures),
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
            capture(graph)
        root.should_stop = stop.is_set
        save("running")
        emit(
            {
                "event": f"{event_namespace}_started",
                "questions": limit,
                "total_questions": len(manifest["plans"]),
                "max_inflight_requests": request_limit,
            }
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
                on_done=capture,
                on_error=on_error,
                on_progress=progress,
            )
        )
        for graph in graphs.values():
            capture(graph)
        status = (
            "blocked"
            if fatal
            else (
                "stopped"
                if len(completed) < limit
                else (
                    "completed_with_failures"
                    if any(r["status"] != "completed" for r in completed.values())
                    else "preflight_completed" if limit < len(manifest["plans"]) else "completed"
                )
            )
        )
        report = save(status)
        if report_enricher:
            # Final summaries belong inside run.lock, not in an unlocked wrapper
            # that could overwrite a concurrently resumed process's report.
            report = report_enricher(report, output)
            atomic_json(output / "progress.json", report)
        atomic_json(output / "report.json", report)
        emit(
            {
                "event": f"{event_namespace}_finished",
                "status": status,
                "completed_questions": len(completed),
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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--questions", type=int, default=QUESTION_COUNT, help="First frozen pilot questions; 1 is the engineering gate"
    )
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest, contexts = prepare(args.plan, mock=args.mock)
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
