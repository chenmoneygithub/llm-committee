"""Rolling questions with optional globally bounded, dependency-ready A–E requests."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import tarfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from .continuation import load_checkpoint, reuse_diagnostic_retry, validate_attempts
from .failures import (
    FORMAT_RETRY_POLICY,
    LocalTaskFailure,
    local_transport_record,
    retry_statistics,
    retryable_format_record,
)
from .models import PilotConfig, canonical
from .providers import LiveProviders, MockProvider
from .questions import load_approved_bank
from .reanalysis import OfflineProvider
from .runner import expected_counts, manifest_for, run_pilot
from .storage import Journal, RunBlocked, RunStopped

UTC = timezone.utc  # noqa: UP017 -- retain Python 3.10 compatibility


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare_study(
    bank,
    *,
    workers=3,
    mock=False,
    no_budget_limit=False,
    cap_usd=None,
    checkpoint=None,
    reconcile_identical_json_duplicates=False,
    max_inflight_requests=None,
    retry_format_failures=False,
    diagnostic_retry=None,
):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Choose 1–8 question workers")
    if not mock and (no_budget_limit == (cap_usd is not None)):
        raise ValueError("Choose an explicit spending cap OR --no-budget-limit")
    if cap_usd is not None and (not math.isfinite(cap_usd) or cap_usd <= 0):
        raise ValueError("Spending cap must be positive and finite, or use --no-budget-limit")
    if reconcile_identical_json_duplicates and checkpoint is None:
        raise ValueError("JSON reconciliation requires --continue-from")
    if max_inflight_requests is not None and (
        type(max_inflight_requests) is not int or not 1 <= max_inflight_requests <= 64
    ):
        raise ValueError("Choose 1–64 global in-flight requests")
    if retry_format_failures and max_inflight_requests is None:
        raise ValueError("Failure isolation requires --max-inflight-requests")
    if diagnostic_retry is not None and (checkpoint is None or not retry_format_failures):
        raise ValueError("Diagnostic reuse requires a checkpoint and the format-retry policy")
    questions, provenance = load_approved_bank(bank)
    config = PilotConfig(roster="mixed_family", judge_model="gemini-3.8-flash")
    manifest = manifest_for(questions, config, mock=mock)
    if not mock:
        manifest["kind"] = "paid_main_study"
    manifest["study"] = {
        "scope": "approved_bank_mixed_family",
        "question_bank": provenance,
        "workers": workers,
        "concurrency_unit": "question; all dependencies within a question remain serial",
        "independent_repetitions": 1,
    }
    if max_inflight_requests is not None:
        manifest["study"].update(
            scheduler="ready_nodes_v1",
            max_inflight_requests=max_inflight_requests,
            concurrency_unit="rolling questions and dependency-ready nodes; one global request pool",
        )
    manifest["execution"].update(
        budget_policy="no_limit_user_requested" if no_budget_limit else "capped",
        spend_cap_usd=None if no_budget_limit else 1000.0 if mock else cap_usd,
        stop_on_error=True,
    )
    if retry_format_failures:
        manifest["execution"].update(
            failure_policy=FORMAT_RETRY_POLICY, stop_on_error="except_exhausted_retryable_failures"
        )
    inherited = None
    if checkpoint is not None:
        inherited = load_checkpoint(
            checkpoint,
            manifest,
            allow_question_expansion=True,
            reconcile_identical_json_duplicates=reconcile_identical_json_duplicates,
        )
        if diagnostic_retry is not None:
            inherited = reuse_diagnostic_retry(diagnostic_retry, inherited)
        manifest["continuation"] = inherited[0]
    return questions, config, manifest, inherited


def native_token_count():
    # Tokenizers only; this replay helper never opens a provider session.
    from .tinker_provider import make_renderer

    tokenizers = {}

    def count(model, text):
        if model not in tokenizers:
            tokenizers[model] = make_renderer(model, "none")[0]
        return len(tokenizers[model].encode(text, add_special_tokens=False))

    return count


def run_study(questions, config, manifest, inherited, output, *, provider_factory=None, replay_token_counter=None):
    """One process lock and durable journal; bounded questions and, optionally, ready nodes."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunBlocked("Another study process already owns this output directory") from exc
        return _run_locked(
            questions,
            config,
            manifest,
            inherited,
            output,
            provider_factory=provider_factory,
            replay_token_counter=replay_token_counter,
        )


def _run_locked(questions, config, manifest, inherited, output, *, provider_factory, replay_token_counter):
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RunBlocked("Study manifest changed; do not overwrite or silently resume")
    else:
        atomic_json(path, manifest)
    snapshot = output / "implementation-start.tar.gz"
    if not snapshot.exists():
        with tarfile.open(snapshot, "x:gz") as archive:
            for source in sorted(Path(__file__).parent.glob("*.py")):
                archive.add(source, arcname=f"llm_committee/pivot/{source.name}")
    # The whole approved bank, including exclusions and review notes, remains auditable.
    bank = Path(manifest["study"]["question_bank"]["path"])
    bank_bytes = bank.read_bytes()
    expected_sha = manifest["study"]["question_bank"].get("sha256")
    if expected_sha and hashlib.sha256(bank_bytes).hexdigest() != expected_sha:
        raise RunBlocked("Approved bank changed after planning; no model calls made")
    if not (output / "question-bank.json").exists():
        (output / "question-bank.json").write_bytes(bank_bytes)
    elif expected_sha and hashlib.sha256((output / "question-bank.json").read_bytes()).hexdigest() != expected_sha:
        raise RunBlocked("Frozen question bank does not match the run manifest")
    stop = threading.Event()
    print_lock = threading.Lock()

    def emit(event):
        with print_lock:
            print(canonical({"time_utc": datetime.now(UTC).isoformat(), **event}), flush=True)

    cap = manifest["execution"]["spend_cap_usd"]
    root = Journal(output / "requests.sqlite3", manifest, cap, progress=emit)
    if provider_factory is None:
        provider_factory = (
            MockProvider
            if manifest["kind"] == "offline_mock"
            else lambda: LiveProviders(
                judge=True,
                mixed=True,
                debate_effort=config.debate_effort,
                closed_provider=config.closed_provider,
                databricks_profile=config.databricks_profile,
            )
        )
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop.set())
    completed, failures, scheduler_stats, graphs = {}, {}, {}, {}
    planned = {p["question_id"]: p for p in manifest["plans"]}

    def view(question):
        return {
            **manifest,
            "questions": [next(q for q in manifest["questions"] if q["id"] == question.id)],
            "plans": [planned[question.id]],
            "question_fingerprints": [question.fingerprint],
        }

    def save_result(question, result, *, reused=False):
        result["interpretation"] = (
            "Approved-bank mixed-family study; synthetic outputs are test data only. "
            "Descriptive measurement records, not statistical inference."
        )
        result["probability_readout"] = manifest["probability_readout"]
        atomic_json(output / "questions" / f"{question.id}.json", result)
        return {"status": result["status"], "report": f"questions/{question.id}.json", "replayed": reused}

    def save_progress(status):
        imported = root.db.execute("SELECT value FROM metadata WHERE key='imported_keys'").fetchone()
        old_keys = set(json.loads(imported[0])) if imported else set()
        rows = root.db.execute("SELECT key,status,charge FROM calls").fetchall()
        new = [(key, state, cost) for key, state, cost in rows if key not in old_keys]
        report = {
            "status": status,
            "kind": manifest["kind"],
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "planned_counts": expected_counts(manifest),
            "workers": manifest["study"]["workers"],
            **(
                {
                    "max_inflight_requests": manifest["study"]["max_inflight_requests"],
                    "scheduler": manifest["study"]["scheduler"],
                    **scheduler_stats,
                }
                if "max_inflight_requests" in manifest["study"]
                else {}
            ),
            "completed_questions": len(completed),
            "fully_successful_questions": sum(r["status"] == "completed" for r in completed.values()),
            "questions_with_missing_data": sum(r["status"] == "completed_with_failures" for r in completed.values()),
            "question_reports": dict(completed),
            "failed_questions": dict(failures),
            "reused_checkpoint_records": len(old_keys),
            "new_call_status_counts": dict(Counter(state for _, state, _ in new)),
            "new_charged_or_reserved_usd": sum(cost for _, _, cost in new),
            "prior_charged_or_reserved_usd": manifest.get("continuation", {}).get("prior_charged_or_reserved_usd", 0),
            "budget_policy": manifest["execution"]["budget_policy"],
            "spend_cap_usd": cap,
            "cost_note": "Token-based estimate including unresolved reservations, not a provider invoice",
        }
        if graphs:
            from .outcomes import outcome_summary, task_support, trajectory_outcomes

            outcomes = [row for graph in graphs.values() for row in trajectory_outcomes(graph)]
            report["trajectories"] = outcome_summary(outcomes)
            report["task_failures"] = {k: v for g in graphs.values() for k, v in g.failed.items()}
            totals = {}
            for graph in graphs.values():
                for purpose, counts in task_support(graph).items():
                    aggregate = totals.setdefault(purpose, Counter())
                    aggregate.update(counts)
            report["task_support"] = totals
            attempts = root.db.execute("SELECT key,request,status FROM calls").fetchall()
            report.update(retry_statistics(attempts, report["task_failures"]))
            report["transport_missing_requests"] = sum(
                v["status"] == "failed_after_transport_retries" for v in report["task_failures"].values()
            )
            atomic_json(
                output / "trajectory-outcomes.json",
                {
                    "summary": report["trajectories"],
                    "trajectories": outcomes,
                    "note": "Debate paths only; measurement failures reported separately. Shared-prefix failures can affect multiple paths; paths are not independent failure events.",
                },
            )
        atomic_json(output / "progress.json", report)
        return report

    def worker(question):
        if stop.is_set():
            return {"status": "cancelled", "reason": "Stopped before question start"}
        provider = journal = None
        session_status = "errored"
        try:
            journal = Journal(output / "requests.sqlite3", manifest, cap, progress=emit, should_stop=stop.is_set)
            provider = provider_factory()
            emit({"event": "question_started", "question_id": question.id})
            result = run_pilot((question,), config, view(question), journal, provider)
            if result["status"] != "completed":
                raise RunBlocked("Question finished with measurement failures")
            saved = save_result(question, result)
            session_status = "success"
            return saved
        except RunStopped as exc:
            return {"status": "cancelled", "reason": str(exc)}
        except Exception as exc:
            stop.set()
            emit(
                {
                    "event": "question_failed",
                    "question_id": question.id,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
            return {"status": "blocked", "reason": str(exc), "error_type": type(exc).__name__}
        finally:
            if journal is not None:
                journal.close()
            if provider is not None and hasattr(provider, "close"):
                try:
                    provider.close(status=session_status)
                except Exception as exc:
                    emit({"event": "cleanup_failed", "question_id": question.id, "error_type": type(exc).__name__})

    try:
        if inherited is not None:
            root.import_checkpoint(*inherited)
        imported = root.db.execute("SELECT value FROM metadata WHERE key='imported_keys'").fetchone()
        old_keys = set(json.loads(imported[0])) if imported else set()
        all_rows = [
            dict(zip(("key", "request", "request_hash", "status", "response", "error"), row, strict=True))
            for row in root.db.execute("SELECT key,request,request_hash,status,response,error FROM calls")
        ]
        validate_attempts(all_rows, allow_retries=root.format_retries, closed_provider=root.closed_provider)
        unresolved = [
            (r["key"], r["status"])
            for r in all_rows
            if r["status"] != "completed"
            and r["key"] not in old_keys
            and not (
                root.format_retries
                and (
                    retryable_format_record(json.loads(r["request"]), r)
                    or local_transport_record(json.loads(r["request"]), r, root.closed_provider)
                )
            )
        ]
        if unresolved:
            raise RunBlocked(f"Existing run has unresolved requests; no paid replay: {unresolved[0]}")
        save_progress("preflight_replay")
        reused_ids = set(manifest.get("continuation", {}).get("study_extension", {}).get("reused_question_ids", []))
        counter = replay_token_counter or (
            MockProvider().token_count if manifest["kind"] == "offline_mock" else native_token_count()
        )
        dag_mode = manifest["study"].get("scheduler") == "ready_nodes_v1"
        if dag_mode:
            from .taskgraph import QuestionGraph

            graphs = {q.id: QuestionGraph(q, config, planned[q.id], manifest) for q in questions}

        def graph_report(graph):
            if not graph.successful:
                from .outcomes import partial_report

                return partial_report(graph, root, counter)
            # A no-failure run retains the existing canonical report and exact-request verification.
            previous_stop = root.should_stop
            root.should_stop = lambda: True
            try:
                return run_pilot((graph.question,), config, view(graph.question), root, OfflineProvider(counter))
            finally:
                root.should_stop = previous_stop

        # Replay cached prefixes before paid calls; a partial question is not a completed one.
        root.should_stop = lambda: True
        for question in questions:
            has_report = (output / "questions" / f"{question.id}.json").exists()
            if dag_mode:
                graph = graphs[question.id]
                graph.restore(root, OfflineProvider(counter))
                if not graph.complete:
                    if has_report:
                        raise RunBlocked("Completed question report lacks saved graph requests")
                    if graph.values:
                        emit(
                            {
                                "event": "question_partially_replayed",
                                "question_id": question.id,
                                "api_calls": 0,
                                "verified_saved_nodes": len(graph.values),
                            }
                        )
                    continue
            if dag_mode or question.id in reused_ids or has_report:
                try:
                    result = (
                        graph_report(graphs[question.id])
                        if dag_mode
                        else run_pilot((question,), config, view(question), root, OfflineProvider(counter))
                    )
                except RunStopped:
                    if has_report:
                        raise RunBlocked(
                            "Completed question report lacks saved requests; inspect the journal"
                        ) from None
                    emit({"event": "question_partially_replayed", "question_id": question.id, "api_calls": 0})
                    continue
                if result["status"] not in ("completed", "completed_with_failures"):
                    raise RunBlocked("Offline question replay has measurement failures")
                completed[question.id] = save_result(question, result, reused=True)
                emit({"event": "question_replayed", "question_id": question.id, "api_calls": 0})
        root.should_stop = stop.is_set
        pending_questions = iter(q for q in questions if q.id not in completed)
        save_progress("running")
        emit(
            {
                "event": "study_started",
                "questions": len(questions),
                "replayed_questions": len(completed),
                "workers": manifest["study"]["workers"],
                "spend_cap_usd": cap,
                **({"max_inflight_requests": manifest["study"]["max_inflight_requests"]} if dag_mode else {}),
            }
        )
        if dag_mode:
            from .scheduling import run_ready_pool

            local = threading.local()
            providers = []
            providers_lock = threading.Lock()

            def execute(request, parse):
                journal = None
                try:
                    if stop.is_set():
                        raise RunStopped("Run stopped before request worker initialization")
                    if not hasattr(local, "provider"):
                        local.provider = provider_factory()
                        with providers_lock:
                            providers.append(local.provider)
                    # Each connection is created, used and closed on its worker thread.
                    journal = Journal(
                        output / "requests.sqlite3", manifest, cap, progress=emit, should_stop=stop.is_set
                    )
                    return journal.call(request, local.provider, parse)
                except LocalTaskFailure:
                    raise  # Coordinator isolates the task and its true dependents; no global stop.
                except Exception:
                    stop.set()
                    raise
                finally:
                    if journal is not None:
                        journal.close()

            def graph_done(graph):
                result = graph_report(graph)
                completed[graph.question.id] = save_result(graph.question, result)
                emit(
                    {
                        "event": "question_completed",
                        "question_id": graph.question.id,
                        "completed_questions": len(completed),
                        "status": result["status"],
                    }
                )

            def graph_error(graph, exc):
                qid = graph.question.id
                if isinstance(exc, LocalTaskFailure):
                    emit(
                        {
                            "event": "task_missing",
                            "question_id": qid,
                            **graph.failed[exc.key],
                            "blocked_tasks": len(graph.blocked),
                        }
                    )
                    return
                cancelled = isinstance(exc, RunStopped)
                if not cancelled or qid not in failures:
                    failures[qid] = {
                        "status": "cancelled" if cancelled else "blocked",
                        "reason": str(exc),
                        "error_type": type(exc).__name__,
                    }
                    emit(
                        {
                            "event": "question_cancelled" if cancelled else "question_failed",
                            "question_id": qid,
                            **failures[qid],
                        }
                    )

            last_progress = 0.0

            def graph_progress(stats):
                nonlocal last_progress
                scheduler_stats.update(stats)
                if stop.is_set() or time.monotonic() - last_progress >= 1:
                    save_progress("stopping" if stop.is_set() else "running")
                    last_progress = time.monotonic()

            try:
                scheduler_stats.update(
                    run_ready_pool(
                        (graphs[q.id] for q in pending_questions),
                        question_limit=manifest["study"]["workers"],
                        request_limit=manifest["study"]["max_inflight_requests"],
                        token_count=counter,
                        execute=execute,
                        stop=stop,
                        on_start=lambda g: emit({"event": "question_started", "question_id": g.question.id}),
                        on_done=graph_done,
                        on_error=graph_error,
                        on_progress=graph_progress,
                    )
                )
            finally:
                # No request remains in flight after run_ready_pool's executor has drained.
                def close_provider(provider):
                    if hasattr(provider, "close"):
                        try:
                            provider.close(
                                status="errored" if failures else "interrupted" if stop.is_set() else "success"
                            )
                        except Exception as exc:
                            emit({"event": "cleanup_failed", "error_type": type(exc).__name__})

                with ThreadPoolExecutor(max_workers=min(8, len(providers)) or 1) as cleanup:
                    list(cleanup.map(close_provider, providers))
        else:
            with ThreadPoolExecutor(max_workers=manifest["study"]["workers"]) as pool:
                active = {}

                def fill():
                    while not stop.is_set() and len(active) < manifest["study"]["workers"]:
                        question = next(pending_questions, None)
                        if question is None:
                            break
                        active[pool.submit(worker, question)] = question

                fill()
                while active:
                    done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                    for future in done:
                        question = active.pop(future)
                        result = future.result()
                        if result["status"] == "completed":
                            completed[question.id] = result
                            emit(
                                {
                                    "event": "question_completed",
                                    "question_id": question.id,
                                    "completed_questions": len(completed),
                                }
                            )
                        else:
                            failures[question.id] = result
                            stop.set()
                    save_progress("stopping" if stop.is_set() else "running")
                    fill()
        status = (
            ("completed_with_failures" if any(g.failed for g in graphs.values()) else "completed")
            if len(completed) == len(questions)
            else "blocked"
            if any(v["status"] == "blocked" for v in failures.values())
            else "stopped"
        )
        report = save_progress(status)
        atomic_json(output / "report.json", report)
        emit({"event": "study_finished", **report})
        return report
    except BaseException as exc:
        stop.set()
        report = save_progress("blocked")
        report["reason"] = str(exc)
        atomic_json(output / "report.json", report)
        raise
    finally:
        root.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
