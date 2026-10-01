"""Private HLE 50-question continuation; medium first, logged Qwen low fallback.

The original 20-question journal and implementation remain immutable. Dataset
selection is outcome-blind. Neither answer keys nor fallback metadata enter prompts.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from llm_committee.pivot.failures import LocalTaskFailure, retry_statistics
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.providers import LiveProviders
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.scheduling import run_ready_pool
from llm_committee.pivot.storage import Journal, RunStopped
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import SuperGPQAMockProvider, freeze
from scripts.hle_diamond_study import (
    BANK as OLD_BANK,
)
from scripts.hle_diamond_study import (
    ROOT,
    PairedGraph,
    implementation_hash,
    read,
    sha,
)
from scripts.hle_diamond_study import (
    RUN as OLD_RUN,
)
from scripts.hle_diamond_study import (
    prepare as prepare_base,
)

VERSION = "hle-diamond-scale50-medium-low-fallback-20260929-v1"
MODEL = "Qwen/Qwen3.8-27B"
LOW_SUFFIX = "::low-fallback"
SEED = 20260929
PLAN = ROOT / "runs/hle-diamond-50-20260929-plan"
RUN = ROOT / "runs/hle-diamond-50-20260929/live"
PAGE = RUN.parent / "hle-diamond-50.html"
EXTENSION_TEMPLATE = ROOT / "runs/public-history-extension30-20260928/plans/dyadic-plan.json"
FALLBACK = {
    "model": MODEL,
    "primary_effort": "medium",
    "primary_max_attempts": 3,
    "trigger": "All ordinary attempts failed; final Qwen initial/debate response exhausted its output length",
    "fallback_effort": "low",
    "fallback_max_attempts": 3,
    "unchanged": "messages, temperature, schema, output limit; D1 always reasoning-off",
    "selection": "First format-valid response, irrespective of correctness or substantive answer",
    "failure": "Keep all attempts; mark only exhausted task and actual dependents missing",
    "diagnostic_low_samples": "Not imported; fresh fallback under this frozen continuation policy",
}


def code_hash():
    return digest({p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("hle_diamond_scale_report.py"))})


def select_extension(bank, *, count=30, seed=SEED):
    old = [r["id"] for r in bank["selected"]]
    pool = bank["pool_ids"]
    if len(set(old)) != len(old) or len(set(pool)) != len(pool) or not set(old) <= set(pool):
        raise ValueError("Invalid frozen question IDs")
    new = sorted(random.Random(seed).sample(sorted(set(pool) - set(old)), count))
    return {
        "seed": seed,
        "existing_ids": old,
        "new_ids": new,
        "method": "Retain all original 20; uniformly sample 30 from the remaining 55 IDs, no outcomes or redraw",
    }


def download_extension(plan=PLAN):
    plan = Path(plan).resolve()
    old = read(OLD_BANK)
    selection = select_extension(old)
    freeze(plan / "selection.json", selection)
    bank_path = plan / "question-bank.json"
    if bank_path.exists():
        bank = read(bank_path)
        if bank["selection"] != selection:
            raise ValueError("Frozen selection changed")
        return bank_path
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, get_token

    fs = HfFileSystem(token=get_token())
    source = f"datasets/{old['dataset']}@{old['revision']}/data/test-00000-of-00001.parquet"
    with fs.open(source, "rb") as stream:
        rows = (
            pq.ParquetFile(stream)
            .read(columns=["id", "question", "answer", "answer_type", "partition", "category"])
            .to_pylist()
        )
    pool = sorted((r for r in rows if r["id"] in set(old["pool_ids"])), key=lambda r: r["id"])
    if len(pool) != 75 or digest(pool) != old["pool_content_sha256"]:
        raise ValueError("Pinned eligible-pool content differs; do not redraw")
    by_id = {r["id"]: r for r in pool}
    if any(by_id[r["id"]] != r for r in old["selected"]):
        raise ValueError("Original question content changed")
    freeze(
        bank_path,
        {
            **old,
            "selection": selection,
            "selected": [*old["selected"], *(by_id[k] for k in selection["new_ids"])],
        },
    )
    return bank_path


def checkpoint(source):
    source = Path(source).resolve()
    with closing(sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM calls ORDER BY key")]
    if any(r["status"] not in ("completed", "invalid") or r["response"] is None for r in rows):
        raise ValueError("Source has unresolved attempts; inspect before import")
    for r in rows:
        if digest(json.loads(r["request"])) != r["request_hash"]:
            raise ValueError("Source request hash mismatch")
    files = [source / "manifest.json", source / "source-contexts.json", source / "progress.json"]
    files += sorted((source / "questions").glob("*.json"))
    return {
        "source_directory": str(source),
        "import_rows_sha256": digest(rows),
        "source_files": {str(p): sha(p) for p in files},
        "source_cost_estimate_usd": sum(r["charge"] for r in rows),
        "successful_requests_reused": sum(r["status"] == "completed" for r in rows),
        "imported_attempts": len(rows),
        "import_cost": "zero new inference cost; historical cost retained separately",
    }, rows


def prepare(bank_path, *, plan=PLAN, source=OLD_RUN, template=EXTENSION_TEMPLATE):
    plan, source, bank_path, template = (Path(p).resolve() for p in (plan, source, bank_path, template))
    old = read(source / "manifest.json")
    if old["implementation_sha256"] != implementation_hash():
        raise ValueError("Original implementation changed")
    bank = read(bank_path)
    old_contexts = read(source / "source-contexts.json")
    old_ids = {v["source_id"] for v in old_contexts.values()}
    selected = bank["selected"]
    if not old_ids <= {r["id"] for r in selected} or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Cannot discard or duplicate pilot questions")
    extension_bank = {**bank, "selected": [r for r in selected if r["id"] not in old_ids]}
    freeze(plan / "extension-question-bank.json", extension_bank)
    extension, new_contexts = prepare_base(
        plan / "extension-question-bank.json", mock=old["kind"] == "offline_mock", template_file=template
    )
    if old["config"] != extension["config"]:
        raise ValueError("Configuration changed beyond authorized fallback")
    # Preserve the original ID-to-tone mapping, rather than re-zipping all 50 IDs.
    manifest, contexts = copy.deepcopy(old), {**old_contexts, **new_contexts}
    manifest["plans"] = [*old["plans"], *extension["plans"]]
    if len(contexts) != len(manifest["plans"]):
        raise ValueError("Overlapping extension")
    manifest.update(scale_protocol_version=VERSION, scale_implementation_sha256=code_hash())
    manifest["source"].update(
        directory=str(plan),
        bank_file=str(bank_path),
        bank_sha256=sha(bank_path),
        contexts_sha256=digest(contexts),
        extension_tone_template=str(template),
        extension_tone_template_sha256=sha(template),
    )
    n = len(contexts)
    manifest["design"].update(
        scope=f"{n}-question text-only reasoning MCQ study; not the full HLE benchmark",
        same_GOQA_tone_instructions_and_20_schedules=False,
        same_GOQA_tone_instructions_and_50_schedules=True,
    )
    manifest["execution"]["qwen_low_fallback"] = FALLBACK
    manifest["planned_counts"] = {
        "questions": n,
        "trajectories": n * 6,
        "logical_calls": n * 42,
        "by_purpose": {"initial": n * 3, "debate": n * 18, "d_choice": n * 12, "synthesis": n * 9},
    }
    descriptor, rows = checkpoint(source)
    manifest["continuation"] = descriptor
    freeze(plan / "manifest.json", manifest)
    freeze(plan / "source-contexts.json", contexts)
    return manifest, contexts, rows


class FallbackJournal(Journal):
    """Identical retries first; separate immutable low request only on a length failure."""

    def __init__(self, path, manifest, cap_usd, **kwargs):
        if manifest.get("execution", {}).get("qwen_low_fallback") != FALLBACK:
            raise ValueError("Missing or changed authorized fallback policy")
        super().__init__(path, manifest, cap_usd, **kwargs)

    def call(self, request, provider, parse):
        try:
            return super().call(request, provider, parse)
        except LocalTaskFailure as medium_failure:
            if not (
                request.model == MODEL
                and request.effort == "medium"
                and request.purpose in ("initial", "debate")
                and not request.candidate_labels
                and request.scoring is None
            ):
                raise
            row = self.db.execute(
                "SELECT status,response,error FROM calls WHERE key=?", (medium_failure.attempts[-1],)
            ).fetchone()
            response = json.loads(row[1]) if row and row[1] else {}
            raw = response.get("raw") or {}
            if not (
                row
                and row[0] == "invalid"
                and row[2] == "Response status: incomplete"
                and response.get("status") == "incomplete"
                and raw.get("provider") == "tinker"
                and raw.get("model") == MODEL
                and raw.get("effort") == "medium"
                and raw.get("stop_reason") == "length"
                and response.get("output_tokens") == request.max_output_tokens
            ):
                raise
            low = replace(request, key=request.key + LOW_SUFFIX, effort="low")
            if self.progress:
                self.progress({"event": "qwen_low_fallback", "logical_key": request.key, "low_key": low.key})
            try:
                return super().call(low, provider, parse)
            except LocalTaskFailure as low_failure:
                raise LocalTaskFailure(
                    request.key,
                    [*medium_failure.attempts, *low_failure.attempts],
                    low_failure.reason,
                    kind="failed_after_low_fallback",
                ) from None


def all_rows(directory):
    with closing(sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM calls ORDER BY key")]


def fallback_summary(rows):
    selected = [r for r in rows if json.loads(r["request"])["key"].endswith(LOW_SUFFIX)]
    keys = {json.loads(r["request"])["key"] for r in selected}
    successes = {json.loads(r["request"])["key"] for r in selected if r["status"] == "completed"}
    return {
        "triggered_requests": len(keys),
        "recovered_requests": len(successes),
        "low_attempts": len(selected),
        "recovered_by_purpose": dict(
            Counter(json.loads(r["request"])["purpose"] for r in selected if r["status"] == "completed")
        ),
        "recovered_logical_keys": sorted(k.removesuffix(LOW_SUFFIX) for k in successes),
    }


def verify(manifest, contexts):
    if (
        manifest["implementation_sha256"] != implementation_hash()
        or manifest["scale_implementation_sha256"] != code_hash()
    ):
        raise ValueError("Frozen implementation changed")
    if manifest["scale_protocol_version"] != VERSION or digest(contexts) != manifest["source"]["contexts_sha256"]:
        raise ValueError("Frozen protocol or contexts changed")
    source = manifest["source"]
    for path_key, hash_key in (
        ("bank_file", "bank_sha256"),
        ("tone_template", "tone_template_sha256"),
        ("extension_tone_template", "extension_tone_template_sha256"),
    ):
        if sha(source[path_key]) != source[hash_key]:
            raise ValueError("Frozen source changed")
    for path, expected in manifest["continuation"]["source_files"].items():
        if sha(path) != expected:
            raise ValueError("Original pilot changed")


def run(manifest, contexts, output, *, question_limit=None, provider_factory=None, request_limit=32, workers=8):
    """Same bounded ready-node scheduler as the pilot, with an explicit journal subclass."""
    output = Path(output).resolve()
    verify(manifest, contexts)
    if output == Path(manifest["source"]["directory"]) or output == Path(manifest["continuation"]["source_directory"]):
        raise ValueError("Never write into source or plan")
    limit = question_limit or len(manifest["plans"])
    if not 1 <= limit <= len(manifest["plans"]):
        raise ValueError("Invalid question limit")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, data in (("manifest.json", manifest), ("source-contexts.json", contexts)):
            freeze(output / name, data)
        for path in (Path(__file__), Path(__file__).with_name("hle_diamond_scale_report.py")):
            frozen = output / path.name
            if frozen.exists():
                if sha(frozen) != sha(path):
                    raise ValueError("Runner snapshot changed")
            else:
                frozen.write_bytes(path.read_bytes())
        descriptor, rows = checkpoint(manifest["continuation"]["source_directory"])
        if descriptor != manifest["continuation"]:
            raise ValueError("Checkpoint changed")
        with_journal = FallbackJournal(output / "requests.sqlite3", manifest, None)
        try:
            with_journal.import_checkpoint(descriptor, rows)
        finally:
            with_journal.close()
        return execute(manifest, contexts, output, limit, provider_factory, request_limit, workers)


def execute(manifest, contexts, output, limit, provider_factory, request_limit, workers):
    stop, emit_lock = threading.Event(), threading.Lock()
    started = time.monotonic()

    def emit(event):
        with emit_lock:
            print(canonical({"time_utc": datetime.now(UTC).isoformat(), **event}), flush=True)

    if provider_factory is None:
        provider_factory = (
            SuperGPQAMockProvider
            if manifest["kind"] == "offline_mock"
            else lambda: LiveProviders(
                judge=False, mixed=True, debate_effort="medium", closed_provider="databricks", databricks_profile="un"
            )
        )
    root = FallbackJournal(output / "requests.sqlite3", manifest, None, progress=emit)
    graphs = {p["question_id"]: PairedGraph(contexts[p["question_id"]], p, manifest) for p in manifest["plans"][:limit]}
    completed, fatal, stats = {}, {}, {}
    local, providers, provider_lock = threading.local(), [], threading.Lock()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop.set())

    def capture(graph):
        if graph.question.id not in completed and graph.complete:
            record = graph.report(None)
            atomic_json(output / "questions" / f"{graph.question.id}.json", record)
            completed[graph.question.id] = record
            emit(
                {
                    "event": "question_completed",
                    "question_id": graph.question.id,
                    "status": record["status"],
                    "completed_questions": len(completed),
                }
            )

    def save(status):
        root.db.row_factory = sqlite3.Row
        rows = [dict(r) for r in root.db.execute("SELECT key,request,status,charge FROM calls")]
        failures = {k: v for g in graphs.values() for k, v in g.failed.items()}
        report = {
            "status": status,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "target_this_phase": limit,
            "planned_counts": manifest["planned_counts"],
            "completed_questions": len(completed),
            "successful_questions": sum(r["status"] == "completed" for r in completed.values()),
            "elapsed_this_process_seconds": time.monotonic() - started,
            "task_failures": failures,
            "fatal_errors": fatal.copy(),
            "call_status_counts": dict(Counter(r["status"] for r in rows)),
            "charged_or_reserved_usd": sum(r["charge"] for r in rows),
            "original_pilot_cost_estimate_usd": manifest["continuation"]["source_cost_estimate_usd"],
            "qwen_low_fallback": fallback_summary(rows),
            **stats,
            **retry_statistics([(r["key"], r["request"], r["status"]) for r in rows], failures),
        }
        atomic_json(output / "progress.json", report)
        return report

    def call(request, parse):
        journal = None
        try:
            if stop.is_set():
                raise RunStopped("Stopped before dispatch")
            if not hasattr(local, "provider"):
                local.provider = provider_factory()
                with provider_lock:
                    providers.append(local.provider)
            journal = FallbackJournal(
                output / "requests.sqlite3", manifest, None, progress=emit, should_stop=stop.is_set
            )
            return journal.call(request, local.provider, parse)
        except LocalTaskFailure:
            raise
        except Exception:
            stop.set()
            raise
        finally:
            if journal:
                journal.close()

    def error(graph, exc):
        if isinstance(exc, LocalTaskFailure):
            emit({"event": "task_missing", "question_id": graph.question.id, **exc.document()})
        else:
            fatal[graph.question.id] = {"type": type(exc).__name__, "reason": str(exc)}
            emit({"event": "fatal_error", **fatal[graph.question.id]})

    last_save = 0.0

    def progress(current):
        nonlocal last_save
        stats.update(current)
        if time.monotonic() - last_save >= 2 or stop.is_set():
            save("stopping" if stop.is_set() else "running")
            last_save = time.monotonic()

    try:
        root.should_stop = lambda: True
        for graph in graphs.values():
            graph.restore(root, OfflineProvider(lambda *_: 0))
            capture(graph)
        root.should_stop = stop.is_set
        save("running")
        emit({"event": "hle_scale_started", "questions": limit, "max_inflight_requests": request_limit})
        stats.update(
            run_ready_pool(
                (g for g in graphs.values() if not g.complete),
                question_limit=workers,
                request_limit=request_limit,
                token_count=lambda *_: 0,
                execute=call,
                stop=stop,
                on_start=lambda g: emit({"event": "question_started", "question_id": g.question.id}),
                on_done=capture,
                on_error=error,
                on_progress=progress,
            )
        )
        status = (
            "blocked"
            if fatal
            else "stopped"
            if len(completed) < limit
            else (
                "completed_with_failures"
                if any(r["status"] != "completed" for r in completed.values())
                else "preflight_completed"
                if limit < len(manifest["plans"])
                else "completed"
            )
        )
        result = save(status)
        atomic_json(output / "report.json", result)
        emit({"event": "hle_scale_finished", "status": status, "completed_questions": len(completed)})
        return result
    except BaseException as exc:
        stop.set()
        fatal["run"] = {"type": type(exc).__name__, "reason": str(exc)}
        save("blocked")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        root.close()
        for provider in providers:
            if hasattr(provider, "close"):
                try:
                    provider.close(status="interrupted" if stop.is_set() else "success")
                except Exception as exc:
                    emit({"event": "provider_cleanup_warning", "type": type(exc).__name__})


def audit(directory, *, native=True):
    directory = Path(directory).resolve()
    manifest, contexts = read(directory / "manifest.json"), read(directory / "source-contexts.json")
    verify(manifest, contexts)
    rows = all_rows(directory)
    saved = {}
    native_counts, renderers = Counter(), {}
    for row in rows:
        req = json.loads(row["request"])
        if digest(req) != row["request_hash"]:
            raise ValueError("Request hash changed")
        if row["status"] != "completed":
            continue
        logical = req["key"].removesuffix(LOW_SUFFIX)
        assert logical not in saved, "More than one accepted answer for a logical task"
        saved[logical] = req
        payload = canonical(req["messages"])
        assert '"answer_letter"' not in payload and "your_current_position" not in payload
        response, parsed = json.loads(row["response"]), json.loads(row["parsed"])
        if req["candidate_labels"]:
            assert req["effort"] == "none" and response["reasoning_tokens"] == 0
            if native and manifest["kind"] != "offline_mock":
                from llm_committee.pivot.tinker_provider import make_renderer
                from scripts.audit_stateful_dyadic import check_native

                if req["model"] not in renderers:
                    renderers[req["model"]] = make_renderer(req["model"], "none")
                check_native(req, response, parsed, *renderers[req["model"]])
                native_counts[req["model"]] += 1
        elif req["key"].endswith(LOW_SUFFIX):
            assert req["model"] == MODEL and req["effort"] == "low"
            if native and manifest["kind"] != "offline_mock":
                from llm_committee.pivot.tinker_provider import make_renderer

                if "low" not in renderers:
                    renderers["low"] = make_renderer(MODEL, "low")
                _, renderer = renderers["low"]
                messages = [
                    {"role": "system" if m["role"] == "developer" else m["role"], "content": m["text"]}
                    for m in req["messages"]
                ]
                assert renderer.build_generation_prompt(messages).to_ints() == response["raw"]["prompt_token_ids"]
                native_counts["qwen_low_prompts"] += 1
        else:
            assert req["effort"] == "medium"
    journal = FallbackJournal(directory / "requests.sqlite3", manifest, None, should_stop=lambda: True)
    questions = requests = 0
    try:
        for plan in manifest["plans"]:
            record_file = directory / "questions" / f"{plan['question_id']}.json"
            if not record_file.exists():
                continue
            graph = PairedGraph(contexts[plan["question_id"]], plan, manifest)
            graph.restore(journal, OfflineProvider(lambda *_: 0))
            assert graph.complete and canonical(graph.report(None)) == canonical(read(record_file))
            questions += 1
            requests += len(graph.values)
            for arm in ("original", "alternate"):
                for node in graph.route.nodes:
                    formal, probe = (
                        saved.get(graph.key(arm, f"debate/{node.id}")),
                        saved.get(graph.key(arm, f"D1/{node.id}")),
                    )
                    if formal and probe:
                        assert formal["messages"][:-1] == probe["messages"][:-1]
    finally:
        journal.close()
    result = {
        "questions_reconstructed": questions,
        "requests_reconstructed": requests,
        "successful_requests": len(saved),
        "native_checks": dict(native_counts),
        "qwen_low_fallback": fallback_summary(rows),
        "source_20_unchanged": True,
    }
    atomic_json(directory / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "live", "launch", "report-only", "audit-only"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--questions", type=int, default=50)
    args = parser.parse_args()
    from scripts.hle_diamond_scale_report import publish

    if args.report_only:
        result = publish(RUN, PAGE)
        print(canonical({"html": str(PAGE), "coverage": result["coverage"]}))
        return
    if args.audit_only:
        print(canonical(audit(RUN)))
        return
    if args.launch:
        # Require a previously tested, frozen plan; launching never reselects questions.
        manifest, contexts = read(PLAN / "manifest.json"), read(PLAN / "source-contexts.json")
        verify(manifest, contexts)
        RUN.parent.mkdir(parents=True, exist_ok=True)
        with (RUN.parent / "run.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "scripts.hle_diamond_scale", "--live", "--questions", str(args.questions)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(canonical({"pid": proc.pid, "log": str(RUN.parent / "run.log"), "html": str(PAGE)}))
        return
    if (PLAN / "manifest.json").exists():
        manifest, contexts = read(PLAN / "manifest.json"), read(PLAN / "source-contexts.json")
        verify(manifest, contexts)
    else:
        manifest, contexts, _ = prepare(download_extension())
    print(
        canonical(
            {
                "planned": manifest["planned_counts"],
                "reused": manifest["continuation"]["successful_requests_reused"],
                "fallback": FALLBACK,
            }
        ),
        flush=True,
    )
    if args.prepare:
        return
    try:
        result = run(manifest, contexts, RUN, question_limit=args.questions)
    finally:
        if (RUN / "progress.json").exists():
            publish(RUN, PAGE)
    checked = audit(RUN)
    print(canonical({"html": str(PAGE), "audit": checked}), flush=True)
    if result["status"] not in ("completed", "completed_with_failures", "preflight_completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
