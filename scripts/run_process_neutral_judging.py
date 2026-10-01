"""Explicit live runner for the already-frozen, judge-only E2 rubric revision."""

from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_committee.pivot import prompts
from llm_committee.pivot.databricks_provider import DatabricksProvider, databricks_payload
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY, LocalTaskFailure
from llm_committee.pivot.models import digest
from llm_committee.pivot.storage import Journal
from llm_committee.pivot.study import atomic_json
from scripts.grok_quality_check import register_provider, request_from, verify_source
from scripts.prepare_process_neutral_judging import OUTPUT, VERSION
from scripts.scale_public_history import ROOT, freeze, read, sha

CAP_USD = 8.0
WORKERS = 4


def verify_prepared(output):
    output = Path(output)
    manifest, tasks = read(output / "manifest.json"), read(output / "tasks.json")
    if manifest["version"] != VERSION or digest(tasks) != manifest["tasks_sha256"]:
        raise ValueError("Prepared judge revision changed")
    if sha(ROOT / "scripts/prepare_process_neutral_judging.py") != manifest["implementation_sha256"]:
        raise ValueError("Frozen request preparation changed")
    verify_source(manifest)
    previous = Path(manifest["previous_judge_directory"])
    for name, expected in manifest["previous_files_sha256"].items():
        if sha(previous / name) != expected:
            raise ValueError("Previous judging inputs changed")
    register_provider()
    for task in tasks:
        request = request_from(task["request"])
        if (
            request.purpose != "judge_e"
            or request.model not in ("gemini-3.8-flash", "grok-4-6")
            or not request.key.startswith(VERSION + "/")
        ):
            raise ValueError("Only the prepared Gemini/Grok final-answer judging is authorized")
        databricks_payload(request)
    return manifest, tasks


def prepare_execution(output=OUTPUT):
    output = Path(output)
    prepared, tasks = verify_prepared(output)
    paths = [
        Path(__file__),
        ROOT / "scripts/grok_quality_check.py",
        ROOT / "llm_committee/pivot/prompts.py",
        ROOT / "llm_committee/pivot/storage.py",
        ROOT / "llm_committee/pivot/failures.py",
        ROOT / "llm_committee/pivot/databricks_provider.py",
        ROOT / "llm_committee/pivot/providers.py",
    ]
    previous = Path(prepared["previous_judge_directory"])
    manifest = {
        "version": VERSION,
        "kind": "paid_E2_judge_rubric_sensitivity",
        "prepared_manifest_sha256": sha(output / "manifest.json"),
        "tasks_sha256": prepared["tasks_sha256"],
        "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "old_results_sha256": {name: sha(previous / name) for name in ("report.json", "requests.sqlite3")},
        "config": {"closed_provider": "databricks"},
        "execution": {"failure_policy": FORMAT_RETRY_POLICY, "max_inflight_requests": WORKERS},
        "cap_usd": CAP_USD,
        "planned_calls": len(tasks),
        "scope": prepared["scope"],
        "interpretation": prepared["interpretation"],
    }
    freeze(output / "execution-manifest.json", manifest)
    return manifest, tasks


def execution_report(output=OUTPUT):
    output = Path(output)
    prepared, tasks = verify_prepared(output)
    manifest = read(output / "execution-manifest.json")
    previous = Path(prepared["previous_judge_directory"])
    for name, expected in manifest["old_results_sha256"].items():
        if sha(previous / name) != expected:
            raise ValueError("Old judge results changed")
    expected = {t["request"]["key"]: t["request"] for t in tasks}
    completed, by_model, failed = {}, Counter(), []
    usage, unknown, attempts = 0.0, 0.0, 0
    with sqlite3.connect(f"file:{output}/requests.sqlite3?mode=ro", uri=True) as db:
        for key, raw, status, charge, parsed, error in db.execute(
            "SELECT key,request,status,charge,parsed,error FROM calls"
        ):
            attempts += 1
            request = json.loads(raw)
            if request != expected.get(request["key"]):
                raise ValueError("Journal contains an unplanned or changed request")
            if status in ("completed", "invalid", "received"):
                usage += charge
            else:
                unknown += charge
            if status == "completed":
                if request["key"] in completed:
                    raise ValueError("Multiple accepted results for a logical request")
                completed[request["key"]] = json.loads(parsed)
                by_model[request["model"]] += 1
            else:
                failed.append({"key": key, "status": status, "error": error, "charge": charge})
    report = {
        "version": VERSION,
        "status": "completed" if len(completed) == len(tasks) else "incomplete",
        "planned_calls": len(tasks),
        "completed_calls": len(completed),
        "by_model": dict(by_model),
        "attempts": attempts,
        "noncompleted_attempts": failed,
        "cost_accounting": {"received_response_estimate_usd": usage, "unresolved_reservations_usd": unknown},
        "source_unchanged": True,
        "old_judgments_unchanged": True,
    }
    atomic_json(output / "execution-report.json", report)
    return report


def run(output=OUTPUT, limit=400, provider_factory=DatabricksProvider):
    output = Path(output)
    manifest, tasks = prepare_execution(output)
    if not 1 <= limit <= len(tasks):
        raise ValueError("Invalid request limit")
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(output / "requests.sqlite3", manifest, CAP_USD)
        journal.close()
        stop = threading.Event()

        def execute(task):
            provider = provider_factory()
            journal = Journal(output / "requests.sqlite3", manifest, CAP_USD, should_stop=stop.is_set)
            try:
                journal.call(request_from(task["request"]), provider, lambda s: prompts.parse_json(s, prompts.E_SCHEMA))
                print(
                    json.dumps(
                        {"event": "completed", "model": task["request"]["model"], "key": task["request"]["key"]}
                    ),
                    flush=True,
                )
            except LocalTaskFailure as exc:
                print(json.dumps({"event": "missing", **exc.document()}), flush=True)
            except BaseException:
                stop.set()
                raise
            finally:
                journal.close()
                provider.close()

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(execute, tasks[:limit]))
        result = execution_report(output)
        print(json.dumps(result), flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()
    if args.live:
        run(args.output, args.limit)
    else:
        _, tasks = prepare_execution(args.output)
        print(json.dumps({"status": "ready_not_dispatched", "planned_calls": len(tasks), "model_calls": 0}))
