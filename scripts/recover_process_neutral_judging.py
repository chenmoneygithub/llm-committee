"""Sequential, identical-request recovery of the frozen E2 judge-only run."""

from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from llm_committee.pivot import prompts, storage
from llm_committee.pivot.databricks_provider import DatabricksProvider
from llm_committee.pivot.failures import LocalTaskFailure
from llm_committee.pivot.models import digest
from scripts.grok_quality_check import request_from
from scripts.prepare_process_neutral_judging import OUTPUT
from scripts.run_process_neutral_judging import CAP_USD, execution_report, prepare_execution
from scripts.scale_public_history import ROOT, freeze, read, sha


def attempt_hashes(output):
    with sqlite3.connect(f"file:{output}/requests.sqlite3?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute("SELECT * FROM calls ORDER BY key")]
    if any(row["status"] == "pending" for row in rows):
        raise ValueError("Pending requests require reconciliation before recovery")
    return {row["key"]: digest(row) for row in rows}


def prepare_recovery(output=OUTPUT):
    output = Path(output)
    manifest, tasks = prepare_execution(output)
    path = output / "recovery-manifest.json"
    current = attempt_hashes(output)
    policy = {
        "execution_manifest_sha256": sha(output / "execution-manifest.json"),
        "implementation": str(Path(__file__).relative_to(ROOT)),
        "implementation_sha256": sha(Path(__file__)),
        "max_inflight_requests": 1,
        "transport_extension": {
            "closed_provider": "databricks",
            "model": "grok-4-6",
            "purpose": "judge_e",
            "status": "uncertain",
            "response": None,
            "errors": ["ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"],
        },
        "retry_policy": "Same original limit of two additional identical attempts; first valid response",
        "accounting": "Keep original journal, cap and all unknown reservations unchanged",
    }
    if path.exists():
        saved = read(path)
        if {k: v for k, v in saved.items() if k != "prior_attempt_hashes"} != policy:
            raise ValueError("Frozen recovery configuration changed")
        if any(current.get(k) != v for k, v in saved["prior_attempt_hashes"].items()):
            raise ValueError("Pre-recovery attempt was changed or removed")
    else:
        freeze(path, {**policy, "prior_attempt_hashes": current})
    return manifest, tasks


@contextmanager
def grok_transport_recovery():
    original = storage.local_transport_record

    def scoped(request, row, closed_provider):
        return original(request, row, closed_provider) or (
            closed_provider == "databricks"
            and request.get("model") == "grok-4-6"
            and request.get("purpose") == "judge_e"
            and row["status"] == "uncertain"
            and row.get("response") is None
            and row.get("error") in ("ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout")
        )

    storage.local_transport_record = scoped
    try:
        yield
    finally:
        storage.local_transport_record = original


def recover(output=OUTPUT, limit=400, provider_factory=DatabricksProvider):
    output = Path(output)
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, tasks = prepare_recovery(output)
        if not 1 <= limit <= len(tasks):
            raise ValueError("Invalid request limit")
        journal = storage.Journal(output / "requests.sqlite3", manifest, CAP_USD)
        try:
            with grok_transport_recovery():
                for task in tasks[:limit]:
                    provider = provider_factory()
                    try:
                        journal.call(
                            request_from(task["request"]), provider, lambda s: prompts.parse_json(s, prompts.E_SCHEMA)
                        )
                        print(
                            json.dumps(
                                {"event": "completed", "model": task["request"]["model"], "key": task["request"]["key"]}
                            ),
                            flush=True,
                        )
                    except LocalTaskFailure as exc:
                        print(json.dumps({"event": "missing", **exc.document()}), flush=True)
                    finally:
                        provider.close()
        finally:
            journal.close()
            prepare_recovery(output)
            result = execution_report(output)
            print(json.dumps(result), flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.live:
        recover(args.output)
    else:
        prepare_recovery(args.output)
        print(json.dumps({"status": "recovery_ready_not_dispatched", "model_calls": 0}))
