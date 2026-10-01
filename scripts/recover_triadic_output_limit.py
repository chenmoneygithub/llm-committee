"""Resume the stopped triadic batch in a new journal, preserving every old attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile
from pathlib import Path

from llm_committee.pivot.failures import local_output_limit_record
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.storage import Journal
from llm_committee.pivot.triadic_run import run
from llm_committee.pivot.triadic_study import TriadicGraph
from llm_committee.pivot.triadic_study import prepare as prepare_current
from scripts.scale_public_history import OUTPUT, freeze, read, sha

SOURCE = OUTPUT / "same_model/triadic"
DESTINATION = OUTPUT / "same_model/triadic-output-limit-recovery"
ALLOWED_RUNTIME_CHANGES = {
    "failures.py",
    "storage.py",
    "stateful_run.py",
    "strong_study.py",
    "triadic_study.py",
    "quality_study.py",
}


def source_rows(source):
    with sqlite3.connect(f"file:{source / 'requests.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM calls ORDER BY rowid")]


def verify_preservation(output):
    manifest = read(output / "manifest.json")
    provenance = manifest["recovery"]
    source = Path(provenance["source_directory"])
    for name, expected in provenance["source_file_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Recovery source changed: {name}")
    rows = source_rows(source)
    if digest(rows) != provenance["import_rows_sha256"]:
        raise ValueError("Recovery source rows changed")
    with sqlite3.connect(f"file:{output / 'requests.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in rows:
            actual = db.execute("SELECT * FROM calls WHERE key=?", (row["key"],)).fetchone()
            if actual is None or dict(actual) != row:
                raise ValueError(f"Imported attempt changed: {row['key']}")
        successful = [row for row in rows if row["status"] == "completed"]
        for row in successful:
            logical = json.loads(row["request"])["key"]
            count = db.execute(
                "SELECT COUNT(*) FROM calls WHERE json_extract(request,'$.key')=?", (logical,)
            ).fetchone()[0]
            if count != 1:
                raise ValueError("A previously successful request was dispatched again")
    return {
        "preserved_attempts": len(rows),
        "preserved_successful_requests": len(successful),
        "source_unchanged": True,
        "successful_requests_regenerated": 0,
    }


def prepare(source=SOURCE, output=DESTINATION):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Recovery must have its own sibling journal")
    previous = read(source / "manifest.json")
    contexts = read(source / "source-contexts.json")
    if read(source / "report.json")["status"] != "blocked":
        raise ValueError("Expected an inspected, stopped source batch")
    manifest, expected_contexts = prepare_current(
        previous["source"]["plan_file"], roster=previous["config"]["roster"], mock=previous["kind"] == "offline_mock"
    )
    for value in (previous, manifest):
        if value["protocol_version"] != "public-history-triadic-ABE-2026-09-28-v1":
            raise ValueError("Not the approved triadic protocol")

    def scientific(m):
        value = json.loads(canonical(m))
        value.pop("implementation_sha256")
        value["execution"].pop("output_limit_retry_policy", None)
        return value

    if scientific(previous) != scientific(manifest) or contexts != expected_contexts:
        raise ValueError("Recovery would change the frozen experiment")
    from llm_committee.pivot import triadic_study

    current = Path(triadic_study.__file__).parent
    with tarfile.open(source / "implementation-start.tar.gz") as tar:
        old_hashes = {
            Path(member.name).name: hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            for member in tar
            if member.isfile() and member.name.endswith(".py")
        }
    new_hashes = {p.name: sha(p) for p in current.glob("*.py")}
    if old_hashes.keys() != new_hashes.keys():
        raise ValueError("Unexpected implementation file additions/removals")
    changed = {name for name in old_hashes if old_hashes[name] != new_hashes[name]}
    if not changed <= ALLOWED_RUNTIME_CHANGES:
        raise ValueError(f"Unexpected change outside the runtime fix: {changed}")
    rows = source_rows(source)
    failed = []
    for row in rows:
        request = json.loads(row["request"])
        if row["request_hash"] != digest(request):
            raise ValueError("Stored request hash mismatch")
        if row["status"] == "completed":
            continue
        if not local_output_limit_record(request, row, previous["config"]["closed_provider"]):
            raise ValueError(f"Unrecognized source failure: {row['key']}")
        failed.append(row["key"])
    if not failed:
        raise ValueError("No inspected output-limit failure to recover")
    files = [
        source / name
        for name in (
            "manifest.json",
            "source-contexts.json",
            "report.json",
            "requests.sqlite3",
            "implementation-start.tar.gz",
        )
    ]
    files += sorted((source / "questions").glob("*.json"))
    manifest["recovery"] = {
        "source_directory": str(source),
        "source_file_sha256": {str(p.relative_to(source)): sha(p) for p in files},
        "import_rows_sha256": digest(rows),
        "prior_successful_requests": sum(r["status"] == "completed" for r in rows),
        "original_failure_keys": failed,
        "runtime_files_changed": sorted(changed),
        "original_implementation_sha256": previous["implementation_sha256"],
        "cost_accounting": "Carry source charges/reservations once into this replacement journal; never add the stopped journal again",
        "scientific_protocol_unchanged": True,
    }
    freeze(output / "manifest.json", manifest)
    freeze(output / "source-contexts.json", contexts)
    journal = Journal(output / "requests.sqlite3", manifest, None, should_stop=lambda: True)
    try:
        db = journal.db
        if db.execute("SELECT value FROM metadata WHERE key='recovery_import'").fetchone() is None:
            if db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]:
                raise ValueError("Cannot import into a nonempty recovery journal")
            db.execute("BEGIN IMMEDIATE")
            try:
                db.executemany(
                    "INSERT INTO calls VALUES(:key,:request,:request_hash,:status,:charge,:response,:parsed,:error)",
                    rows,
                )
                db.execute("INSERT INTO metadata VALUES('recovery_import',?)", (canonical(manifest["recovery"]),))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        for plan in manifest["plans"]:
            graph = TriadicGraph(contexts[plan["question_id"]], plan, manifest)
            graph.restore(journal, OfflineProvider(lambda model, text: len(text.split())))
            record = source / "questions" / f"{plan['question_id']}.json"
            if record.exists():
                if not graph.successful or canonical(graph.report(None)) != canonical(read(record)):
                    raise ValueError("Completed source question did not reconstruct exactly")
                target = output / "questions" / record.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(record, target)
    finally:
        journal.close()
    freeze(output / "recovery-preflight.json", verify_preservation(output))
    freeze(
        source.parent.parent / "recovery-overrides.json",
        {
            "replacements": [
                {
                    "roster": previous["config"]["roster"],
                    "setting": "triadic",
                    "original_output_30": str(source),
                    "replacement_output_30": str(output),
                    "replacement_manifest_sha256": sha(output / "manifest.json"),
                }
            ]
        },
    )
    return manifest, contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    manifest, contexts = prepare(args.source, args.output)
    print(json.dumps(read(args.output / "recovery-preflight.json")), flush=True)
    if args.live:
        result = run(
            manifest, contexts, args.output, question_limit=len(manifest["plans"]), request_limit=32, workers=8
        )
        freeze(args.output / "recovery-preservation-audit.json", verify_preservation(args.output))
        if result["status"] not in ("completed", "completed_with_failures"):
            raise SystemExit(1)
