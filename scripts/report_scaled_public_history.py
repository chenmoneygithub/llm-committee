"""Build analysis-only 20+30 views without altering either source journal."""

from __future__ import annotations

import argparse
import copy
import shutil
import sqlite3
from pathlib import Path

from bs4 import BeautifulSoup

from llm_committee.pivot.models import digest
from llm_committee.pivot.study import atomic_json
from scripts.scale_public_history import OUTPUT, ROOT, ROSTERS, execution_specs, freeze, read, sha


def sum_counts(items):
    result = {}
    for key in set().union(*(item.keys() for item in items)):
        values = [item[key] for item in items if key in item]
        if isinstance(values[0], dict):
            result[key] = sum_counts(values)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            result[key] = sum(values)
        else:
            raise ValueError(f"Not an additive count: {key}")
    return result


def validate_batches(manifests):
    first = manifests[0]
    for field in ("kind", "protocol_version", "prompt_version", "config", "agreement_rubric", "probability_readout"):
        if any(m.get(field) != first.get(field) for m in manifests):
            raise ValueError(f"Incompatible batches: {field}")
    ids = [p["question_id"] for m in manifests for p in m["plans"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping questions must not be double-counted")


def merge(sources, output, *, debate_parent=None):
    sources = [Path(p).resolve() for p in sources]
    output = Path(output).resolve()
    if any(output == source or source in output.parents or output in source.parents for source in sources):
        raise ValueError("Merged view must be separate from source journals")
    manifests = [read(p / "manifest.json") for p in sources]
    reports = [read(p / "report.json") for p in sources]
    validate_batches(manifests)
    if any(r["status"] not in ("completed", "completed_with_failures") for r in reports):
        raise ValueError("Only finished batches can be combined")
    for m, r in zip(manifests, reports, strict=True):
        if r["completed_questions"] != len(m["plans"]):
            raise ValueError("Incomplete batch")
    descriptors = [
        {
            "directory": str(p),
            "manifest_sha256": sha(p / "manifest.json"),
            "report_sha256": sha(p / "report.json"),
            "request_journal_sha256": sha(p / "requests.sqlite3"),
            "questions": len(m["plans"]),
            "implementation_sha256": m["implementation_sha256"],
        }
        for p, m in zip(sources, manifests, strict=True)
    ]
    # This marker is frozen before copying; rerunning only resumes this exact merge.
    freeze(output / "analysis-view.json", {"analysis_only": True, "batches": descriptors})
    contexts, record_hashes = {}, {}
    for source, manifest in zip(sources, manifests, strict=True):
        contexts.update(read(source / "source-contexts.json"))
        for plan in manifest["plans"]:
            qid = plan["question_id"]
            src, dst = source / "questions" / f"{qid}.json", output / "questions" / f"{qid}.json"
            if dst.exists() and sha(src) != sha(dst):
                raise ValueError(f"Record changed: {qid}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            record_hashes[qid] = sha(dst)
    manifest = copy.deepcopy(manifests[0])
    manifest.update(
        analysis_only=True,
        batches=descriptors,
        plans=[p for m in manifests for p in m["plans"]],
        planned_counts=sum_counts([m["planned_counts"] for m in manifests]),
        implementation_sha256=digest([d["implementation_sha256"] for d in descriptors]),
    )
    manifest["execution"]["dispatch_enabled"] = False
    manifest["design"]["independent_question_count"] = len(contexts)
    if "E_calibration_cases" in manifest["design"]:
        manifest["design"]["E_calibration_cases"] = sum(m["design"]["E_calibration_cases"] for m in manifests)
    if "calibration" in manifest["design"]:
        calibration = manifest["design"]["calibration"]
        calibration["count"] = sum(m["design"]["calibration"]["count"] for m in manifests)
        calibration["selection"] = {q: v for m in manifests for q, v in m["design"]["calibration"]["selection"].items()}
    manifest["source"] = {
        "directory": str(output),
        "batches": descriptors,
        "contexts_sha256": digest(contexts),
        "record_sha256": record_hashes,
    }
    if debate_parent is not None:
        parent = Path(debate_parent).resolve()
        # Verify each E batch was derived from its own unchanged debate source.
        for m in manifests:
            original = Path(m["source"]["directory"])
            assert sha(original / "manifest.json") == m["source"]["manifest_sha256"]
            for qid, expected in m["source"]["question_sha256"].items():
                assert sha(original / "questions" / f"{qid}.json") == expected
                assert sha(parent / "questions" / f"{qid}.json") == expected
        manifest["source"].update(
            directory=str(parent),
            manifest_sha256=sha(parent / "manifest.json"),
            question_sha256={
                p["question_id"]: sha(parent / "questions" / f"{p['question_id']}.json") for p in manifest["plans"]
            },
        )
    dbpath = output / "requests.sqlite3"
    if not dbpath.exists():
        # A temporary DB is atomically published; duplicate request keys are fatal.
        temporary = output / "requests.build.sqlite3"
        with sqlite3.connect(temporary) as target:
            target.execute(
                "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, request TEXT NOT NULL, request_hash TEXT NOT NULL, status TEXT NOT NULL, charge REAL NOT NULL, response TEXT, parsed TEXT, error TEXT)"
            )
            if target.execute("SELECT COUNT(*) FROM calls").fetchone()[0]:
                raise ValueError("Interrupted nonempty build requires inspection")
            for source in sources:
                with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as origin:
                    target.executemany(
                        "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
                        origin.execute(
                            "SELECT key,request,request_hash,status,charge,response,parsed,error FROM calls"
                        ),
                    )
        temporary.replace(dbpath)
    status = (
        "completed_with_failures" if any(r["status"] == "completed_with_failures" for r in reports) else "completed"
    )
    report = {
        "kind": manifest["kind"],
        "protocol_version": manifest["protocol_version"],
        "status": status,
        "analysis_only": True,
        "batches": descriptors,
        "planned_counts": manifest["planned_counts"],
        "completed_questions": len(contexts),
        "target_this_phase": len(contexts),
        "successful_questions": sum(r["successful_questions"] for r in reports),
        "charged_or_reserved_usd": sum(r["charged_or_reserved_usd"] for r in reports),
        "cost_note": "Sum of source batch token estimates, including retries; copying and reporting made no paid calls",
        "cache_usage": [row for r in reports for row in r.get("cache_usage", [])],
        "in_flight_requests": 0,
    }
    for key in ("question_statuses", "task_failures", "fatal_errors"):
        report[key] = {k: v for r in reports for k, v in r.get(key, {}).items()}
    for key in (
        "trajectory_statuses",
        "comparison_statuses",
        "quality_comparison_statuses",
        "call_status_counts",
        "request_retries",
        "format_retries",
        "transport_retries",
        "output_limit_retries",
    ):
        if any(key in r for r in reports):
            report[key] = sum_counts([r.get(key, {}) for r in reports])
    with sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True) as db:
        actual = dict(db.execute("SELECT status,COUNT(*) FROM calls GROUP BY status"))
        assert actual == report["call_status_counts"]
        ledger = db.execute("SELECT status,COUNT(*),SUM(charge) FROM calls GROUP BY status").fetchall()
    received = {"completed", "received", "invalid"}
    report["cost_accounting"] = {
        "received_response_estimate_usd": sum(cost for status, _, cost in ledger if status in received),
        "unresolved_reservations_usd": sum(cost for status, _, cost in ledger if status not in received),
        "unresolved_attempts": sum(n for status, n, _ in ledger if status not in received),
    }
    freeze(output / "source-contexts.json", contexts)
    freeze(output / "manifest.json", manifest)
    freeze(output / "report.json", report)
    return output


def publish(output=OUTPUT):
    from scripts import report_experiment_dashboard as dashboard
    from scripts.report_roster_triadic import publish as publish_triadic
    from scripts.report_stateful_dyadic import publish as publish_dyadic

    output = Path(output).resolve()
    specs = execution_specs(output)
    for path, expected in read(output / "preserved-source-hashes.json").items():
        if sha(path) != expected:
            raise ValueError(f"Preserved source changed: {path}")
    combined = {}
    for roster in ROSTERS:
        for setting in ("dyadic", "triadic", "quality"):
            spec = next(s for s in specs if s["roster"] == roster and s["setting"] == setting)
            combined[roster, setting] = merge(
                [spec["source_20"], spec["output_30"]],
                output / "combined" / roster / setting,
                debate_parent=combined[roster, "dyadic"] if setting == "quality" else None,
            )
        dyad = ROOT / "docs" / f"turn-tone-dyadic-{roster.replace('_', '-')}-50-public-history.html"
        triad = ROOT / "docs" / f"turn-tone-triadic-{roster.replace('_', '-')}-50-public-history.html"
        publish_dyadic(combined[roster, "dyadic"], dyad, quality_source=combined[roster, "quality"])
        publish_triadic(combined[roster, "triadic"], triad, dyadic_report=dyad)
        for members, path in ((2, dyad), (3, triad)):
            soup = BeautifulSoup(path.read_text(), "html.parser")
            note = soup.new_tag("p", attrs={"class": "batch-provenance"})
            note.string = "50 distinct questions: the unchanged original 20 plus 30 additional approved questions. The same added questions, routing and tone plans are used across rosters. No earlier generation or failure was replaced. Results are recomputed from individual records, not averaged across batch percentages."
            soup.h1.insert_after(note)
            # Keep the original 20-pair explanation only in the archived report.
            text = str(soup).replace(
                "A 20-pair row has 40 decisions.", "Each complete pair contributes two decisions, one per answer order."
            )
            if members == 3:
                diagnostic_count = read(combined[roster, "triadic"] / "manifest.json")["design"]["E_calibration_cases"]
                text = text.replace(
                    "Twelve questions were selected before generation.",
                    f"{diagnostic_count} questions were selected before generation across the two frozen batches.",
                )
            path.write_text(text)
            data = read(path.with_suffix(".json"))
            data["batches"] = read(combined[roster, "dyadic" if members == 2 else "triadic"] / "manifest.json")[
                "batches"
            ]
            atomic_json(path.with_suffix(".json"), data)
            dashboard.REPORTS[roster, members] = (path.name, data["protocol_version"])
    old_dashboard = ROOT / "docs/committee-experiment-dashboard.html"
    reference = ROOT / "docs/committee-experiment-dashboard-20-question-reference.html"
    if not reference.exists():
        shutil.copyfile(old_dashboard, reference)
    # Preserve the independently journaled judge addition when rebuilding reports.
    from scripts.grok_quality_check import OUTPUT as grok_output

    if (grok_output / "report.json").exists() and read(grok_output / "report.json")["status"] == "completed":
        from scripts.report_grok_quality import attach

        attach(judge_output=grok_output)
    dashboard.publish(output=old_dashboard)
    atomic_json(
        output / "published-reports.json",
        {
            "dashboard": str(old_dashboard),
            "reference_dashboard": str(reference),
            "reports": {f"{r}/{m}": name for (r, m), (name, _) in dashboard.REPORTS.items()},
        },
    )
    return old_dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(publish(args.output))
