"""Merged reports retain failures and never turn a view into a runnable journal."""

import json
import sqlite3

import pytest

from llm_committee.pivot.models import canonical
from llm_committee.pivot.study import atomic_json
from scripts.report_scaled_public_history import merge, sum_counts, validate_batches


def test_counts_and_compatibility():
    assert sum_counts([{"q": 20, "purpose": {"a": 5}}, {"q": 30, "purpose": {"a": 7, "b": 1}}]) == {
        "q": 50,
        "purpose": {"a": 12, "b": 1},
    }
    a = {
        "kind": "paid_example",
        "protocol_version": "v1",
        "config": {"roster": "same_model"},
        "plans": [{"question_id": "q1"}],
    }
    b = {**a, "plans": [{"question_id": "q2"}]}
    validate_batches([a, b])
    with pytest.raises(ValueError, match="Overlapping"):
        validate_batches([a, a])
    with pytest.raises(ValueError, match="config"):
        validate_batches([a, {**b, "config": {"roster": "mixed"}}])


def batch(root, qid, failed=False):
    manifest = {
        "kind": "paid_example",
        "protocol_version": "v1",
        "config": {"roster": "same_model"},
        "implementation_sha256": "code",
        "plans": [{"question_id": qid}],
        "execution": {},
        "design": {},
        "planned_counts": {"questions": 1, "logical_calls": 1},
        "source": {},
    }
    report = {
        "kind": "paid_example",
        "protocol_version": "v1",
        "status": "completed_with_failures" if failed else "completed",
        "completed_questions": 1,
        "successful_questions": int(not failed),
        "charged_or_reserved_usd": 0.25,
        "call_status_counts": {"invalid" if failed else "completed": 1},
        "task_failures": {qid: "format"} if failed else {},
        "question_statuses": {qid: "failed" if failed else "completed"},
        "request_retries": {"additional_attempts": 0},
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "report.json", report)
    atomic_json(root / "source-contexts.json", {qid: {"question": {"id": qid}}})
    atomic_json(root / "questions" / f"{qid}.json", {"question_id": qid, "value": None if failed else 1})
    with sqlite3.connect(root / "requests.sqlite3") as db:
        db.execute(
            "CREATE TABLE calls (key TEXT PRIMARY KEY, request TEXT, request_hash TEXT, status TEXT, charge REAL, response TEXT, parsed TEXT, error TEXT)"
        )
        db.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
            (
                qid,
                canonical({"key": qid}),
                "hash",
                "invalid" if failed else "completed",
                0.25,
                "{}",
                "{}",
                "format" if failed else None,
            ),
        )


def test_merge_preserves_records_failures_and_is_idempotent(tmp_path):
    a, b, output = (tmp_path / n for n in ("a", "b", "view"))
    batch(a, "q1")
    batch(b, "q2", failed=True)
    original = (b / "questions/q2.json").read_bytes()
    merge([a, b], output)
    m = json.loads((output / "manifest.json").read_text())
    r = json.loads((output / "report.json").read_text())
    assert m["analysis_only"] and not m["execution"]["dispatch_enabled"]
    assert m["planned_counts"]["questions"] == r["completed_questions"] == 2
    assert r["status"] == "completed_with_failures" and r["task_failures"] == {"q2": "format"}
    assert r["charged_or_reserved_usd"] == 0.5
    assert (output / "questions/q2.json").read_bytes() == original
    assert merge([a, b], output) == output
    assert (b / "questions/q2.json").read_bytes() == original
    with pytest.raises(ValueError, match="separate"):
        merge([a, b], a / "view")
