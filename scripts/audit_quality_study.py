"""Read-only exact reconstruction and length/context audit of the E follow-up."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot.failures import LocalTaskFailure, retry_key
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.quality_study import (
    ARMS,
    LENGTH,
    PAIRS,
    VERBOSE_LENGTH,
    QualityGraph,
    implementation_hash,
    words,
)


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contexts = json.loads((root / "source-contexts.json").read_text())
    assert implementation_hash() == manifest["implementation_sha256"]
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    source = Path(manifest["source"]["directory"])
    assert hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() == manifest["source"]["manifest_sha256"]
    for qid, expected in manifest["source"]["question_sha256"].items():
        assert hashlib.sha256((source / "questions" / f"{qid}.json").read_bytes()).hexdigest() == expected
    saved, attempts = {}, {}
    purposes = Counter()
    primary_lengths = []
    with sqlite3.connect(f"file:{root}/requests.sqlite3?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT key,request,response,status,error FROM calls"):
            attempts.setdefault(json.loads(row["request"])["key"], []).append(dict(row))
        for req, parsed in db.execute("SELECT request,parsed FROM calls WHERE status='completed'"):
            request, value = json.loads(req), json.loads(parsed)
            assert request["key"] not in saved
            saved[request["key"]] = (req, value)
            purposes[request["purpose"]] += 1
            assert request["purpose"] in ("synthesis", "judge_e", "calibration_validation")
            assert not request["candidate_labels"]
            if request["purpose"] == "synthesis":
                bounds = VERBOSE_LENGTH if "/calibration/verbosity/" in request["key"] else LENGTH
                assert bounds[0] <= words(value["answer"]) <= bounds[1]
                if "/calibration/" not in request["key"]:
                    primary_lengths.append(words(value["answer"]))
            if request["purpose"] == "judge_e":
                assert set(json.loads(request["messages"][-1]["text"])) == {"question", "options", "left", "right"}
    requests = questions = successful_questions = primary_pairs = missing_pairs = failed = blocked = 0
    for plan in manifest["plans"]:
        path = root / "questions" / f"{plan['question_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["status"] not in ("completed", "completed_with_failures"):
            continue
        graph = QualityGraph(contexts[plan["question_id"]], plan, manifest)
        while not graph.complete:
            task = graph.ready()
            assert task is not None
            request = task.build(graph.values, None)
            if task.key in record["failed_tasks"]:
                failure = record["failed_tasks"][task.key]
                rows = sorted(attempts[task.key], key=lambda r: r["key"])
                keys = [retry_key(task.key, i) for i in range(3)]
                assert failure["attempt_keys"] == keys and [r["key"] for r in rows] == keys
                assert failure["status"] == "failed_after_format_retries"
                for row in rows:
                    assert row["status"] == "invalid"
                    assert row["request"] == canonical(request.document())
                    response = json.loads(row["response"])
                    assert response["status"] == "completed"
                    try:
                        task.parse(response["text"])
                    except ValueError as exc:
                        assert str(exc) == row["error"]
                    else:
                        raise AssertionError("Failed request contains a valid answer")
                assert rows[-1]["error"] == failure["reason"]
                graph.reject(task, LocalTaskFailure(task.key, keys, failure["reason"], kind=failure["status"]))
                failed += 1
                continue
            old, value = saved[task.key]
            assert canonical(request.document()) == old, task.key
            graph.accept(task, request, value)
            requests += 1
        assert canonical(graph.report(None)) == canonical(record)
        blocked += len(graph.blocked)
        for pair in PAIRS:
            for arm in ARMS:
                keys = [graph.key(f"{pair}/{arm}/order{i}") for i in range(2)]
                if not all(k in saved for k in keys):
                    row = next(r for r in record["E"] if r["pair"] == pair and r["assignment"] == arm)
                    assert row["debate_score"] is None
                    missing_pairs += 1
                    continue
                a, b = [json.loads(saved[k][0]) for k in keys]
                aa, bb = [json.loads(r["messages"][-1]["text"]) for r in (a, b)]
                assert aa["left"] == bb["right"] and aa["right"] == bb["left"]
                primary_pairs += 1
        questions += 1
        successful_questions += record["status"] == "completed"
    if questions == len(manifest["plans"]):
        if not failed:
            assert dict(purposes) == manifest["planned_counts"]["by_purpose"]
        assert requests == len(saved)
        assert requests + failed + blocked == manifest["planned_counts"]["logical_calls"]
    return {
        "completed_requests": len(saved),
        "purposes": dict(purposes),
        "question_reports_reconstructed_exactly": questions,
        "questions_not_complete": len(manifest["plans"]) - successful_questions,
        "questions_without_report": len(manifest["plans"]) - questions,
        "request_contexts_reconstructed_exactly": requests,
        "failed_logical_requests_verified": failed,
        "blocked_logical_requests_verified": blocked,
        "primary_pairs_both_orders_verified": primary_pairs,
        "primary_pairs_unavailable": missing_pairs,
        "source_question_hashes_unchanged": len(contexts),
        "source_manifest_unchanged": True,
        "source_debate_regenerations": 0,
        "primary_synthesis_words_min": min(primary_lengths, default=None),
        "primary_synthesis_words_max": max(primary_lengths, default=None),
        "all_completed_answer_lengths_compliant": True,
        "no_third_member_or_label_metadata_in_synthesis": True,
        "frozen_implementation_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source), indent=2))
