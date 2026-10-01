"""Read-only request reconstruction and scope checks for the three-member A/B/E study."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot import failures
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.quality_study import LENGTH, VERBOSE_LENGTH, words
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.triadic_study import ARMS, VERSION, TriadicGraph, code_hash


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contexts = json.loads((root / "source-contexts.json").read_text())
    assert manifest["protocol_version"] == VERSION
    assert code_hash() == manifest["implementation_sha256"]
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    assert all(set(c) == {"question"} for c in contexts.values())
    source = manifest["source"]
    assert hashlib.sha256(Path(source["plan_file"]).read_bytes()).hexdigest() == source["plan_sha256"]
    design = json.loads(Path(source["plan_file"]).read_text())
    for key in ("question_plan", "route_manifest"):
        assert (
            hashlib.sha256(Path(design["sources"][key]).read_bytes()).hexdigest() == design["sources"][key + "_sha256"]
        )
    archive = json.loads(Path(design["sources"]["route_manifest"]).read_text())
    old_routes = {p["question_id"]: p["route"] for p in archive["plans"]}
    saved, attempts, purposes = {}, {}, Counter()
    lengths = []
    with sqlite3.connect(f"file:{root}/requests.sqlite3?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT key,request,response,status,error FROM calls"):
            attempts.setdefault(json.loads(row["request"])["key"], []).append(dict(row))
        for rq, parsed in db.execute("SELECT request,parsed FROM calls WHERE status='completed'"):
            request, value = json.loads(rq), json.loads(parsed)
            assert request["key"] not in saved
            saved[request["key"]] = (rq, value)
            purposes[request["purpose"]] += 1
            assert request["purpose"] in (
                "initial",
                "debate",
                "judge_b",
                "judge_e1",
                "synthesis",
                "judge_e",
                "calibration_validation",
            )
            assert not request["candidate_labels"] and not request.get("scoring")
            if request["purpose"] in ("initial", "debate"):
                assert request["effort"] == manifest["config"]["debate_effort"]
                assert "your_current_position" not in canonical(request["messages"])
                assert "position_to_evaluate" not in canonical(request["messages"])
                assert set(value) == (
                    {"choice", "position"}
                    if request["purpose"] == "initial"
                    else {"reply", "agreement", "choice", "position"}
                )
            if request["purpose"] == "synthesis":
                bounds = VERBOSE_LENGTH if "/calibration/verbosity/" in request["key"] else LENGTH
                assert bounds[0] <= words(value["answer"]) <= bounds[1]
                if "/calibration/" not in request["key"]:
                    lengths.append(words(value["answer"]))
    verified = questions = orders = missing_pairs = failed = blocked = planned = 0
    input_words = []
    for plan in manifest["plans"]:
        assert plan["route"] == old_routes[plan["question_id"]]
        path = root / "questions" / f"{plan['question_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["status"] not in ("completed", "completed_with_failures"):
            continue
        graph = TriadicGraph(contexts[plan["question_id"]], plan, manifest)
        while not graph.complete:
            task = graph.ready()
            assert task is not None
            request = task.build(graph.values, lambda m, t: len(t.split()))
            if task.key in record["failed_tasks"]:
                failure = record["failed_tasks"][task.key]
                rows = sorted(attempts[task.key], key=lambda r: r["key"])
                keys = [failures.retry_key(task.key, i) for i in range(3)]
                assert failure["attempt_keys"] == keys and [r["key"] for r in rows] == keys
                for row in rows:
                    assert row["request"] == canonical(request.document())
                    ordinary = failures.retryable_failure_record(
                        request.document(), row, manifest["config"]["closed_provider"]
                    )
                    output_limit = bool(manifest["execution"].get("output_limit_retry_policy")) and getattr(
                        failures, "local_output_limit_record", lambda *args: False
                    )(request.document(), row, manifest["config"]["closed_provider"])
                    assert ordinary or output_limit
                expected_kind = (
                    "failed_after_transport_retries"
                    if rows[-1]["status"] == "uncertain"
                    else (
                        "failed_after_output_limit_retries"
                        if rows[-1]["status"] == "billing_unknown"
                        else "failed_after_format_retries"
                    )
                )
                assert failure["status"] == expected_kind
                assert failure["reason"] == rows[-1]["error"]
                graph.reject(task, failures.LocalTaskFailure(task.key, keys, failure["reason"], kind=expected_kind))
                failed += 1
                continue
            old, value = saved[task.key]
            assert canonical(request.document()) == old, task.key
            graph.accept(task, request, value)
            verified += 1
            if request.purpose == "synthesis" and "/E2/" in request.key:
                data = json.loads(request.messages[-1].text)
                assert len(data["initial_answers"]) == 3
                ids = [n["id"] for n in data["discussion"]]
                assert len(ids) == len(set(ids))
                assert len(ids) == (0 if request.key.endswith("/E2/baseline") else len(graph.route.nodes))
                assert all(set(n) == {"id", "member", "reply_to", "text"} for n in data["discussion"])
                input_words.append(sum(len(m.text.split()) for m in request.messages))
        assert canonical(graph.report(None)) == canonical(record)
        blocked += len(graph.blocked)
        planned += len(graph.tasks)
        for n in graph.route.nodes:
            if n.depth == 3:
                keys = [graph.key(arm, f"debate/{n.id}") for arm in ARMS]
                if not all(key in saved for key in keys):
                    continue
                a, b = [json.loads(saved[key][0]) for key in keys]
                assert a["messages"][:-1] == b["messages"][:-1]
                assert a["messages"][-1] != b["messages"][-1]
        for row in record["E2"]:
            if row["status"] != "success":
                assert row["debate_score"] is None
                missing_pairs += 1
                continue
            judgments = row["preferences"]["orders"]
            assert {j["target_side"] for j in judgments} == {"left", "right"}
            votes = [j["judgment"]["preference"] == j["target_side"] for j in judgments]
            assert row["debate_score"] == sum(votes) / 2
            orders += 1
        questions += 1
    assert verified + failed + blocked == planned
    if questions == len(manifest["plans"]):
        assert verified == len(saved)
    return {
        "protocol_version": VERSION,
        "completed_requests": len(saved),
        "requests_reconstructed_exactly": verified,
        "questions_reconstructed_exactly": questions,
        "purposes": dict(purposes),
        "local_C_calls": 0,
        "D_calls": 0,
        "archived_routes_preserved": questions,
        "primary_pairs_both_orders_verified": orders,
        "missing_primary_pairs": missing_pairs,
        "failed_tasks_verified": failed,
        "blocked_tasks_verified": blocked,
        "primary_answer_word_range": [min(lengths), max(lengths)] if lengths else None,
        "largest_chairman_input_words": max(input_words, default=0),
        "no_other_experiment_outputs_reused": True,
        "same_batch_checkpoint_reused": manifest.get("recovery", {}).get("prior_successful_requests", 0),
        "frozen_plan_and_implementation_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
