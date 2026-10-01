"""Read-only reconstruction and prompt audit of the turn-level-tone dyadic pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot import prompts
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.study import native_token_count
from llm_committee.pivot.turn_tone import TurnToneGraph, implementation_hash


def audit(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contexts = json.loads((root / "source-contexts.json").read_text())
    if manifest["kind"] != "paid_turn_tone_dyadic_pilot":
        raise ValueError("This audit expects the real turn-level-tone pilot")
    assert implementation_hash() == manifest["implementation_sha256"]
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    source = Path(manifest["source"]["directory"])
    assert hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() == manifest["source"]["manifest_sha256"]
    for qid, expected in manifest["source"]["question_sha256"].items():
        assert hashlib.sha256((source / "questions" / f"{qid}.json").read_bytes()).hexdigest() == expected, qid
    records = {}
    for plan in manifest["plans"]:
        path = root / "questions" / f"{plan['question_id']}.json"
        if path.exists():
            records[plan["question_id"]] = json.loads(path.read_text())
    saved, direct, probability = {}, 0, 0
    purposes = Counter()
    schedules = {p["question_id"]: p["tone_schedule"] for p in manifest["plans"]}
    with sqlite3.connect(f"file:{root}/requests.sqlite3?mode=ro", uri=True) as db:
        for storage_key, raw_request, raw_response, raw_parsed in db.execute(
            "SELECT key, request, response, parsed FROM calls WHERE status='completed'"
        ):
            request, value = json.loads(raw_request), json.loads(raw_parsed)
            key = request["key"]
            assert key not in saved, f"Multiple completed attempts for {key}"
            saved[key] = raw_request, value
            purposes[request["purpose"]] += 1
            messages = request["messages"]
            if request["purpose"] == "debate":
                tone = schedules[key.split("/")[0]][key.split("/")[-1]]
                assert messages[0]["text"] == prompts.BASE + "\n" + prompts.TONE_TEXT[tone] + "\n" + prompts.REPLY_RULE
                assert request["effort"] != "none"
            if request["purpose"] in ("position", "d_text"):
                assert messages[0]["text"] == prompts.BASE + "\n"
                assert request["effort"] == "none"
            if request["purpose"] in ("debate", "position", "d_text"):
                for message in messages[1:]:
                    for tone in ("friendly", "hostile"):
                        assert prompts.TONE_TEXT[tone] not in message["text"], key
            if request["effort"] == "none":
                direct += 1
                assert json.loads(raw_response)["reasoning_tokens"] == 0, storage_key
            if request["candidate_labels"]:
                probability += 1
                assert value["_readout"]["read_temperature"] == 1
                assert value["_readout"]["source"] == "sample_topk"
    count = native_token_count()
    verified, missing, verified_tasks = 0, 0, 0
    for plan in manifest["plans"]:
        record = records.get(plan["question_id"])
        if record is None or record["status"] != "completed":
            missing += 1
            continue
        graph = TurnToneGraph(contexts[plan["question_id"]], plan, manifest)
        while not graph.complete:
            task = graph.ready()
            assert task is not None, plan["question_id"]
            request = task.build(graph.values, count)
            original_request, value = saved[task.key]
            assert canonical(request.document()) == original_request, task.key
            graph.accept(task, request, value)
            verified_tasks += 1
        assert canonical(graph.report(count)) == canonical(record), plan["question_id"]
        verified += 1
    return {
        "completed_requests": len(saved),
        "purposes": dict(purposes),
        "direct_reads": direct,
        "direct_reads_with_reasoning": 0,
        "original_topk_probability_reads": probability,
        "question_reports_reconstructed_exactly": verified,
        "questions_not_complete": missing,
        "request_contexts_verified_exactly": verified_tasks,
        "source_questions_unchanged": len(manifest["source"]["question_sha256"]),
        "current_tone_only_and_neutral_probes_verified": True,
        "frozen_implementation_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source), indent=2))
