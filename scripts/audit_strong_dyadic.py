"""Read-only reconstruction of the fresh strong-label dyadic run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot import prompts
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.strong_agreement import AGREEMENT, AGREEMENT_RUBRIC_TEXT
from llm_committee.pivot.strong_study import StrongGraph, StrongMockProvider, implementation_hash
from llm_committee.pivot.study import native_token_count


def audit(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contexts = json.loads((root / "source-contexts.json").read_text())
    assert manifest["kind"] in ("paid_strong_dyadic_pilot", "offline_mock")
    assert implementation_hash() == manifest["implementation_sha256"]
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    assert all(set(c) == {"question"} for c in contexts.values()), "Never import old generated outputs"
    assert (
        hashlib.sha256(Path(manifest["source"]["plan_file"]).read_bytes()).hexdigest()
        == manifest["source"]["plan_sha256"]
    )
    saved, purposes = {}, Counter()
    direct = probability = 0
    with sqlite3.connect(f"file:{root}/requests.sqlite3?mode=ro", uri=True) as db:
        for raw_request, raw_response, raw_parsed in db.execute(
            "SELECT request, response, parsed FROM calls WHERE status='completed'"
        ):
            req, response, value = map(json.loads, (raw_request, raw_response, raw_parsed))
            assert req["key"] not in saved
            saved[req["key"]] = raw_request, value
            purposes[req["purpose"]] += 1
            if req["purpose"] in ("debate", "judge_b"):
                assert AGREEMENT_RUBRIC_TEXT in req["messages"][0]["text"]
                assert "fully_agree" not in canonical(req["schema"])
                assert "fully_disagree" not in canonical(req["schema"])
                label = value.get("agreement") if req["purpose"] == "debate" else value.get("label")
                assert label in (*AGREEMENT, None, "no_position", "unjudgeable")
            if req["purpose"] == "debate":
                assert req["effort"] != "none"
            if req["purpose"] in ("position", "d_text"):
                assert req["messages"][0]["text"] == prompts.BASE + "\n"
                assert req["effort"] == "none"
            if req["purpose"] in ("debate", "position", "d_text"):
                for message in req["messages"][1:]:
                    assert all(prompts.TONE_TEXT[t] not in message["text"] for t in ("friendly", "hostile"))
            if req["effort"] == "none":
                direct += 1
                assert response["reasoning_tokens"] == 0
            if req["candidate_labels"]:
                probability += 1
                assert value["_readout"]["read_temperature"] == 1
                if manifest["kind"] != "offline_mock":
                    assert value["_readout"]["source"] == "sample_topk"
    count = StrongMockProvider().token_count if manifest["kind"] == "offline_mock" else native_token_count()
    verified = matched_t3 = verified_tasks = shared_debate = shared_dtext = 0
    for plan in manifest["plans"]:
        path = root / "questions" / f"{plan['question_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            continue
        graph = StrongGraph(contexts[plan["question_id"]], plan, manifest)
        assert not graph.values
        while not graph.complete:
            task = graph.ready()
            assert task is not None
            request = task.build(graph.values, count)
            original_request, value = saved[task.key]
            assert canonical(request.document()) == original_request, task.key
            graph.accept(task, request, value)
            verified_tasks += 1
        assert canonical(graph.report(count)) == canonical(record)
        for pair in ("AB", "CA", "BC"):
            keys = [graph.key(arm, f"turn_level/debate/{pair}-3") for arm in ("original", "alternate")]
            a, b = [json.loads(saved[key][0]) for key in keys]
            assert a["messages"][1:] == b["messages"][1:]
            assert a["messages"][0] != b["messages"][0]
            assert {**a, "key": b["key"], "messages": b["messages"]} == b
            matched_t3 += 1
            for t in (1, 2):
                assert graph.key("original", f"turn_level/debate/{pair}-{t}") == graph.key(
                    "alternate", f"turn_level/debate/{pair}-{t}"
                )
                shared_debate += 1
        for event in plan["arms"]["original"]["events"]:
            if event["member"] in (1, 2) and event["T"] <= 3:
                for treatment in ("argument", "control"):
                    suffix = f"Dtext/{event['id']}/{treatment}"
                    assert graph.key("original", suffix) == graph.key("alternate", suffix)
                    shared_dtext += 1
        verified += 1
    if verified == len(manifest["plans"]):
        assert dict(purposes) == manifest["planned_counts"]["by_purpose"]
        assert verified_tasks == len(saved) == manifest["planned_counts"]["logical_calls"]
    return {
        "completed_requests": len(saved),
        "purposes": dict(purposes),
        "fresh_initial_answers": purposes["initial"],
        "imported_old_outputs": 0,
        "direct_reads": direct,
        "direct_reads_with_reasoning": 0,
        "original_topk_probability_reads": probability,
        "question_reports_reconstructed_exactly": verified,
        "questions_not_complete": len(manifest["plans"]) - verified,
        "new_request_contexts_verified_exactly": verified_tasks,
        "T3_requests_differ_only_in_current_tone_instruction": matched_t3,
        "shared_prefix_replies_counted_once": shared_debate,
        "shared_pre_reply_D_requests_counted_once": shared_dtext,
        "current_tone_only_and_neutral_probes_verified": True,
        "frozen_implementation_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source), indent=2))
