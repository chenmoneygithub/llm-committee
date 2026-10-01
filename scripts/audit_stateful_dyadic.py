"""Reconstruct frozen stateful requests and independently check native probability tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot import prompts, stateful_study
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.stateful_study import (
    PROTOCOL_PROMPTS,
    PUBLIC_HISTORY_VERSION,
    StatefulGraph,
    StatefulMockProvider,
)
from llm_committee.pivot.study import atomic_json, native_token_count


def check_native(request, response, value, tokenizer, renderer):
    from llm_committee.pivot.tinker_provider import message_content

    raw, saved = response["raw"], value["_readout"]
    tokens, index = raw["sampled_token_ids"], saved["sampled_output_index"]
    visible = response["text"]
    labels = request["candidate_labels"]
    assert visible in labels
    assert [i for i, t in enumerate(tokens) if tokenizer.decode([t]) == visible] == [index]
    assert len(tokens) == len(raw["sampled_logprobs"]) == len(raw["topk_logprobs"])
    message, termination = renderer.parse_response(tokens)
    text, reasoning = message_content(message)
    assert text == visible and not reasoning.strip() and termination.is_clean
    assert response["reasoning_tokens"] == 0 and raw["stop_reason"] == "stop"
    messages = [
        {"role": "system" if m["role"] == "developer" else m["role"], "content": m["text"]} for m in request["messages"]
    ]
    kwargs = {"effort": 0.0} if request["model"] == "thinkingmachines/Inkling" else {}
    prompt_ids = renderer.build_generation_prompt(messages, **kwargs).to_ints()
    assert prompt_ids == raw["prompt_token_ids"]
    prefix = prompt_ids + tokens[:index]
    assert prefix == saved["prefix_token_ids"] and digest(prefix) == saved["prefix_sha256"]
    ids = {k: tokenizer.encode(k, add_special_tokens=False) for k in labels}
    assert all(len(v) == 1 and tokenizer.decode(v) == k for k, v in ids.items())
    assert {k: v[0] for k, v in ids.items()} == saved["candidate_ids"]
    assert tokens[index] == saved["sampled_token_id"] == ids[visible][0]
    top = dict(raw["topk_logprobs"][index])
    assert {k: top[v[0]] for k, v in ids.items() if v[0] in top} == saved["candidate_logprobs"]
    assert raw["sampled_logprobs"][index] == saved["sampled_token_logprob"]
    assert saved == response["readout"] == raw["probability_readout"]
    for k, expected in (("sampling_temperature", 1.0), ("top_p", 1.0), ("top_k", -1)):
        assert saved[k] == raw[k] == expected
    assert saved["read_temperature"] == 1 and saved["source"] == "sample_topk"


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contexts = json.loads((root / "source-contexts.json").read_text())
    assert manifest["protocol_version"] in PROTOCOL_PROMPTS
    public_history = manifest["protocol_version"] == PUBLIC_HISTORY_VERSION
    assert manifest["kind"] in ("paid_stateful_dyadic_pilot", "offline_mock")
    mock = manifest["kind"] == "offline_mock"
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(stateful_study.__file__).parent.glob("*.py"))
    }
    assert digest(hashes) == manifest["implementation_sha256"]
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    assert all(set(c) == {"question"} for c in contexts.values())
    assert (
        hashlib.sha256(Path(manifest["source"]["plan_file"]).read_bytes()).hexdigest()
        == manifest["source"]["plan_sha256"]
    )
    saved, purposes, native = {}, Counter(), Counter()
    renderers = {}
    with sqlite3.connect(f"file:{root}/requests.sqlite3?mode=ro", uri=True) as db:
        for rq, rs, parsed in db.execute("SELECT request,response,parsed FROM calls WHERE status='completed'"):
            request, response, value = map(json.loads, (rq, rs, parsed))
            if public_history and request["purpose"] in ("debate", "d_choice", "d_text"):
                assert "your_current_position" not in canonical(request["messages"])
                if request["purpose"] != "d_text":
                    assert "position_to_evaluate" not in canonical(request["messages"])
            assert request["key"] not in saved
            saved[request["key"]] = (rq, value)
            purposes[request["purpose"]] += 1
            if request["purpose"] in ("initial", "debate"):
                fields = {"choice", "position"}
                if request["purpose"] == "debate":
                    fields |= {"reply", "agreement"}
                    assert value["reply"].strip()
                assert set(value) == fields and value["position"].strip()
                assert request["effort"] == "medium"
                assert value["choice"] in request["schema"]["properties"]["choice"]["enum"]
            if request["candidate_labels"]:
                assert request["purpose"] in ("d_choice", "d_text")
                assert request["effort"] == "none" and response["reasoning_tokens"] == 0
                assert all(prompts.TONE_TEXT[t] not in canonical(request["messages"]) for t in ("friendly", "hostile"))
                if not mock:
                    from llm_committee.pivot.tinker_provider import make_renderer

                    model = request["model"]
                    if model not in renderers:
                        renderers[model] = make_renderer(model, "none")
                    check_native(request, response, value, *renderers[model])
                native[request["model"], request["purpose"]] += 1
    count = StatefulMockProvider().token_count if mock else native_token_count()
    verified = tasks = pairs = 0
    for plan in manifest["plans"]:
        path = root / "questions" / f"{plan['question_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            continue
        graph = StatefulGraph(contexts[plan["question_id"]], plan, manifest)
        while not graph.complete:
            task = graph.ready()
            assert task is not None
            request = task.build(graph.values, count)
            original, value = saved[task.key]
            assert canonical(request.document()) == original, task.key
            graph.accept(task, request, value)
            tasks += 1
        assert canonical(graph.report(count)) == canonical(record)
        for pair in ("AB", "CA", "BC"):
            a, b = [
                json.loads(saved[graph.key(arm, f"turn_level/debate/{pair}-3")][0]) for arm in ("original", "alternate")
            ]
            assert a["messages"][:-1] == b["messages"][:-1]
            assert a["messages"][-1] != b["messages"][-1]
        for key, (rq, _) in saved.items():
            if key.startswith(plan["question_id"] + "/") and "/Dtext/" in key and key.endswith("/argument"):
                a = json.loads(rq)["messages"]
                b = json.loads(saved[key.removesuffix("argument") + "control"][0])["messages"]
                assert len(a) == len(b)
                assert [i for i in range(len(a)) if a[i] != b[i]] == [len(a) - 2]
                if public_history:
                    target = json.loads(a[-3]["text"])["position_to_evaluate"]
                    assert set(target) == {"source", "text"}
                    arm = key.split("/mixed_family/", 1)[1].split("/", 1)[0]
                    nid = key.split("/Dtext/turn_level/", 1)[1].split("/", 1)[0]
                    d1 = json.loads(saved[graph.key(arm, f"Dchoice/turn_level/{nid}")][0])["messages"]
                    assert d1[:-1] == a[:-3] + [a[-2]]
                    prior = graph.route.previous_own(nid)
                    reading = f"turn_level/{prior.id}" if prior else f"initial/{graph.route.get(nid).receiver}"
                    assert target["text"] == graph.position(arm, reading)["position"]
                pairs += 1
        verified += 1
    if verified == len(manifest["plans"]):
        assert dict(purposes) == manifest["planned_counts"]["by_purpose"]
        assert tasks == len(saved) == manifest["planned_counts"]["logical_calls"]
    return {
        "protocol_version": manifest["protocol_version"],
        "public_history_only_debate_and_D1": public_history,
        "source": str(root),
        "completed_requests": len(saved),
        "purposes": dict(purposes),
        "questions_reconstructed_exactly": verified,
        "questions_not_complete": len(manifest["plans"]) - verified,
        "requests_reconstructed_exactly": tasks,
        "native_token_checks": [{"model": m, "purpose": p, "checked": n} for (m, p), n in sorted(native.items())],
        "argument_control_pairs_differ_only_in_incoming": pairs,
        "frozen_code_and_plan_verified": True,
        "old_outputs_reused": False,
        "native_checks_are_synthetic": mock,
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
