"""Matched continuations: reuse exact T1/T2, change private tone at T3 and T4."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from . import turn_tone
from .models import canonical, digest
from .planning import rng_for
from .turn_tone import QUESTION_COUNT, STREAM, TurnToneGraph

VERSION = "turn-tone-fork-2026-09-26-v1"
SEED = 20260928


def alternate_plans(original, seed=SEED):
    """Randomized matching preserves per-node tone counts without using any outcomes."""
    plans = sorted(copy.deepcopy(original), key=lambda p: p["question_id"])
    if len(plans) < 3 or len({p["question_id"] for p in plans}) != len(plans):
        raise ValueError("Need at least three distinct questions for paired tones")
    rng = rng_for(seed, VERSION, "alternate-tones")
    for plan in plans:
        plan["original_tone_schedule"] = dict(plan["tone_schedule"])
    for node in sorted(plans[0]["tone_schedule"]):
        if int(node.rsplit("-", 1)[1]) <= 2:
            continue
        old = [p["tone_schedule"][node] for p in plans]
        slots = old.copy()
        rng.shuffle(slots)
        choices = [[j for j, tone in enumerate(slots) if tone != old[i]] for i in range(len(plans))]
        for options in choices:
            rng.shuffle(options)
        owners = {}

        def assign(i, seen, choices=choices, owners=owners):
            for slot in choices[i]:
                if slot in seen:
                    continue
                seen.add(slot)
                if slot not in owners or assign(owners[slot], seen):
                    owners[slot] = i
                    return True
            return False

        order = list(range(len(plans)))
        rng.shuffle(order)
        if not all(assign(i, set()) for i in order):
            raise ValueError("Cannot balance alternate tones without an unchanged turn")
        for slot, i in owners.items():
            plans[i]["tone_schedule"][node] = slots[slot]
        assert Counter(p["tone_schedule"][node] for p in plans) == Counter(old)
    for plan in plans:
        for event in plan["events"]:
            nid = event["node_id"]
            parent = f"{event['pair']}-{event['T'] - 1}"
            event["current_tone"] = plan["tone_schedule"][nid]
            event["previous_tone"] = plan["tone_schedule"][parent]
    return plans


def reusable_suffixes(plan):
    """D-text at T3 is BEFORE the changed reply and has the same neutral input."""
    keys = {f"{STREAM}/debate/{pair}-{t}" for pair in ("AB", "CA", "BC") for t in (1, 2)}
    readings = {r["id"] for r in plan["readings"] if r["T"] <= 2}
    keys.update(f"C/{rid}" for rid in readings)
    keys.update(f"B/{e['id']}" for e in plan["events"] if e["T"] == 2)
    keys.update(f"Cjudge/{p['id']}" for p in plan["text_pairs"] if p["before"] in readings and p["after"] in readings)
    keys.update(
        f"Dtext/{e['id']}/{arm}"
        for e in plan["events"]
        if e["T"] <= 3 and e["member"] in (1, 2)
        for arm in ("argument", "control")
    )
    return keys


def implementation_hash():
    return digest(
        {
            "base_execution": turn_tone.implementation_hash(),
            **{
                n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                for n in ("turn_tone_fork.py", "turn_tone_fork_run.py")
            },
        }
    )


def prepare(source, *, mock=False, seed=SEED):
    source = Path(source).resolve()
    original = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if original["protocol_version"] != turn_tone.VERSION or report["status"] != "completed":
        raise ValueError("Need the completed turn-level-tone pilot")
    if original["kind"] != "paid_turn_tone_dyadic_pilot" and not (mock and original["kind"] == "offline_mock"):
        raise ValueError("Paid continuation requires a real source run")
    if original["implementation_sha256"] != turn_tone.implementation_hash():
        raise ValueError("Source generation code changed")
    initial_contexts = json.loads((source / "source-contexts.json").read_text())
    if digest(initial_contexts) != original["source"]["contexts_sha256"]:
        raise ValueError("Source initial contexts changed")
    saved = {}
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        for key, request, parsed in db.execute("SELECT key, request, parsed FROM calls WHERE status='completed'"):
            request = json.loads(request)
            if request["key"] in saved:
                raise ValueError("Multiple successful source attempts")
            saved[request["key"]] = {"storage_key": key, "request": request, "parsed": json.loads(parsed)}
    plans = alternate_plans(original["plans"], seed)
    contexts, hashes = {}, {}
    for plan in plans:
        qid = plan["question_id"]
        raw = (source / "questions" / f"{qid}.json").read_bytes()
        record = json.loads(raw)
        if record["status"] != "completed" or record["tone_schedule"] != plan["original_tone_schedule"]:
            raise ValueError("Incomplete or changed source question")
        hashes[qid] = hashlib.sha256(raw).hexdigest()
        contexts[qid] = {
            **initial_contexts[qid],
            "reused_requests": {
                suffix: saved[f"{qid}/{turn_tone.VERSION}/{suffix}"] for suffix in sorted(reusable_suffixes(plan))
            },
        }
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_turn_tone_fork",
        "protocol_version": VERSION,
        "implementation_sha256": implementation_hash(),
        "base_prompt_version": original["base_prompt_version"],
        "config": original["config"],
        "agreement_rubric": original["agreement_rubric"],
        "probability_readout": original["probability_readout"],
        "question_selection": original["question_selection"],
        "source": {
            "directory": str(source),
            "kind": original["kind"],
            "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
            "question_sha256": hashes,
            "contexts_sha256": digest(contexts),
            "reused_requests_sha256": digest({q: c["reused_requests"] for q, c in contexts.items()}),
        },
        "execution": original["execution"],
        "sampling": {**original["sampling"], "new_events": "Original sampled T3/T4 only; shared T2 counted once"},
        "design": {
            **original["design"],
            "seed": seed,
            "shared_prefix": "Exact archived T1/T2 text and measurements, not regenerated",
            "alternate_assignment": "Different tone at each T3 and T4; same per-node tone counts; assigned before outputs",
            "comparison": "T3: identical history, different private instruction; T4: changed instruction AND divergent history",
            "D_text_T3": "Same neutral pre-reply probe and peer text; reuse, not an independent contrast",
            "independent_repetitions": 0,
            "comparability": "Same questions, original options, initial positions, routes, prompts and neutral probes",
        },
        "plans": plans,
    }
    new, reused = Counter(), Counter()
    for plan in plans:
        graph = ForkGraph(contexts[plan["question_id"]], plan, manifest)
        for key, task in graph.tasks.items():
            (reused if key in graph.reused_keys else new)[task.purpose] += 1
    manifest["planned_counts"] = {
        "questions": QUESTION_COUNT,
        "paired_paths": 60,
        "new_trajectories": 60,
        "formal_replies": new["debate"],
        "shared_prefix_replies": reused["debate"],
        "total_unique_replies_with_source": original["planned_counts"]["formal_replies"] + new["debate"],
        "sampled_new_events": sum(e["T"] >= 3 for p in plans for e in p["events"]),
        "logical_calls": sum(new.values()),
        "by_purpose": dict(new),
        "reused_requests": sum(reused.values()),
        "reused_by_purpose": dict(reused),
    }
    return json.loads(canonical(manifest)), json.loads(canonical(contexts))


class ForkGraph(TurnToneGraph):
    def __init__(self, context, plan, manifest):
        super().__init__(context, plan, manifest)
        expected = reusable_suffixes(plan)
        if set(context["reused_requests"]) != expected:
            raise ValueError("Reused requests differ from the permitted prefix and neutral probes")
        self.reused_keys = {self.key(suffix) for suffix in expected}
        for suffix, saved in context["reused_requests"].items():
            self.values[self.key(suffix)] = copy.deepcopy(saved["parsed"])
            self.submitted.add(self.key(suffix))

    def key(self, suffix):
        return f"{self.question.id}/{VERSION}/{suffix}"

    def verify_reuse(self, count):
        for suffix, saved in self.context["reused_requests"].items():
            task = self.tasks[self.key(suffix)]
            request = task.build(self.values, count)
            original = {**saved["request"], "key": request.key}
            if canonical(request.document()) != canonical(original):
                raise ValueError(f"Attempted reuse of a changed request: {suffix}")
            self.accept(task, request, self.values[task.key])

    def report(self, count):
        record = super().report(count)
        return {
            **record,
            "assignment": "alternate",
            "original_tone_schedule": self.plan["original_tone_schedule"],
            "reused_request_keys": sorted(self.reused_keys),
            "new_request_keys": sorted(self.tasks.keys() - self.reused_keys),
            "events": [
                {**e, "shared_prefix_event": e["T"] <= 2, "D_text_reused": bool(e["D_text"]) and e["T"] <= 3}
                for e in record["events"]
            ],
        }
