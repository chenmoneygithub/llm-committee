"""Twenty-question dyadic pilot with prescheduled turn-level tones and neutral probes.

The legacy event field `tone` is a storage-stream identifier, always `turn_level`.
Actual assignments are explicit `current_tone` and `previous_tone` fields; neither
is transmitted as historical metadata. Archived fixed-tone executors are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from . import prompts
from .dyadic import DEPTH, PAIRS, DyadicGraph, make_route
from .dyadic import prepare as prepare_fixed_source
from .models import OPEN_MODELS, TONES, Request, canonical, digest
from .planning import rng_for
from .probabilities import RATING_LABELS
from .taskgraph import Task

VERSION = "turn-tone-dyadic-2026-09-26-v1"
STREAM = "turn_level"
QUESTION_COUNT = 20
SEED = 20260927


def plan_questions(qids, seed=SEED):
    """Balance assignments within pair × T; sample 8/9 eligible nodes per question."""
    qids = sorted(qids)
    count = len(qids)
    if count < 3 or len(set(qids)) != count:
        raise ValueError("Need at least three distinct presampled questions")
    route = make_route()
    rng = rng_for(seed, VERSION, "turn-tones")
    schedules = {qid: {} for qid in qids}
    for index, node in enumerate(route.nodes):
        tones = [TONES[(i + index) % 3] for i in range(count)]
        rng.shuffle(tones)
        for qid, tone in zip(qids, tones, strict=True):
            schedules[qid][node.id] = tone
    assert Counter(t for s in schedules.values() for t in s.values()) == dict.fromkeys(TONES, count * 4)
    sample_rng = rng_for(seed, VERSION, "measurement-sample")
    eligible = [n.id for n in route.nodes if n.depth >= 2]
    omitted = (eligible * ((count + len(eligible) - 1) // len(eligible)))[:count]
    sample_rng.shuffle(omitted)
    plans = []
    for qid, omitted_node in zip(qids, omitted, strict=True):
        readings, pairs, events = {}, {}, []

        def read(member, node, readings=readings):
            rid = f"{STREAM}/{node.id}" if node else f"initial/{member}"
            readings[rid] = {
                "id": rid,
                "tone": STREAM if node else "initial",
                "probe_tone": "neutral",
                "member": member,
                "node_id": node.id if node else None,
                "T": node.depth if node else 0,
                "participation_index": route.participation(node.id) if node else 0,
            }
            return rid

        def compare(before, after, kind, pairs=pairs):
            pid = f"{before}->{after}"
            item = pairs.setdefault(pid, {"id": pid, "before": before, "after": after, "kinds": []})
            if kind not in item["kinds"]:
                item["kinds"].append(kind)
            return pid

        for node in route.nodes:
            if node.depth == 1 or node.id == omitted_node:
                continue
            previous = read(node.receiver, route.previous_own(node.id))
            initial = read(node.receiver, None)
            current = read(node.receiver, node)
            events.append(
                {
                    "id": f"{STREAM}/{node.id}",
                    "pair": node.id.split("-")[0],
                    "tone": STREAM,
                    "current_tone": schedules[qid][node.id],
                    "previous_tone": schedules[qid][node.parent],
                    "probe_tone": "neutral",
                    "node_id": node.id,
                    "member": node.receiver,
                    "peer": node.sender,
                    "T": node.depth,
                    "participation_index": route.participation(node.id),
                    "previous_reading": previous,
                    "initial_reading": initial,
                    "current_reading": current,
                    "adjacent_pair": compare(previous, current, "adjacent"),
                    "final_pair": compare(initial, current, "initial_to_endpoint") if node.depth >= 3 else None,
                }
            )
        plans.append(
            {
                "question_id": qid,
                "tone_schedule": schedules[qid],
                "omitted_measurement_node": omitted_node,
                "events": events,
                "readings": list(readings.values()),
                "text_pairs": list(pairs.values()),
            }
        )
    return plans


def implementation_hash():
    names = (
        "turn_tone.py",
        "turn_tone_run.py",
        "dyadic.py",
        "prompts.py",
        "models.py",
        "agreement.py",
        "planning.py",
        "taskgraph.py",
        "scheduling.py",
        "storage.py",
        "failures.py",
        "probabilities.py",
        "providers.py",
        "tinker_provider.py",
        "databricks_provider.py",
    )
    return digest({n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def prepare(source, *, mock=False, seed=SEED):
    original, all_contexts = prepare_fixed_source(source, mock=mock)
    pool = sorted(all_contexts)
    selection_rng = rng_for(seed, VERSION, "question-selection")
    selected = sorted(selection_rng.sample(pool, QUESTION_COUNT))
    contexts = {qid: all_contexts[qid] for qid in selected}
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_turn_tone_dyadic_pilot",
        "protocol_version": VERSION,
        "base_prompt_version": prompts.PROMPT_VERSION,
        "implementation_sha256": implementation_hash(),
        "config": original["config"],
        "agreement_rubric": original["agreement_rubric"],
        "probability_readout": original["probability_readout"],
        "source": {**original["source"], "contexts_sha256": digest(contexts)},
        "question_selection": {
            "seed": seed,
            "pool_question_ids": pool,
            "selected_question_ids": selected,
            "count": QUESTION_COUNT,
            "uses_observed_outcomes": False,
        },
        "design": {
            "pairs": PAIRS,
            "depth": DEPTH,
            "tone_assignment_unit": "formal debate turn",
            "tone_assignment": "Presampled, balanced within directed pair × T; 80 turns of each tone overall",
            "historical_tone_instructions_visible": False,
            "historical_self_labels_visible": False,
            "historical_public_text_retained": True,
            "probe_tone": "neutral",
            "measurement_feedback": False,
            "initial_answers": "Reuse only archived independent initial answers for the selected 20 questions",
            "new_debate_generation": True,
            "new_measurement_generation": True,
            "global_tone_arms": False,
            "synthesis_or_layer_E": False,
            "three_member_run": False,
            "independent_repetitions": 0,
            "comparability": "Both tone assignment and probe instructions differ from the fixed-tone archive",
        },
        "sampling": {
            "seed": seed,
            "events_per_question": 8,
            "events": 160,
            "eligible_T": [2, 3, 4],
            "rule": "Omit one balanced pair × T node among the nine eligible nodes per question",
            "B_C_D_share_sample": True,
            "selection_uses_outputs": False,
            "reporting": "C/D retain previous and current self-labels, T, and D receiver model; tones are separate metadata",
            "T1": "Generate and retain for A; has no preceding reply label, not sampled for the two-label comparison",
        },
        "execution": original["execution"],
        "plans": plan_questions(selected, seed),
    }
    counts = Counter()
    for plan in manifest["plans"]:
        counts.update(
            task.purpose for task in TurnToneGraph(contexts[plan["question_id"]], plan, manifest).tasks.values()
        )
    manifest["planned_counts"] = {
        "questions": QUESTION_COUNT,
        "trajectories": QUESTION_COUNT * 3,
        "formal_replies": counts["debate"],
        "reused_initial_answers": QUESTION_COUNT * 3,
        "sampled_events": QUESTION_COUNT * 8,
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
    }
    return json.loads(canonical(manifest)), json.loads(canonical(contexts))


class TurnToneGraph(DyadicGraph):
    """Reuse validated navigation/recovery/readout arithmetic, not the fixed-tone task plan."""

    def key(self, suffix):
        return f"{self.question.id}/{VERSION}/{suffix}"

    def _build(self):
        question, config, route = self.question, self.config, self.route
        schedule = self.plan["tone_schedule"]
        if set(schedule) != {n.id for n in route.nodes} or any(t not in TONES for t in schedule.values()):
            raise ValueError("Incomplete or invalid frozen turn-level tone schedule")

        def add(suffix, deps, purpose, model, builder, effort, tokens, schema=None, labels=(), parse=None):
            key = self.key(suffix)
            if key in self.tasks:
                raise ValueError("Duplicate turn-tone task")
            self.tasks[key] = Task(
                key,
                frozenset(self.key(d) for d in deps),
                lambda values, count: Request(
                    key,
                    purpose,
                    model,
                    builder(values, count),
                    effort,
                    tokens,
                    schema,
                    config.cache and purpose in ("debate", "position"),
                    candidate_labels=labels,
                ),
                parse or (lambda text: prompts.parse_json(text, schema)),
                purpose,
                model,
            )

        for node in route.nodes:
            add(
                f"{STREAM}/debate/{node.id}",
                [f"{STREAM}/debate/{node.parent}"] if node.parent else [],
                "debate",
                config.members[node.receiver],
                lambda v, c, n=node: prompts.debate_messages(
                    question, schedule[n.id], self.initial, route, self.history(v, STREAM), n.id
                ),
                config.debate_effort,
                config.debate_tokens,
                prompts.REPLY_SCHEMA,
            )

        for reading in self.plan["readings"]:
            model = config.members[reading["member"]]
            add(
                f"C/{reading['id']}",
                [f"{STREAM}/debate/{reading['node_id']}"] if reading["node_id"] else [],
                "position",
                model,
                lambda v, c, r=reading: prompts.position_messages(
                    question, {**r, "tone": "neutral"}, self.initial, route, self.history(v, STREAM)
                ),
                "none",
                config.position_tokens,
                labels=question.labels if model in OPEN_MODELS else (),
                parse=lambda text: prompts.parse_position(text, question),
            )

        for event in self.plan["events"]:
            node = route.get(event["node_id"])
            model = config.members[node.receiver]
            add(
                f"B/{event['id']}",
                [f"{STREAM}/debate/{node.id}"],
                "judge_b",
                config.judge_model,
                lambda v, c, n=node: prompts.b_messages(
                    question, self.history(v, STREAM)[n.parent]["reply"], self.history(v, STREAM)[n.id]["reply"]
                ),
                "low",
                config.judge_tokens,
                prompts.B_SCHEMA,
            )
            if model not in OPEN_MODELS:
                continue
            for arm in ("argument", "control"):

                def d_messages(values, count, e=event, n=node, a=arm, m=model):
                    history = self.history(values, STREAM)
                    incoming = history[n.parent]["reply"]
                    if a == "control":
                        repeats = max(1, round(count(m, incoming) / count(m, prompts.FILLER_SENTENCE)))
                        incoming = " ".join([prompts.FILLER_SENTENCE] * repeats)
                    return prompts.d_text_messages(
                        question,
                        {**e, "tone": "neutral"},
                        self.initial,
                        route,
                        history,
                        values[self.key(f"C/{e['previous_reading']}")]["position"],
                        incoming,
                    )

                add(
                    f"Dtext/{event['id']}/{arm}",
                    [f"C/{event['previous_reading']}", f"{STREAM}/debate/{node.parent}"],
                    "d_text",
                    model,
                    d_messages,
                    "none",
                    config.probability_tokens,
                    labels=RATING_LABELS,
                    parse=prompts.parse_rating,
                )

        for pair in self.plan["text_pairs"]:
            add(
                f"Cjudge/{pair['id']}",
                [f"C/{pair['before']}", f"C/{pair['after']}"],
                "judge_c",
                config.judge_model,
                lambda v, c, p=pair: prompts.c_messages(
                    question, v[self.key(f"C/{p['before']}")]["position"], v[self.key(f"C/{p['after']}")]["position"]
                ),
                "low",
                config.judge_tokens,
                prompts.C_SCHEMA,
            )

    def report(self, count):
        # The base reconstructs C/D from saved reads without changing the measurements.
        # Replace its fixed-tone discussion/outcome inventory with this protocol's three chains.
        record = super().report(count)
        trajectories = []
        for pair in PAIRS:
            keys = {self.key(f"{STREAM}/debate/{pair}-{t}") for t in range(1, DEPTH + 1)}
            trajectories.append(
                {
                    "pair": pair,
                    "tone_sequence": [self.plan["tone_schedule"][f"{pair}-{t}"] for t in range(1, DEPTH + 1)],
                    "status": "success"
                    if keys <= self.values.keys()
                    else "failed"
                    if keys & (self.failed.keys() | self.blocked.keys())
                    else "pending",
                }
            )
        return {
            **record,
            "formal_replies": {STREAM: self.history(self.values, STREAM)},
            "tone_schedule": self.plan["tone_schedule"],
            "trajectories": trajectories,
            "probe_tone": "neutral",
        }
