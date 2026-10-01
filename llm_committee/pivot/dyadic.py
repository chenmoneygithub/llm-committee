"""Frozen two-member, four-reply supplements; no third-member context or branching."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import prompts
from .agreement import agreement_manifest
from .failures import FORMAT_RETRY_POLICY
from .models import OPEN_MODELS, TONES, PilotConfig, Question, Request, canonical, digest
from .planning import Node, Route, rng_for
from .probabilities import RATING_LABELS, TOPK_ZERO_FILL_POLICY, choice_change, distribution, own_position_difference
from .taskgraph import QuestionGraph, Task

VERSION = "dyadic-2026-09-26-v1"
PAIRS = {"AB": (0, 1), "CA": (2, 0), "BC": (1, 2)}
DEPTH = 4


def make_route():
    nodes = []
    for pair, (sender, receiver) in PAIRS.items():
        parent = None
        for depth in range(1, DEPTH + 1):
            node = Node(f"{pair}-{depth}", parent, sender, receiver, depth)
            nodes.append(node)
            parent, sender, receiver = node.id, receiver, sender
    # Route's navigation is generic; its main-study-only six-leaf validator is not.
    return Route(tuple(nodes))


def plan_questions(questions, seed):
    """Eight of nine pair/tone strata per question, one preselected T=2/3/4 each.

    Omitted strata are balanced across questions, independently of outputs.
    T weights 1/1/1.5 retain the agreed slight preference for later positions.
    T=1 has no previous reply label and is not a two-label measurement event.
    """
    route = make_route()
    strata = [(pair, tone) for pair in PAIRS for tone in TONES]
    order = sorted(questions, key=lambda q: q.id)
    rng = rng_for(seed, VERSION, "sample")
    omissions = (strata * ((len(order) + 8) // 9))[: len(order)]
    rng.shuffle(omissions)
    plans = []
    for question, omitted in zip(order, omissions, strict=True):
        readings, comparisons, events = {}, {}, []

        def read(tone, member, node, readings=readings):
            rid = f"{tone}/{node.id}" if node else f"initial/{member}"
            readings[rid] = {
                "id": rid,
                "tone": tone if node else "initial",
                "member": member,
                "node_id": node.id if node else None,
                "T": node.depth if node else 0,
                "participation_index": route.participation(node.id) if node else 0,
            }
            return rid

        def compare(before, after, kind, comparisons=comparisons):
            pid = f"{before}->{after}"
            comparisons.setdefault(pid, {"id": pid, "before": before, "after": after, "kinds": []})
            if kind not in comparisons[pid]["kinds"]:
                comparisons[pid]["kinds"].append(kind)
            return pid

        for pair, tone in strata:
            if (pair, tone) == omitted:
                continue
            depth = rng.choices((2, 3, 4), weights=(1, 1, 1.5), k=1)[0]
            node = route.get(f"{pair}-{depth}")
            previous = read(tone, node.receiver, route.previous_own(node.id))
            initial = read(tone, node.receiver, None)
            current = read(tone, node.receiver, node)
            events.append(
                {
                    "id": f"{tone}/{node.id}",
                    "pair": pair,
                    "tone": tone,
                    "node_id": node.id,
                    "member": node.receiver,
                    "peer": node.sender,
                    "T": depth,
                    "participation_index": route.participation(node.id),
                    "previous_reading": previous,
                    "initial_reading": initial,
                    "current_reading": current,
                    "adjacent_pair": compare(previous, current, "adjacent"),
                    "final_pair": compare(initial, current, "initial_to_endpoint") if depth >= 3 else None,
                    "depth_selection_probability": (1.5 if depth == 4 else 1) / 3.5,
                }
            )
        plans.append(
            {
                "question_id": question.id,
                "omitted_measurement_stratum": list(omitted),
                "events": events,
                "readings": list(readings.values()),
                "text_pairs": list(comparisons.values()),
            }
        )
    return plans


def implementation_hash():
    names = (
        "dyadic.py",
        "dyadic_run.py",
        "prompts.py",
        "agreement.py",
        "models.py",
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
    return digest({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})


def prepare(source: Path, *, mock=False, seed=20260926):
    source = source.resolve()
    raw = (source / "manifest.json").read_bytes()
    original = json.loads(raw)
    report = json.loads((source / "report.json").read_text())
    if (
        original.get("kind") != "paid_main_study"
        or report["status"] != "completed"
        or original.get("prompt_version") != prompts.PROMPT_VERSION
        or original.get("agreement_rubric") != agreement_manifest()
    ):
        raise ValueError("Need the completed real leaning-label main study")
    questions = [Question.from_dict(q) for q in original["questions"]]
    if len(questions) != 60:
        raise ValueError("Expected all 60 approved questions")
    config = PilotConfig(**original["config"])
    config.validate()
    if config.roster != "mixed_family" or config.judge_model != "gemini-3.8-flash":
        raise ValueError("Do not change the original models or judge")
    contexts, hashes = {}, {}
    for question in questions:
        data = (source / "questions" / f"{question.id}.json").read_bytes()
        hashes[question.id] = hashlib.sha256(data).hexdigest()
        record = json.loads(data)
        if record["status"] != "completed" or len(record["questions"]) != 1:
            raise ValueError("Incomplete source question")
        result = record["questions"][0]
        if result["question_fingerprint"] != question.fingerprint or result["question_id"] != question.id:
            raise ValueError("Source question mismatch")
        contexts[question.id] = {"question": asdict(question), "initial_answers": result["initial_answers"]}
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_dyadic_supplement",
        "protocol_version": VERSION,
        "base_prompt_version": prompts.PROMPT_VERSION,
        "implementation_sha256": implementation_hash(),
        "config": asdict(config),
        "agreement_rubric": agreement_manifest(),
        "probability_readout": dict(TOPK_ZERO_FILL_POLICY),
        "source": {
            "directory": str(source),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "question_sha256": hashes,
            "contexts_sha256": digest(contexts),
        },
        "design": {
            "pairs": PAIRS,
            "depth": DEPTH,
            "tones": TONES,
            "independent_repetitions": 0,
            "initial_answers": "Reuse only each member's archived independent initial answer; no third member",
            "measurement_feedback": False,
            "synthesis_or_layer_E": False,
            "direction_note": "Fixed AB/CA/BC starts; each model starts once and receives first once per question/tone",
        },
        "sampling": {
            "seed": seed,
            "events_per_question": 8,
            "events": 480,
            "eligible_T": [2, 3, 4],
            "T_weights": [1, 1, 1.5],
            "rule": "Omit one balanced pair/tone stratum per question; sample one depth in each other stratum",
            "B_C_D_share_sample": True,
            "selection_uses_outputs": False,
            "reporting": "Condition C/D on previous peer self-label AND current receiver self-label; retain T",
            "T1": "All T=1 debate replies retained for A; no invented previous self-label",
            "missing_labels": "Retain null as not reported; never replace with an agreement level",
        },
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "spend_cap_usd": None,
            "failure_policy": FORMAT_RETRY_POLICY,
            "max_inflight_requests": 64,
            "question_workers": 8,
        },
        "plans": plan_questions(questions, seed),
    }
    counts = Counter()
    for plan in manifest["plans"]:
        graph = DyadicGraph(contexts[plan["question_id"]], plan, manifest)
        counts.update(task.purpose for task in graph.tasks.values())
    manifest["planned_counts"] = {
        "questions": 60,
        "trajectories": 540,
        "formal_replies": counts["debate"],
        "reused_initial_answers": 180,
        "sampled_events": 480,
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
    }
    return json.loads(canonical(manifest)), json.loads(canonical(contexts))


class DyadicGraph(QuestionGraph):
    def __init__(self, context, plan, manifest):
        self.question = Question.from_dict(context["question"])
        self.config = PilotConfig(**manifest["config"])
        self.context, self.plan, self.manifest = context, plan, manifest
        self.route = make_route()
        self.initial = {int(k): v for k, v in context["initial_answers"].items()}
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted = set()
        self._build()
        reachable = set()
        while True:
            more = {t.key for t in self.tasks.values() if t.dependencies <= reachable} - reachable
            if not more:
                break
            reachable.update(more)
        if reachable != self.tasks.keys():
            raise ValueError("Missing dependencies or cyclic dyadic graph")

    def key(self, suffix):
        return f"{self.question.id}/{VERSION}/{suffix}"

    def history(self, values, tone):
        return {
            n.id: values[self.key(f"{tone}/debate/{n.id}")]
            for n in self.route.nodes
            if self.key(f"{tone}/debate/{n.id}") in values
        }

    def _build(self):
        question, config, route = self.question, self.config, self.route

        def add(suffix, deps, purpose, model, builder, effort, tokens, schema=None, labels=(), parse=None):
            key = self.key(suffix)
            if key in self.tasks:
                raise ValueError("Duplicate dyadic task")
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

        for tone in TONES:
            for node in route.nodes:
                add(
                    f"{tone}/debate/{node.id}",
                    [f"{tone}/debate/{node.parent}"] if node.parent else [],
                    "debate",
                    config.members[node.receiver],
                    lambda v, c, t=tone, n=node: prompts.debate_messages(
                        question, t, self.initial, route, self.history(v, t), n.id
                    ),
                    config.debate_effort,
                    config.debate_tokens,
                    prompts.REPLY_SCHEMA,
                )

        for reading in self.plan["readings"]:
            model = config.members[reading["member"]]
            add(
                f"C/{reading['id']}",
                [f"{reading['tone']}/debate/{reading['node_id']}"] if reading["node_id"] else [],
                "position",
                model,
                lambda v, c, r=reading: prompts.position_messages(
                    question, r, self.initial, route, self.history(v, r["tone"])
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
                [f"{event['tone']}/debate/{node.id}"],
                "judge_b",
                config.judge_model,
                lambda v, c, e=event, n=node: prompts.b_messages(
                    question, self.history(v, e["tone"])[n.parent]["reply"], self.history(v, e["tone"])[n.id]["reply"]
                ),
                "low",
                config.judge_tokens,
                prompts.B_SCHEMA,
            )
            if model not in OPEN_MODELS:
                continue
            for arm in ("argument", "control"):

                def d_messages(values, count, e=event, n=node, a=arm, m=model):
                    history = self.history(values, e["tone"])
                    incoming = history[n.parent]["reply"]
                    if a == "control":
                        repeats = max(1, round(count(m, incoming) / count(m, prompts.FILLER_SENTENCE)))
                        incoming = " ".join([prompts.FILLER_SENTENCE] * repeats)
                    return prompts.d_text_messages(
                        question,
                        e,
                        self.initial,
                        route,
                        history,
                        values[self.key(f"C/{e['previous_reading']}")]["position"],
                        incoming,
                    )

                add(
                    f"Dtext/{event['id']}/{arm}",
                    [f"C/{event['previous_reading']}", f"{event['tone']}/debate/{node.parent}"],
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
        def get(suffix):
            return self.values.get(self.key(suffix))

        def dist(value, labels):
            return (
                distribution(value["_readout"], labels, missing_as_zero=True) if value and "_readout" in value else None
            )

        def position(value):
            return {k: v for k, v in value.items() if k != "_readout"} if value else None

        def changed(before, after):
            return (
                before["choice"] != after["choice"]
                if before and after and before["choice"] is not None and after["choice"] is not None
                else None
            )

        events = []
        for event in self.plan["events"]:
            node = self.route.get(event["node_id"])
            peer = get(f"{event['tone']}/debate/{node.parent}")
            reply = get(f"{event['tone']}/debate/{node.id}")
            before, initial, after = [
                get(f"C/{event[name]}") for name in ("previous_reading", "initial_reading", "current_reading")
            ]
            before_dist, initial_dist, after_dist = [dist(v, self.question.labels) for v in (before, initial, after)]
            rating = {arm: dist(get(f"Dtext/{event['id']}/{arm}"), RATING_LABELS) for arm in ("argument", "control")}
            d_text = None
            if all(rating.values()):
                model = self.config.members[node.receiver]
                repeats = max(1, round(count(model, peer["reply"]) / count(model, prompts.FILLER_SENTENCE)))
                d_text = {
                    **rating,
                    "fixed_full_position": before["position"],
                    "filler_repetitions": repeats,
                    "argument_tokens": count(model, peer["reply"]),
                    "filler_tokens": count(model, " ".join([prompts.FILLER_SENTENCE] * repeats)),
                    "mean_own_agreement_argument_minus_control": own_position_difference(
                        rating["argument"], rating["control"]
                    ),
                }
            events.append(
                {
                    **event,
                    "previous_peer_self_label": peer["agreement"] if peer else None,
                    "current_self_label": reply["agreement"] if reply else None,
                    "peer_reply_available": peer is not None,
                    "current_reply_available": reply is not None,
                    "B": get(f"B/{event['id']}"),
                    "position_before": position(before),
                    "position_initial": position(initial),
                    "position_after": position(after),
                    "C_choice_changed": changed(before, after),
                    "C_initial_choice_changed": changed(initial, after),
                    "C_text": get(f"Cjudge/{event['adjacent_pair']}"),
                    "C_initial_to_endpoint": get(f"Cjudge/{event['final_pair']}") if event["final_pair"] else None,
                    "D_choice_before": before_dist,
                    "D_choice_initial": initial_dist,
                    "D_choice_after": after_dist,
                    "D_choice_adjacent": choice_change(before_dist, after_dist, before["choice"])
                    if before_dist and after_dist
                    else None,
                    "D_choice_initial_to_current": choice_change(initial_dist, after_dist, initial["choice"])
                    if initial_dist and after_dist
                    else None,
                    "D_text": d_text,
                }
            )
        trajectories = []
        for tone in TONES:
            for pair in PAIRS:
                keys = {self.key(f"{tone}/debate/{pair}-{t}") for t in range(1, DEPTH + 1)}
                trajectories.append(
                    {
                        "pair": pair,
                        "tone": tone,
                        "status": "success"
                        if keys <= self.values.keys()
                        else "failed"
                        if keys & (self.failed.keys() | self.blocked.keys())
                        else "pending",
                    }
                )
        return {
            "question_id": self.question.id,
            "question": asdict(self.question),
            "status": "completed" if self.successful else "completed_with_failures" if self.complete else "partial",
            "initial_answers": self.context["initial_answers"],
            "formal_replies": {tone: self.history(self.values, tone) for tone in TONES},
            "trajectories": trajectories,
            "events": events,
            "failed_tasks": self.failed,
            "blocked_tasks": self.blocked,
            "request_keys": sorted(self.tasks),
        }
