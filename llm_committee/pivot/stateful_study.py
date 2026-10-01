"""Versioned four-field debate, observational C, and independent D side reads."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from . import prompts, stateful_prompts, strong_prompts
from .dyadic import PAIRS, make_route
from .models import OPEN_MODELS, Completion, PilotConfig, Request, canonical, digest
from .probabilities import RATING_LABELS, choice_change, distribution, own_position_difference
from .strong_agreement import AGREEMENT
from .strong_study import ARMS, QUESTION_COUNT
from .strong_study import prepare as prepare_layout
from .taskgraph import QuestionGraph, Task
from .turn_tone import STREAM

VERSION = "stateful-dyadic-paired-2026-09-27-v1"
PUBLIC_HISTORY_VERSION = "public-history-dyadic-paired-2026-09-27-v1"
PROTOCOL_PROMPTS = {
    VERSION: stateful_prompts.PROMPT_VERSION,
    PUBLIC_HISTORY_VERSION: stateful_prompts.PUBLIC_HISTORY_PROMPT_VERSION,
}


def prepare(source, *, mock=False, roster="mixed_family", public_history=False):
    # Reuse only the validated question/routing/tone/sample plan, never old outputs.
    previous, contexts = prepare_layout(source, mock=mock)
    config = replace(PilotConfig(**previous["config"]), roster=roster)
    config.validate()
    manifest = {
        **previous,
        "schema_version": 2,
        "kind": "offline_mock" if mock else "paid_stateful_dyadic_pilot",
        "protocol_version": PUBLIC_HISTORY_VERSION if public_history else VERSION,
        "prompt_version": PROTOCOL_PROMPTS[PUBLIC_HISTORY_VERSION if public_history else VERSION],
        "implementation_sha256": digest(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "config": asdict(config),
        "design": {
            "fresh_initial_answers": True,
            "fresh_debate_both_assignments": True,
            "tone_assignment_unit": "formal debate turn",
            "global_tone_arms": False,
            "shared_T1_T2_within_pair": True,
            "debate_output_fields": ["reply", "agreement", "choice", "position"],
            "initial_output_fields": ["choice", "position"],
            "formal_reasoning": config.debate_effort,
            "position_source": "same formal generation as the reply, not a separate re-ask",
            "position_update": "latest same-member ancestor on this branch; unchanged views are allowed",
            "explicit_position_feedback": not public_history,
            "formal_position_input": (
                "Initial answers and public replies only; updated choice/position fields are recorded, never reinjected"
                if public_history
                else "Latest same-member choice and full position explicitly supplied as your_current_position"
            ),
            "public_history": "public reply text only; no peer structured positions, self-labels or historical tones",
            "C": "Compare formal choices in code and formal full position texts with the judge",
            "D_enabled": roster == "mixed_family",
            "D_scope": "Only supported open-weight members of mixed_family; never the other rosters",
            "D1": "Neutral no-reasoning replay of each referenced formal generation's input, not its output",
            "D2": "Neutral argument/filler endorsement of the receiver's fixed pre-turn formal position",
            "D_outputs_enter_debate": False,
            "D2_modal_ties": "Retain all modes; category difference/reference-probability change null if ambiguous",
            "independent_question_count": len(contexts),
            "three_member_run": False,
            "layer_E": False,
            "reuse_archived_generations": False,
        },
        "execution": {
            **previous["execution"],
            "cache_order": "Within a sample, D1 then D2 argument then filler; other samples stay parallel",
            "cache_order_is_soft": True,
            "cache_note": "Wait for predecessor termination, including failure; no promised cache-hit discount",
        },
    }
    counts = Counter()
    for plan in manifest["plans"]:
        graph = StatefulGraph(contexts[plan["question_id"]], plan, manifest)
        counts.update(t.purpose for t in graph.tasks.values())
    manifest["planned_counts"] = {
        **previous["planned_counts"],
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
        "standalone_C_position_calls": 0,
        "formal_position_records": counts["initial"] + counts["debate"],
    }
    return json.loads(canonical(manifest)), contexts


def rating_metrics(argument, control):
    """All three complementary readouts; never use the sampled letter as the mode."""
    pa, pc = argument["probabilities"], control["probabilities"]
    modes = {
        name: [k for k in RATING_LABELS if p[k] == max(p.values())] for name, p in (("argument", pa), ("control", pc))
    }
    original = modes["control"][0] if len(modes["control"]) == 1 else None
    current = modes["argument"][0] if len(modes["argument"]) == 1 else None
    ratings = {k: i + 1 for i, k in enumerate(RATING_LABELS)}
    return {
        "modal_labels_control": modes["control"],
        "modal_labels_argument": modes["argument"],
        "modal_rating_control": ratings[original] if original else None,
        "modal_rating_argument": ratings[current] if current else None,
        "modal_rating_change": ratings[current] - ratings[original] if current and original else None,
        "reference_category": original,
        "reference_probability_control": pc[original] if original else None,
        "reference_probability_argument": pa[original] if original else None,
        "reference_probability_change_pp": 100 * (pa[original] - pc[original]) if original else None,
        "mean_rating_control": sum(ratings[k] * pc[k] for k in RATING_LABELS),
        "mean_rating_argument": sum(ratings[k] * pa[k] for k in RATING_LABELS),
        "mean_own_agreement_argument_minus_control": own_position_difference(argument, control),
    }


class StatefulGraph(QuestionGraph):
    def __init__(self, context, plan, manifest):
        from .models import Question

        if (
            manifest["protocol_version"] not in PROTOCOL_PROMPTS
            or manifest["prompt_version"] != PROTOCOL_PROMPTS[manifest["protocol_version"]]
        ):
            raise ValueError("Wrong protocol for the stateful executor")
        self.explicit_position = manifest["protocol_version"] == VERSION
        if not self.explicit_position and manifest["design"].get("explicit_position_feedback") is not False:
            raise ValueError("Public-history protocol must not feed private positions back")
        self.question = Question.from_dict(context["question"])
        self.config = PilotConfig(**manifest["config"])
        self.config.validate()
        self.context, self.plan, self.manifest = context, plan, manifest
        self.route = make_route()
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted = set()
        self.arm_keys = {a: set() for a in ARMS}
        self.cache_after = {}
        self.shared = set()
        for n in self.route.nodes:
            if n.depth <= 2:
                self.shared.add(f"{STREAM}/debate/{n.id}")
            if n.depth <= 3:
                self.shared.add(f"Dchoice/{STREAM}/{n.id}")
        for e in plan["arms"]["original"]["events"]:
            if e["T"] <= 2:
                self.shared.add(f"B/{e['id']}")
            if e["T"] <= 3:
                self.shared.update(f"Dtext/{e['id']}/{t}" for t in ("argument", "control"))
        for p in plan["arms"]["original"]["text_pairs"]:
            if self.route.get(p["after"].split("/", 1)[1]).depth <= 2:
                self.shared.add(f"Cjudge/{p['id']}")
        self._build()
        reachable = set()
        while more := {t.key for t in self.tasks.values() if t.dependencies <= reachable} - reachable:
            reachable.update(more)
        if reachable != self.tasks.keys():
            raise ValueError("Stateful graph has invalid data dependencies")
        # Soft ordering must also be acyclic, but is never a failure dependency.
        reachable = set()
        while (
            more := {
                t.key
                for t in self.tasks.values()
                if t.dependencies | self.cache_after.get(t.key, frozenset()) <= reachable
            }
            - reachable
        ):
            reachable.update(more)
        if reachable != self.tasks.keys():
            raise ValueError("Stateful graph has invalid cache ordering")

    def key(self, arm, suffix):
        prefix = f"{self.question.id}/{self.manifest['protocol_version']}/{self.config.roster}"
        if suffix.startswith(("initial/", "Dchoice/initial/")):
            return f"{prefix}/{suffix}"
        return f"{prefix}/{'original' if suffix in self.shared else arm}/{suffix}"

    def source_key(self, arm, reading):
        suffix = reading if reading.startswith("initial/") else reading.replace(f"{STREAM}/", f"{STREAM}/debate/", 1)
        return self.key(arm, suffix)

    def initial(self, values):
        return {
            m: values[self.key("original", f"initial/{m}")]
            for m in range(3)
            if self.key("original", f"initial/{m}") in values
        }

    def history(self, values, arm):
        return {
            n.id: values[self.key(arm, f"{STREAM}/debate/{n.id}")]
            for n in self.route.nodes
            if self.key(arm, f"{STREAM}/debate/{n.id}") in values
        }

    def position(self, arm, reading):
        key = self.source_key(arm, reading)
        value = self.values.get(key)
        return {"choice": value["choice"], "position": value["position"], "source_request": key} if value else None

    def context_dependencies(self, arm, node):
        deps = {
            self.key(arm, f"initial/{node.receiver}"),
            self.key(arm, f"initial/{self.route.path(node.id)[0].sender}"),
        }
        if node.parent:
            deps.add(self.key(arm, f"{STREAM}/debate/{node.parent}"))
        previous = self.route.previous_own(node.id)
        if previous:
            deps.add(self.key(arm, f"{STREAM}/debate/{previous.id}"))
        return deps

    def ready(self):
        terminal = self.values.keys() | self.failed.keys() | self.blocked.keys()
        return next(
            (
                t
                for t in self.tasks.values()
                if t.key not in self.submitted
                and t.key not in self.blocked
                and t.dependencies <= self.values.keys()
                and self.cache_after.get(t.key, frozenset()) <= terminal
            ),
            None,
        )

    def _build(self):
        q, cfg, route = self.question, self.config, self.route
        d_enabled = cfg.roster == "mixed_family"

        def add(arm, suffix, deps, purpose, model, builder, effort, tokens, schema=None, labels=(), parse=None):
            key = self.key(arm, suffix)
            self.arm_keys[arm].add(key)
            if key in self.tasks:
                if self.tasks[key].dependencies != frozenset(deps) or self.tasks[key].purpose != purpose:
                    raise ValueError("Conflicting shared stateful task")
                return key
            self.tasks[key] = Task(
                key,
                frozenset(deps),
                lambda v, count: Request(
                    key, purpose, model, builder(v, count), effort, tokens, schema, cfg.cache, candidate_labels=labels
                ),
                parse or (lambda text: prompts.parse_json(text, schema)),
                purpose,
                model,
            )
            return key

        for arm in ARMS:
            plan = self.plan["arms"][arm]
            for m, model in enumerate(cfg.members):
                add(
                    arm,
                    f"initial/{m}",
                    [],
                    "initial",
                    model,
                    lambda v, c, m=m: stateful_prompts.initial_messages(q, m),
                    cfg.debate_effort,
                    cfg.initial_tokens,
                    stateful_prompts.position_schema(q),
                )
            for node in route.nodes:
                add(
                    arm,
                    f"{STREAM}/debate/{node.id}",
                    self.context_dependencies(arm, node),
                    "debate",
                    cfg.members[node.receiver],
                    lambda v, c, a=arm, n=node: stateful_prompts.debate_messages(
                        q,
                        self.plan["arms"][a]["tone_schedule"][n.id],
                        self.initial(v),
                        route,
                        self.history(v, a),
                        n.id,
                        explicit_position=self.explicit_position,
                    ),
                    cfg.debate_effort,
                    cfg.debate_tokens,
                    stateful_prompts.reply_schema(q),
                )
            for r in plan["readings"]:
                model = cfg.members[r["member"]]
                if not d_enabled or model not in OPEN_MODELS:
                    continue
                deps = self.context_dependencies(arm, route.get(r["node_id"])) if r["node_id"] else set()
                add(
                    arm,
                    f"Dchoice/{r['id']}",
                    deps,
                    "d_choice",
                    model,
                    lambda v, c, a=arm, r=r: stateful_prompts.choice_messages(
                        q,
                        r["member"],
                        self.initial(v),
                        route,
                        self.history(v, a),
                        r["node_id"],
                        explicit_position=self.explicit_position,
                    ),
                    "none",
                    cfg.probability_tokens,
                    labels=q.labels,
                    parse=lambda text: stateful_prompts.parse_choice(text, q),
                )
            for event in plan["events"]:
                node = route.get(event["node_id"])
                if cfg.judge_model:
                    add(
                        arm,
                        f"B/{event['id']}",
                        [self.key(arm, f"{STREAM}/debate/{node.id}")],
                        "judge_b",
                        cfg.judge_model,
                        lambda v, c, a=arm, n=node: strong_prompts.b_messages(
                            q, self.history(v, a)[n.parent]["reply"], self.history(v, a)[n.id]["reply"]
                        ),
                        "low",
                        cfg.judge_tokens,
                        strong_prompts.B_SCHEMA,
                    )
                model = cfg.members[node.receiver]
                if not d_enabled or model not in OPEN_MODELS:
                    continue
                for treatment in ("argument", "control"):

                    def messages(v, count, a=arm, n=node, t=treatment, m=model):
                        history = self.history(v, a)
                        incoming = history[n.parent]["reply"]
                        if t == "control":
                            repeats = max(1, round(count(m, incoming) / count(m, prompts.FILLER_SENTENCE)))
                            incoming = " ".join([prompts.FILLER_SENTENCE] * repeats)
                        return stateful_prompts.text_messages(
                            q,
                            self.initial(v),
                            route,
                            history,
                            n.id,
                            incoming,
                            explicit_position=self.explicit_position,
                        )

                    key = add(
                        arm,
                        f"Dtext/{event['id']}/{treatment}",
                        self.context_dependencies(arm, node),
                        "d_text",
                        model,
                        messages,
                        "none",
                        cfg.probability_tokens,
                        labels=RATING_LABELS,
                        parse=prompts.parse_rating,
                    )
                    predecessor = (
                        self.key(arm, f"Dchoice/{event['current_reading']}")
                        if treatment == "argument"
                        else self.key(arm, f"Dtext/{event['id']}/argument")
                    )
                    self.cache_after[key] = frozenset({predecessor})
            if cfg.judge_model:
                for pair in plan["text_pairs"]:
                    before, after = (self.source_key(arm, pair[k]) for k in ("before", "after"))
                    add(
                        arm,
                        f"Cjudge/{pair['id']}",
                        [before, after],
                        "judge_c",
                        cfg.judge_model,
                        lambda v, c, b=before, a=after: prompts.c_messages(q, v[b]["position"], v[a]["position"]),
                        "low",
                        cfg.judge_tokens,
                        prompts.C_SCHEMA,
                    )

    def report(self, count):
        def dist(key, labels):
            value = self.values.get(key)
            return (
                distribution(value["_readout"], labels, missing_as_zero=True) if value and "_readout" in value else None
            )

        def changed(a, b):
            return (
                a["choice"] != b["choice"] if a and b and a["choice"] is not None and b["choice"] is not None else None
            )

        branches, trajectories, positions = {}, [], {}
        for arm in ARMS:
            plan = self.plan["arms"][arm]
            events = []
            for e in plan["events"]:
                node = self.route.get(e["node_id"])
                peer, reply = (
                    self.values.get(self.key(arm, f"{STREAM}/debate/{nid}")) for nid in (node.parent, node.id)
                )
                before, initial, after = (
                    self.position(arm, e[k]) for k in ("previous_reading", "initial_reading", "current_reading")
                )
                ds = [
                    dist(self.key(arm, f"Dchoice/{e[k]}"), self.question.labels)
                    for k in ("previous_reading", "initial_reading", "current_reading")
                ]
                ratings = {
                    t: dist(self.key(arm, f"Dtext/{e['id']}/{t}"), RATING_LABELS) for t in ("argument", "control")
                }
                d_text = None
                if all(ratings.values()):
                    model = self.config.members[node.receiver]
                    repeats = max(1, round(count(model, peer["reply"]) / count(model, prompts.FILLER_SENTENCE)))
                    d_text = {
                        **ratings,
                        **rating_metrics(ratings["argument"], ratings["control"]),
                        "fixed_full_position": before["position"],
                        "position_source_request": before["source_request"],
                        "filler_repetitions": repeats,
                        "argument_tokens": count(model, peer["reply"]),
                        "filler_tokens": count(model, " ".join([prompts.FILLER_SENTENCE] * repeats)),
                    }
                events.append(
                    {
                        **e,
                        "previous_peer_self_label": peer["agreement"] if peer else None,
                        "current_self_label": reply["agreement"] if reply else None,
                        "peer_reply_available": peer is not None,
                        "current_reply_available": reply is not None,
                        "B": self.values.get(self.key(arm, f"B/{e['id']}")),
                        "position_before": before,
                        "position_initial": initial,
                        "position_after": after,
                        "C_choice_changed": changed(before, after),
                        "C_initial_choice_changed": changed(initial, after),
                        "C_text": self.values.get(self.key(arm, f"Cjudge/{e['adjacent_pair']}")),
                        "C_initial_to_endpoint": (
                            self.values.get(self.key(arm, f"Cjudge/{e['final_pair']}")) if e["final_pair"] else None
                        ),
                        "D_choice_before": ds[0],
                        "D_choice_initial": ds[1],
                        "D_choice_after": ds[2],
                        "D_choice_adjacent": (
                            choice_change(ds[0], ds[2], before["choice"]) if ds[0] and ds[2] and before else None
                        ),
                        "D_choice_initial_to_current": (
                            choice_change(ds[1], ds[2], initial["choice"]) if ds[1] and ds[2] and initial else None
                        ),
                        "D_choice_method": self.manifest["design"]["D1"],
                        "D_choice_request": self.key(arm, f"Dchoice/{e['current_reading']}") if ds[2] else None,
                        "D_choice_shared": bool(ds[2]) and e["T"] <= 3,
                        "D_text": d_text,
                        "D_text_shared": bool(d_text) and e["T"] <= 3,
                        "shared_prefix_event": e["T"] <= 2,
                    }
                )
            paths = []
            for leaf in self.route.leaves:
                pair = leaf.id.split("-", 1)[0]
                path = self.route.path(leaf.id)
                keys = {self.key(arm, f"{STREAM}/debate/{n.id}") for n in path}
                status = (
                    "success"
                    if keys <= self.values.keys()
                    else "failed"
                    if keys & (self.failed.keys() | self.blocked.keys())
                    else "pending"
                )
                paths.append(
                    {
                        "pair": pair,
                        "assignment": arm,
                        "status": status,
                        "tone_sequence": [plan["tone_schedule"][n.id] for n in path],
                    }
                )
                state = {}
                for member in PAIRS[pair]:
                    initial = self.position(arm, f"initial/{member}")
                    rows = [{**initial, "T": 0, "participation_index": 0, "node_id": None}] if initial else []
                    for n in path:
                        if n.receiver != member:
                            continue
                        p = self.position(arm, f"{STREAM}/{n.id}")
                        if p:
                            rows.append(
                                {
                                    **p,
                                    "T": n.depth,
                                    "participation_index": self.route.participation(n.id),
                                    "node_id": n.id,
                                }
                            )
                    state[str(member)] = rows
                positions[f"{arm}/{pair}"] = state
            branch_keys = self.arm_keys[arm]
            branches[arm] = {
                "assignment": arm,
                "question_id": self.question.id,
                "question": asdict(self.question),
                "formal_replies": {STREAM: self.history(self.values, arm)},
                "tone_schedule": plan["tone_schedule"],
                "events": events,
                "trajectories": paths,
                "failed_tasks": {k: v for k, v in self.failed.items() if k in branch_keys},
                "blocked_tasks": {k: v for k, v in self.blocked.items() if k in branch_keys},
            }
            trajectories.extend(paths)
        return {
            "protocol_version": self.manifest["protocol_version"],
            "question_id": self.question.id,
            "question": asdict(self.question),
            "status": "completed" if self.successful else "completed_with_failures" if self.complete else "partial",
            "initial_positions": {str(m): v for m, v in self.initial(self.values).items()},
            "branches": branches,
            "trajectories": trajectories,
            "positions": positions,
            "position_indexing": "positions[assignment/pair][member_id][participation_index]; inspect T/node_id when failures leave gaps",
            "failed_tasks": self.failed,
            "blocked_tasks": self.blocked,
            "request_keys": sorted(self.tasks),
            "cache_after": {k: sorted(v) for k, v in self.cache_after.items()},
        }


class StatefulMockProvider:
    """Offline plumbing fixture, never a source of scientific results."""

    def __init__(self):
        from .providers import MockProvider

        self.base = MockProvider()

    def token_count(self, model, text):
        return self.base.token_count(model, text)

    def generate(self, request):
        if request.purpose in ("initial", "debate"):
            label = request.schema["properties"]["choice"]["enum"][0]
            value = {"choice": label, "position": f"SYNTHETIC recorded full position {request.key}"}
            if request.purpose == "debate":
                value = {
                    "reply": f"SYNTHETIC public reply {request.key}",
                    "agreement": AGREEMENT[int(digest(request.key)[:8], 16) % 4],
                    **value,
                }
            return Completion(canonical(value), 100, 100, reasoning_tokens=50, raw={"mock": True})
        if request.purpose == "d_choice":
            return Completion(
                request.candidate_labels[0], 100, 1, raw={"mock": True}, readout=self.base.synthetic_readout(request)
            )
        if request.purpose == "judge_b":
            return Completion(
                canonical({"label": AGREEMENT[0], "evidence": "SYNTHETIC evidence"}), 100, 30, raw={"mock": True}
            )
        return self.base.generate(request)
