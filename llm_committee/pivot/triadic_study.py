"""Fresh A/B/E experiment on archived three-member trees; no local C/D probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import prompts, stateful_prompts, strong_prompts
from .failures import FORMAT_RETRY_POLICY, OUTPUT_LIMIT_RETRY_POLICY
from .models import TONES, Message, PilotConfig, Question, Request, canonical, digest
from .planning import Node, Route, rng_for
from .quality_study import (
    ANSWER_SCHEMA,
    CALIBRATION_KINDS,
    LENGTH,
    VALIDATION_SCHEMA,
    VERBOSE_LENGTH,
    QualityMockProvider,
    length_contract,
    parse_answer,
    validation_messages,
    variant_messages,
    words,
)
from .stateful_study import StatefulMockProvider
from .strong_agreement import agreement_manifest
from .taskgraph import QuestionGraph, Task

VERSION = "public-history-triadic-ABE-2026-09-28-v1"
DESIGN_VERSION = "triadic-archived-routing-local-tones-2026-09-28-v1"
ARMS = ("original", "alternate")
SEED = 20260928


def code_hash():
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    )


def route_for(plan):
    route = Route(tuple(Node(**n) for n in plan["route"]))
    route.validate()
    return route


def make_design(question_plan, route_manifest):
    """Copy archived routes exactly, but never old responses or global tone arms."""
    question_plan, route_manifest = Path(question_plan).resolve(), Path(route_manifest).resolve()
    template = json.loads(question_plan.read_text())
    if digest({k: v for k, v in template.items() if k != "design_sha256"}) != template["design_sha256"]:
        raise ValueError("Question template changed")
    archive = json.loads(route_manifest.read_text())
    old_questions = {q["id"]: q for q in archive["questions"]}
    old_plans = {p["question_id"]: p for p in archive["plans"]}
    plans = []
    for item in template["plans"]:
        q = Question.from_dict(item["question"])
        if item["question"] != old_questions[q.id]:
            raise ValueError("Question/options differ between frozen sources")
        plan = {"question_id": q.id, "question": item["question"], "route": old_plans[q.id]["route"]}
        route = route_for(plan)
        rng = rng_for(SEED, DESIGN_VERSION, q.id, "sampling")
        # Same eight base B events per question as the dyadic design, with no label-based selection.
        eligible = [n for n in route.nodes if n.depth >= 2]
        chosen = [rng.choice([n for n in eligible if n.depth == t]) for t in range(2, 6)]
        while len(chosen) < 8:
            pool = [n for n in eligible if n not in chosen]
            chosen.append(rng.choices(pool, weights=[1.5 if n.depth == 5 else 1 for n in pool], k=1)[0])
        plan.update(
            sampled_b_nodes=sorted(n.id for n in chosen),
            sampled_endpoint_leaves=sorted(n.id for n in rng.sample(list(route.leaves), 2)),
            tones={a: {} for a in ARMS},
        )
        plans.append(plan)
    if not plans or len({p["question_id"] for p in plans}) != len(plans):
        raise ValueError("Expected distinct presampled questions")
    # Balance original tones by depth and receiver across questions. A cyclic shift gives
    # every suffix node a different tone while retaining those stratum counts up to one.
    for depth in range(1, 6):
        for member in range(3):
            nodes = [(p, n) for p in plans for n in route_for(p).nodes if n.depth == depth and n.receiver == member]
            rng = rng_for(SEED, DESIGN_VERSION, "tones", depth, member)
            tones = [TONES[i % 3] for i in range(len(nodes))]
            rng.shuffle(tones)
            shift = rng.choice((1, 2))
            for (plan, node), tone in zip(nodes, tones, strict=True):
                plan["tones"]["original"][node.id] = tone
                plan["tones"]["alternate"][node.id] = tone if depth <= 2 else TONES[(TONES.index(tone) + shift) % 3]
    calibration_qids = rng_for(SEED, DESIGN_VERSION, "calibration").sample(
        sorted(p["question_id"] for p in plans), min(12, len(plans))
    )
    for plan in plans:
        qid = plan["question_id"]
        plan["first_debate_left"] = {a: bool(rng_for(SEED, qid, a, "order").randrange(2)) for a in ARMS}
        plan["calibration_arm"] = ARMS[calibration_qids.index(qid) % 2] if qid in calibration_qids else None
    body = {
        "design_id": DESIGN_VERSION,
        "seed": SEED,
        "sources": {
            "question_plan": str(question_plan),
            "question_plan_sha256": hashlib.sha256(question_plan.read_bytes()).hexdigest(),
            "route_manifest": str(route_manifest),
            "route_manifest_sha256": hashlib.sha256(route_manifest.read_bytes()).hexdigest(),
        },
        "shared_config": template["shared_config"],
        "contract": {
            "archived_routes_unchanged": True,
            "shared_depths": [1, 2],
            "different_tone_depths": [3, 4, 5],
            "B_base_nodes_per_question": 8,
            "E1_sampled_paired_leaves_per_question": 2,
            "E1_unparticipated_members": "Retain initial-only histories, but exclude from position-change denominators",
            "E2_unit": "one whole six-leaf discussion tree per question and tone assignment",
            "E2_baseline": "one synthesis of the same three initial answers, shared across assignments",
            "local_C_D": "not collected, not zero changes",
            "selection_uses_outputs": False,
        },
        "plans": plans,
    }
    return {**body, "design_sha256": digest(body)}


def prepare(plan_file, *, roster="mixed_family", mock=False):
    plan_file = Path(plan_file).resolve()
    design = json.loads(plan_file.read_text())
    if (
        design["design_id"] != DESIGN_VERSION
        or digest({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]
    ):
        raise ValueError("Frozen triadic design changed")
    cfg = PilotConfig(roster=roster, **design["shared_config"])
    cfg.validate()
    contexts = {p["question_id"]: {"question": p["question"]} for p in design["plans"]}
    if not contexts or len(contexts) != len(design["plans"]):
        raise ValueError("Empty or duplicated question plan")
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_triadic_ABE_pilot",
        "protocol_version": VERSION,
        "prompt_version": stateful_prompts.PUBLIC_HISTORY_PROMPT_VERSION,
        "implementation_sha256": code_hash(),
        "config": asdict(cfg),
        "agreement_rubric": agreement_manifest(),
        "source": {
            "directory": str(Path(design["sources"]["route_manifest"]).parent),
            "plan_file": str(plan_file),
            "plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
            "design_sha256": design["design_sha256"],
            "contexts_sha256": digest(contexts),
        },
        "design": {
            **design["contract"],
            "fresh_initial_answers_and_debates": True,
            "explicit_position_feedback": False,
            "debate_fields": ["reply", "agreement", "choice", "position"],
            "public_history_only": True,
            "global_tone_arms": False,
            "E_length_bounds": LENGTH,
            "E_judge_orders": 2,
            "E_calibration_cases": sum(p["calibration_arm"] is not None for p in design["plans"]),
            "comparison_scope": "Three-member extension, not an isolated effect of committee size versus the four-turn dyads",
        },
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
            "output_limit_retry_policy": OUTPUT_LIMIT_RETRY_POLICY,
            "max_inflight_requests": 64,
            "question_workers": 8,
        },
        "plans": design["plans"],
    }
    counts = Counter()
    for plan in manifest["plans"]:
        counts.update(t.purpose for t in TriadicGraph(contexts[plan["question_id"]], plan, manifest).tasks.values())
    manifest["planned_counts"] = {
        "questions": len(contexts),
        "branch_paths": sum(len(route_for(p).leaves) * len(ARMS) for p in design["plans"]),
        "primary_quality_comparisons": len(contexts) * len(ARMS),
        "by_purpose": dict(counts),
        "logical_calls": sum(counts.values()),
        "formal_replies": counts["debate"],
        "initial_answers": counts["initial"],
        "local_C_calls": 0,
        "D_calls": 0,
    }
    return json.loads(canonical(manifest)), contexts


class TriadicGraph(QuestionGraph):
    def __init__(self, context, plan, manifest):
        if manifest["protocol_version"] != VERSION or manifest["design"]["explicit_position_feedback"]:
            raise ValueError("Not a public-history A/B/E protocol")
        self.context, self.plan, self.manifest = context, plan, manifest
        self.arms = tuple(manifest["design"].get("active_arms", ARMS))
        if not self.arms or len(set(self.arms)) != len(self.arms) or not set(self.arms) <= set(ARMS):
            raise ValueError("Invalid active tone assignments")
        self.question = Question.from_dict(context["question"])
        self.config = PilotConfig(**manifest["config"])
        self.config.validate()
        self.route = route_for(plan)
        for n in self.route.nodes:
            a, b = (plan["tones"][arm][n.id] for arm in ARMS)
            if a not in TONES or b not in TONES or ((a == b) != (n.depth <= 2)):
                raise ValueError("Invalid paired turn-local tones")
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted = set()
        self._build()
        reached = set()
        while more := {t.key for t in self.tasks.values() if t.dependencies <= reached} - reached:
            reached.update(more)
        if reached != self.tasks.keys():
            raise ValueError("Invalid triadic dependencies")

    def key(self, arm, suffix):
        shared = suffix.startswith("initial/") or suffix == "E2/baseline"
        if suffix.startswith(("debate/", "B/")):
            shared = self.route.get(suffix.split("/", 1)[1]).depth <= 2
        elif suffix.startswith("E1/"):
            shared = self.route.get(suffix.rsplit("/", 1)[1]).depth <= 2
        return f"{self.question.id}/{VERSION}/{self.config.roster}/{'shared' if shared else arm}/{suffix}"

    def initial(self, values):
        return {
            m: values[self.key("original", f"initial/{m}")]
            for m in range(3)
            if self.key("original", f"initial/{m}") in values
        }

    def history(self, values, arm):
        return {
            n.id: values[self.key(arm, f"debate/{n.id}")]
            for n in self.route.nodes
            if self.key(arm, f"debate/{n.id}") in values
        }

    def endpoint_node(self, leaf, member):
        return next((n for n in reversed(self.route.path(leaf)) if n.receiver == member), None)

    def synthesis_messages(self, values, arm):
        initial = self.initial(values)
        history = {} if arm == "baseline" else self.history(values, arm)
        nodes = (
            []
            if arm == "baseline"
            else [
                {
                    "id": n.id,
                    "member": n.receiver,
                    "reply_to": n.parent or f"initial/{n.sender}",
                    "text": history[n.id]["reply"],
                }
                for n in self.route.nodes
            ]
        )
        return (
            Message(
                "developer",
                "Produce a considered answer to the original question using the supplied committee material. Weigh the reasons, acknowledge material disagreement, and do not treat repeated claims as independent evidence. The discussion is a tree: follow reply_to links; siblings are alternate branches, not consecutive turns. Shared messages appear once. Do not invent evidence. "
                + length_contract(LENGTH),
            ),
            Message("user", prompts.question_block(self.question)),
            Message(
                "user",
                canonical(
                    {
                        "initial_answers": [
                            {"id": f"initial/{m}", "member": m, "text": initial[m]["position"]} for m in range(3)
                        ],
                        "discussion": nodes,
                    }
                ),
            ),
        )

    def _build(self):
        q, cfg = self.question, self.config

        def add(arm, suffix, deps, purpose, model, messages, schema, *, effort="low", tokens=None, parse=None):
            key = self.key(arm, suffix)
            dependencies = frozenset(deps)
            if key in self.tasks:
                if self.tasks[key].dependencies != dependencies or self.tasks[key].purpose != purpose:
                    raise ValueError("Conflicting shared task")
                return
            self.tasks[key] = Task(
                key,
                dependencies,
                lambda v, count: Request(
                    key, purpose, model, messages(v), effort, tokens or cfg.judge_tokens, schema, cfg.cache
                ),
                parse or (lambda text: prompts.parse_json(text, schema)),
                purpose,
                model,
            )

        initial_keys = [self.key("original", f"initial/{m}") for m in range(3)]
        for m, model in enumerate(cfg.members):
            add(
                "original",
                f"initial/{m}",
                [],
                "initial",
                model,
                lambda v, m=m: stateful_prompts.initial_messages(q, m),
                stateful_prompts.position_schema(q),
                effort=cfg.debate_effort,
                tokens=cfg.initial_tokens,
            )
        for arm in self.arms:
            for n in self.route.nodes:
                path = self.route.path(n.id)
                deps = {self.key(arm, f"initial/{m}") for m in (n.receiver, path[0].sender)} | {
                    self.key(arm, f"debate/{p.id}") for p in path[:-1]
                }
                add(
                    arm,
                    f"debate/{n.id}",
                    deps,
                    "debate",
                    cfg.members[n.receiver],
                    lambda v, a=arm, n=n: stateful_prompts.debate_messages(
                        q,
                        self.plan["tones"][a][n.id],
                        self.initial(v),
                        self.route,
                        self.history(v, a),
                        n.id,
                        explicit_position=False,
                    ),
                    stateful_prompts.reply_schema(q),
                    effort=cfg.debate_effort,
                    tokens=cfg.debate_tokens,
                )
            for nid in self.plan["sampled_b_nodes"]:
                n = self.route.get(nid)
                add(
                    arm,
                    f"B/{nid}",
                    [self.key(arm, f"debate/{nid}")],
                    "judge_b",
                    cfg.judge_model,
                    lambda v, a=arm, n=n: strong_prompts.b_messages(
                        q, self.history(v, a)[n.parent]["reply"], self.history(v, a)[n.id]["reply"]
                    ),
                    strong_prompts.B_SCHEMA,
                )
            for leaf in self.plan["sampled_endpoint_leaves"]:
                for m in range(3):
                    n = self.endpoint_node(leaf, m)
                    if n is None:
                        continue
                    before, after = self.key(arm, f"initial/{m}"), self.key(arm, f"debate/{n.id}")
                    add(
                        arm,
                        f"E1/{m}/{n.id}",
                        [before, after],
                        "judge_e1",
                        cfg.judge_model,
                        lambda v, b=before, a=after: prompts.c_messages(q, v[b]["position"], v[a]["position"]),
                        prompts.C_SCHEMA,
                    )
        for arm in ("baseline", *self.arms):
            deps = initial_keys + (
                [] if arm == "baseline" else [self.key(arm, f"debate/{n.id}") for n in self.route.nodes]
            )
            add(
                arm,
                f"E2/{arm}",
                deps,
                "synthesis",
                cfg.chairman_model,
                lambda v, a=arm: self.synthesis_messages(v, a),
                ANSWER_SCHEMA,
                effort=cfg.chairman_effort,
                tokens=cfg.chairman_tokens,
                parse=parse_answer,
            )
        for arm in self.arms:
            baseline, debate = self.key("baseline", "E2/baseline"), self.key(arm, f"E2/{arm}")
            for order in range(2):
                left = self.plan["first_debate_left"][arm] if order == 0 else not self.plan["first_debate_left"][arm]
                add(
                    arm,
                    f"E2/order{order}",
                    [baseline, debate],
                    "judge_e",
                    cfg.judge_model,
                    lambda v, b=baseline, d=debate, left=left: prompts.e_messages(
                        q, *((v[d]["answer"], v[b]["answer"]) if left else (v[b]["answer"], v[d]["answer"]))
                    ),
                    prompts.E_SCHEMA,
                )
        arm = self.plan["calibration_arm"]
        if arm in self.arms:
            source = self.key(arm, f"E2/{arm}")
            for kind in CALIBRATION_KINDS:
                variant = self.key(arm, f"calibration/{kind}/variant")
                bounds = VERBOSE_LENGTH if kind == "verbosity" else LENGTH
                add(
                    arm,
                    f"calibration/{kind}/variant",
                    [source],
                    "synthesis",
                    cfg.chairman_model,
                    lambda v, k=kind, s=source: variant_messages(q, v[s]["answer"], k),
                    ANSWER_SCHEMA,
                    effort=cfg.chairman_effort,
                    tokens=cfg.chairman_tokens,
                    parse=lambda text, b=bounds: parse_answer(text, b),
                )
                first_left = bool(rng_for(SEED, q.id, kind, "calibration_order").randrange(2))
                for order, left in enumerate((first_left, not first_left)):
                    add(
                        arm,
                        f"calibration/{kind}/order{order}",
                        [source, variant],
                        "judge_e",
                        cfg.judge_model,
                        lambda v, s=source, t=variant, left=left: prompts.e_messages(
                            q, *((v[s]["answer"], v[t]["answer"]) if left else (v[t]["answer"], v[s]["answer"]))
                        ),
                        prompts.E_SCHEMA,
                    )
            variants = {k: self.key(arm, f"calibration/{k}/variant") for k in CALIBRATION_KINDS}
            add(
                arm,
                "calibration/validation",
                [source, *variants.values()],
                "calibration_validation",
                cfg.judge_model,
                lambda v: validation_messages(
                    q, v[source]["answer"], {k: v[key]["answer"] for k, key in variants.items()}
                ),
                VALIDATION_SCHEMA,
            )

    def preferences(self, arm, prefix, first_left):
        orders = [
            {
                "target_side": "left" if left else "right",
                "judgment": self.values.get(self.key(arm, f"{prefix}/order{i}")),
            }
            for i, left in enumerate((first_left, not first_left))
        ]
        votes = [int(o["judgment"]["preference"] == o["target_side"]) for o in orders if o["judgment"]]
        return {"orders": orders, "target_score": sum(votes) / 2 if len(votes) == 2 else None}

    def report(self, count):
        branches, trajectories, positions, endpoints, b_events = {}, [], {}, [], []
        seen_b = set()
        for arm in self.arms:
            history = self.history(self.values, arm)
            branches[arm] = {"replies": history, "tones": self.plan["tones"][arm]}
            for nid in self.plan["sampled_b_nodes"]:
                key = self.key(arm, f"B/{nid}")
                if key not in seen_b:
                    b_events.append(
                        {
                            "assignment": arm,
                            "node_id": nid,
                            "request_key": key,
                            "self_label": history.get(nid, {}).get("agreement"),
                            "judgment": self.values.get(key),
                        }
                    )
                    seen_b.add(key)
            for leaf in self.route.leaves:
                path = self.route.path(leaf.id)
                keys = {self.key(arm, f"debate/{n.id}") for n in path}
                status = (
                    "success"
                    if keys <= self.values.keys()
                    else "failed" if keys & (self.failed.keys() | self.blocked.keys()) else "pending"
                )
                trajectories.append(
                    {
                        "assignment": arm,
                        "leaf": leaf.id,
                        "status": status,
                        "member_sequence": [path[0].sender, *[n.receiver for n in path]],
                        "tone_sequence": [self.plan["tones"][arm][n.id] for n in path],
                    }
                )
                states = {}
                for m in range(3):
                    initial = self.values.get(self.key(arm, f"initial/{m}"))
                    rows = [{**initial, "T": 0, "participation_index": 0, "node_id": None}] if initial else []
                    for n in path:
                        if n.receiver == m and n.id in history:
                            rows.append(
                                {
                                    "choice": history[n.id]["choice"],
                                    "position": history[n.id]["position"],
                                    "node_id": n.id,
                                    "T": n.depth,
                                    "participation_index": self.route.participation(n.id),
                                }
                            )
                    states[str(m)] = rows
                    last = self.endpoint_node(leaf.id, m)
                    after = history.get(last.id) if last else None
                    sampled = leaf.id in self.plan["sampled_endpoint_leaves"]
                    comparable = (
                        status == "success"
                        and initial is not None
                        and after is not None
                        and initial["choice"] is not None
                        and after["choice"] is not None
                    )
                    endpoints.append(
                        {
                            "assignment": arm,
                            "leaf": leaf.id,
                            "member": m,
                            "participated": last is not None,
                            "path_status": status,
                            "last_node": last.id if last else None,
                            "initial": initial,
                            "final": after,
                            "choice_changed": initial["choice"] != after["choice"] if comparable else None,
                            "text_sampled": sampled and last is not None,
                            "text_judgment": (
                                self.values.get(self.key(arm, f"E1/{m}/{last.id}"))
                                if sampled and last and status == "success"
                                else None
                            ),
                        }
                    )
                positions[f"{arm}/{leaf.id}"] = states
        quality = []
        for arm in self.arms:
            base = self.values.get(self.key("baseline", "E2/baseline"))
            debated = self.values.get(self.key(arm, f"E2/{arm}"))
            pref = self.preferences(arm, "E2", self.plan["first_debate_left"][arm])
            quality.append(
                {
                    "assignment": arm,
                    "baseline": base,
                    "debated": debated,
                    "baseline_words": words(base["answer"]) if base else None,
                    "debate_words": words(debated["answer"]) if debated else None,
                    "preferences": pref,
                    "debate_score": pref["target_score"] if base and debated else None,
                    "status": (
                        "success"
                        if base and debated and pref["target_score"] is not None
                        else "failed" if self.complete else "pending"
                    ),
                }
            )
        calibration = None
        if (arm := self.plan["calibration_arm"]) in self.arms:
            calibration = {
                "arm": arm,
                "source": self.values.get(self.key(arm, f"E2/{arm}")),
                "validation": self.values.get(self.key(arm, "calibration/validation")),
                "variants": {},
            }
            for kind in CALIBRATION_KINDS:
                first_left = bool(rng_for(SEED, self.question.id, kind, "calibration_order").randrange(2))
                calibration["variants"][kind] = {
                    "answer": self.values.get(self.key(arm, f"calibration/{kind}/variant")),
                    **self.preferences(arm, f"calibration/{kind}", first_left),
                }
        return {
            "question_id": self.question.id,
            "question": self.context["question"],
            "status": "completed" if self.successful else "completed_with_failures" if self.complete else "in_progress",
            "initial_positions": self.initial(self.values),
            "route": self.plan["route"],
            "branches": branches,
            "trajectories": trajectories,
            "positions": positions,
            "B": b_events,
            "E1": endpoints,
            "E2": quality,
            "calibration": calibration,
            "failed_tasks": self.failed,
            "blocked_tasks": self.blocked,
            "request_keys": sorted(self.tasks),
        }


class TriadicMockProvider(StatefulMockProvider):
    def generate(self, request):
        if request.purpose == "judge_e1":
            from dataclasses import replace

            return self.base.generate(replace(request, purpose="judge_c"))
        if request.purpose in ("synthesis", "calibration_validation", "judge_e"):
            return QualityMockProvider().generate(request)
        return super().generate(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-plan", required=True, type=Path)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    design = make_design(args.question_plan, args.route_manifest)
    if args.output.exists():
        if json.loads(args.output.read_text()) != design:
            raise ValueError("Refusing to overwrite a different frozen plan")
    else:
        from .study import atomic_json

        atomic_json(args.output, design)
    print(args.output.resolve())
