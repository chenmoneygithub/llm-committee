"""Fresh strongly/leaning dyadic experiment, with one shared prefix per paired path."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import prompts, strong_prompts
from .dyadic import make_route
from .failures import FORMAT_RETRY_POLICY, OUTPUT_LIMIT_RETRY_POLICY
from .models import OPEN_MODELS, Completion, PilotConfig, Question, Request, canonical, digest
from .probabilities import RATING_LABELS, TOPK_ZERO_FILL_POLICY
from .providers import MockProvider
from .strong_agreement import AGREEMENT, agreement_manifest
from .taskgraph import QuestionGraph, Task
from .turn_tone import STREAM, TurnToneGraph
from .turn_tone_fork import reusable_suffixes

VERSION = "strong-dyadic-paired-2026-09-26-v1"
QUESTION_COUNT = 20
ARMS = ("original", "alternate")


def branch_plan(item, arm):
    route = make_route()
    schedule = item[f"{arm}_tones"]
    readings, pairs, events = {}, {}, []

    def read(member, node):
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

    def compare(before, after, kind):
        pid = f"{before}->{after}"
        pair = pairs.setdefault(pid, {"id": pid, "before": before, "after": after, "kinds": []})
        if kind not in pair["kinds"]:
            pair["kinds"].append(kind)
        return pid

    for nid in item["sampled_node_ids"]:
        node = route.get(nid)
        before = read(node.receiver, route.previous_own(nid))
        initial = read(node.receiver, None)
        after = read(node.receiver, node)
        events.append(
            {
                "id": f"{STREAM}/{nid}",
                "pair": nid.split("-")[0],
                "tone": STREAM,
                "current_tone": schedule[nid],
                "previous_tone": schedule[node.parent],
                "probe_tone": "neutral",
                "node_id": nid,
                "member": node.receiver,
                "peer": node.sender,
                "T": node.depth,
                "participation_index": route.participation(nid),
                "previous_reading": before,
                "initial_reading": initial,
                "current_reading": after,
                "adjacent_pair": compare(before, after, "adjacent"),
                "final_pair": compare(initial, after, "initial_to_endpoint") if node.depth >= 3 else None,
            }
        )
    return {
        "question_id": item["question"]["id"],
        "tone_schedule": schedule,
        "events": events,
        "readings": list(readings.values()),
        "text_pairs": list(pairs.values()),
    }


def implementation_hash():
    names = (
        "strong_study.py",
        "strong_run.py",
        "strong_prompts.py",
        "strong_agreement.py",
        "prompts.py",
        "models.py",
        "agreement.py",
        "dyadic.py",
        "turn_tone.py",
        "turn_tone_fork.py",
        "taskgraph.py",
        "planning.py",
        "scheduling.py",
        "failures.py",
        "storage.py",
        "providers.py",
        "probabilities.py",
        "tinker_provider.py",
        "databricks_provider.py",
    )
    return digest({n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def prepare(source, *, mock=False):
    source = Path(source).resolve()
    raw = source.read_bytes()
    template = json.loads(raw)
    body = {k: v for k, v in template.items() if k != "design_sha256"}
    question_count = template["scope"]["question_count"]
    if digest(body) != template["design_sha256"] or len(template["plans"]) != question_count or question_count < 1:
        raise ValueError("Frozen trajectory template changed")
    route = make_route()
    if template["route_template"] != json.loads(canonical([asdict(n) for n in route.nodes])):
        raise ValueError("Routing changed")
    config = PilotConfig(roster="mixed_family", **template["shared_config"])
    config.validate()
    if config.judge_model != "gemini-3.8-flash":
        raise ValueError("Do not change the judge")
    plans, contexts = [], {}
    for item in template["plans"]:
        q = Question.from_dict(item["question"])
        if len(item["sampled_node_ids"]) != 8 or len(set(item["sampled_node_ids"])) != 8:
            raise ValueError("Changed measurement sample")
        for node in route.nodes:
            equal = item["original_tones"][node.id] == item["alternate_tones"][node.id]
            if equal != (node.depth <= 2):
                raise ValueError("Shared-prefix or alternate-tone contract changed")
        plans.append({"question_id": q.id, "arms": {arm: branch_plan(item, arm) for arm in ARMS}})
        contexts[q.id] = {"question": item["question"]}  # Deliberately NO archived model outputs.
    if len(contexts) != question_count:
        raise ValueError("Questions duplicated")
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_strong_dyadic_pilot",
        "protocol_version": VERSION,
        "implementation_sha256": implementation_hash(),
        "prompt_version": strong_prompts.PROMPT_VERSION,
        "agreement_rubric": agreement_manifest(),
        "config": asdict(config),
        "probability_readout": TOPK_ZERO_FILL_POLICY,
        "source": {
            "directory": str(source.parent),
            "plan_file": str(source),
            "plan_sha256": hashlib.sha256(raw).hexdigest(),
            "design_sha256": template["design_sha256"],
            "contexts_sha256": digest(contexts),
        },
        "design": {
            "fresh_initial_answers": True,
            "fresh_debate_both_assignments": True,
            "reuse_old_fully_outputs": False,
            "tone_assignment_unit": "formal debate turn",
            "global_tone_arms": False,
            "shared_T1_T2_within_pair": True,
            "neutral_probes": True,
            "historical_tones_or_self_labels_visible": False,
            "measurement_outputs_enter_debate": False,
            "three_member_run": False,
            "layer_E": False,
            "remaining_rosters_started": False,
            "independent_question_count": question_count,
        },
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
            "output_limit_retry_policy": OUTPUT_LIMIT_RETRY_POLICY,
            "max_inflight_requests": 64,
            "question_workers": 8,
        },
        "plans": plans,
    }
    counts = Counter()
    for plan in plans:
        graph = StrongGraph(contexts[plan["question_id"]], plan, manifest)
        counts.update(t.purpose for t in graph.tasks.values())
    manifest["planned_counts"] = {
        "questions": question_count,
        "paired_paths": question_count * 3,
        "branch_paths": question_count * 6,
        "formal_replies": counts["debate"],
        "initial_answers": counts["initial"],
        "sampled_unique_events": sum(1 + (e["T"] > 2) for p in plans for e in p["arms"]["original"]["events"]),
        "sampled_original_events": question_count * 8,
        "sampled_alternate_suffix_events": sum(e["T"] > 2 for p in plans for e in p["arms"]["original"]["events"]),
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
    }
    return json.loads(canonical(manifest)), json.loads(canonical(contexts))


class StrongMockProvider(MockProvider):
    def generate(self, request):
        if request.purpose not in ("debate", "judge_b"):
            return super().generate(request)
        selected = AGREEMENT[int(digest(request.key)[:8], 16) % 4]
        value = (
            {"reply": f"SYNTHETIC strong reply {request.key}", "agreement": selected}
            if request.purpose == "debate"
            else {"label": selected, "evidence": "SYNTHETIC evidence"}
        )
        return Completion(canonical(value), 100, 40)


class StrongGraph(QuestionGraph):
    def __init__(self, context, plan, manifest):
        self.question = Question.from_dict(context["question"])
        self.config = PilotConfig(**manifest["config"])
        self.context, self.plan, self.manifest = context, plan, manifest
        self.route = make_route()
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted = set()
        self.shared_suffixes = reusable_suffixes(plan["arms"]["original"])
        self.arm_keys = {arm: set() for arm in ARMS}
        self._build()
        reachable = set()
        while more := {t.key for t in self.tasks.values() if t.dependencies <= reachable} - reachable:
            reachable.update(more)
        if reachable != self.tasks.keys():
            raise ValueError("Invalid strong-study dependencies")

    def key(self, arm, suffix):
        if suffix.startswith("initial/"):
            return f"{self.question.id}/{VERSION}/{suffix}"
        owner = "original" if suffix in self.shared_suffixes else arm
        return f"{self.question.id}/{VERSION}/{owner}/{suffix}"

    def initial(self, values):
        return {
            m: values[self.key("original", f"initial/{m}")]["position"]
            for m in range(3)
            if self.key("original", f"initial/{m}") in values
        }

    def history(self, values, arm):
        return {
            n.id: values[self.key(arm, f"{STREAM}/debate/{n.id}")]
            for n in self.route.nodes
            if self.key(arm, f"{STREAM}/debate/{n.id}") in values
        }

    def _build(self):
        question, config, route = self.question, self.config, self.route

        def add(arm, suffix, deps, purpose, model, builder, effort, tokens, schema=None, labels=(), parse=None):
            key = self.key(arm, suffix)
            self.arm_keys[arm].add(key)
            if key in self.tasks:
                if self.tasks[key].dependencies != frozenset(deps) or self.tasks[key].purpose != purpose:
                    raise ValueError("Conflicting shared request")
                return
            self.tasks[key] = Task(
                key,
                frozenset(deps),
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

        for arm in ARMS:
            plan = self.plan["arms"][arm]
            for m, model in enumerate(config.members):
                add(
                    arm,
                    f"initial/{m}",
                    [],
                    "initial",
                    model,
                    lambda v, c, m=m: prompts.initial_messages(question, m),
                    config.debate_effort,
                    config.initial_tokens,
                    prompts.INITIAL_SCHEMA,
                )
            for node in route.nodes:
                initials = {
                    self.key(arm, f"initial/{node.receiver}"),
                    self.key(arm, f"initial/{route.path(node.id)[0].sender}"),
                }
                deps = initials | ({self.key(arm, f"{STREAM}/debate/{node.parent}")} if node.parent else set())
                add(
                    arm,
                    f"{STREAM}/debate/{node.id}",
                    deps,
                    "debate",
                    config.members[node.receiver],
                    lambda v, c, a=arm, n=node: strong_prompts.debate_messages(
                        question,
                        self.plan["arms"][a]["tone_schedule"][n.id],
                        self.initial(v),
                        route,
                        self.history(v, a),
                        n.id,
                    ),
                    config.debate_effort,
                    config.debate_tokens,
                    strong_prompts.REPLY_SCHEMA,
                )
            for reading in plan["readings"]:
                model = config.members[reading["member"]]
                deps = (
                    [self.key(arm, f"{STREAM}/debate/{reading['node_id']}")]
                    if reading["node_id"]
                    else [self.key(arm, f"initial/{reading['member']}")]
                )
                add(
                    arm,
                    f"C/{reading['id']}",
                    deps,
                    "position",
                    model,
                    lambda v, c, a=arm, r=reading: prompts.position_messages(
                        question, {**r, "tone": "neutral"}, self.initial(v), route, self.history(v, a)
                    ),
                    "none",
                    config.position_tokens,
                    labels=question.labels if model in OPEN_MODELS else (),
                    parse=lambda text: prompts.parse_position(text, question),
                )
            for event in plan["events"]:
                node = route.get(event["node_id"])
                model = config.members[node.receiver]
                add(
                    arm,
                    f"B/{event['id']}",
                    [self.key(arm, f"{STREAM}/debate/{node.id}")],
                    "judge_b",
                    config.judge_model,
                    lambda v, c, a=arm, n=node: strong_prompts.b_messages(
                        question, self.history(v, a)[n.parent]["reply"], self.history(v, a)[n.id]["reply"]
                    ),
                    "low",
                    config.judge_tokens,
                    strong_prompts.B_SCHEMA,
                )
                if model not in OPEN_MODELS:
                    continue
                for treatment in ("argument", "control"):

                    def messages(values, count, a=arm, e=event, n=node, t=treatment, m=model):
                        history = self.history(values, a)
                        incoming = history[n.parent]["reply"]
                        if t == "control":
                            repeats = max(1, round(count(m, incoming) / count(m, prompts.FILLER_SENTENCE)))
                            incoming = " ".join([prompts.FILLER_SENTENCE] * repeats)
                        return prompts.d_text_messages(
                            question,
                            {**e, "tone": "neutral"},
                            self.initial(values),
                            route,
                            history,
                            values[self.key(a, f"C/{e['previous_reading']}")]["position"],
                            incoming,
                        )

                    add(
                        arm,
                        f"Dtext/{event['id']}/{treatment}",
                        [
                            self.key(arm, f"C/{event['previous_reading']}"),
                            self.key(arm, f"{STREAM}/debate/{node.parent}"),
                        ],
                        "d_text",
                        model,
                        messages,
                        "none",
                        config.probability_tokens,
                        labels=RATING_LABELS,
                        parse=prompts.parse_rating,
                    )
            for pair in plan["text_pairs"]:
                add(
                    arm,
                    f"Cjudge/{pair['id']}",
                    [self.key(arm, f"C/{pair['before']}"), self.key(arm, f"C/{pair['after']}")],
                    "judge_c",
                    config.judge_model,
                    lambda v, c, a=arm, p=pair: prompts.c_messages(
                        question,
                        v[self.key(a, f"C/{p['before']}")]["position"],
                        v[self.key(a, f"C/{p['after']}")]["position"],
                    ),
                    "low",
                    config.judge_tokens,
                    prompts.C_SCHEMA,
                )

    def report(self, count):
        branches = {arm: BranchView(self, arm).report(count) for arm in ARMS}
        return {
            "question_id": self.question.id,
            "question": asdict(self.question),
            "status": "completed" if self.successful else "completed_with_failures" if self.complete else "partial",
            "initial_answers": {str(m): p for m, p in self.initial(self.values).items()},
            "branches": branches,
            "trajectories": [{**t, "assignment": a} for a, b in branches.items() for t in b["trajectories"]],
            "failed_tasks": self.failed,
            "blocked_tasks": self.blocked,
            "request_keys": sorted(self.tasks),
        }


class BranchView(TurnToneGraph):
    """Read-only adapter for the already validated measurement arithmetic, not generation."""

    def __init__(self, parent, arm):
        self.parent, self.arm = parent, arm
        self.question, self.config, self.route = parent.question, parent.config, parent.route
        self.plan, self.manifest = parent.plan["arms"][arm], parent.manifest
        self.context = {"initial_answers": {str(m): p for m, p in parent.initial(parent.values).items()}}
        self.initial = parent.initial(parent.values)
        for field in ("tasks", "values", "failed", "blocked"):
            setattr(self, field, {k: v for k, v in getattr(parent, field).items() if k in parent.arm_keys[arm]})
        self.submitted = parent.submitted & parent.arm_keys[arm]

    def key(self, suffix):
        return self.parent.key(self.arm, suffix)

    def report(self, count):
        record = super().report(count)
        return {
            **record,
            "assignment": self.arm,
            "events": [
                {**e, "shared_prefix_event": e["T"] == 2, "D_text_shared": bool(e["D_text"]) and e["T"] <= 3}
                for e in record["events"]
            ],
        }
