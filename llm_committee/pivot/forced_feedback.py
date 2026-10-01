"""Frozen, paired strong-disagreement interventions on archived trajectory prefixes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import prompts
from .agreement import agreement_manifest
from .failures import FORMAT_RETRY_POLICY
from .models import OPEN_MODELS, TONES, Message, PilotConfig, Question, Request, canonical, digest
from .planning import Node, Route, rng_for, route_from_plan
from .probabilities import RATING_LABELS, TOPK_ZERO_FILL_POLICY, choice_change, distribution, own_position_difference
from .taskgraph import QuestionGraph, Task

VERSION = "forced-feedback-2026-09-26-v1"
ARMS = ("natural", "forced")
FORCE_INSTRUCTION = (
    "For this reply, take the opposing side to the central conclusion of the immediately preceding "
    "contribution. This assigned stance overrides agreement-seeking instructions, while retaining the "
    "assigned delivery tone. Clearly identify the conclusion you reject and defend a substantively "
    "opposing position with concrete reasons. Do not merely criticize wording or add minor qualifications. "
    "Do not invent facts or citations. Use the usual response format and length limit, and report the "
    "agreement label that best describes your actual reply."
)


def implementation_hash():
    names = (
        "forced_feedback.py",
        "forced_feedback_run.py",
        "prompts.py",
        "models.py",
        "agreement.py",
        "planning.py",
        "taskgraph.py",
        "scheduling.py",
        "storage.py",
        "providers.py",
        "failures.py",
        "probabilities.py",
        "tinker_provider.py",
        "databricks_provider.py",
    )
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def select_cases(source_manifest, seed=20260926):
    """Use route metadata only, never replies, agreement labels or observed effects.

    Each question contributes two distinct recipients; a random half contribute
    the third. Exactly 50 cases per recipient and tone, with one source leaf and
    one cut per question/recipient. Cut depth is uniform over eligible 2–4 nodes
    on the sampled source leaf; it is not forced to match the archived next hop.
    """
    plans = sorted(source_manifest["plans"], key=lambda p: p["question_id"])
    if len(plans) != 60:
        raise ValueError("This supplementary protocol requires the frozen 60-question source")
    rng = rng_for(seed, VERSION, "sampling")
    omitted = list(range(3)) * 20
    rng.shuffle(omitted)
    assignments = {p["question_id"]: [a for a in range(3) if a != omit] for p, omit in zip(plans, omitted, strict=True)}
    for recipient in range(3):
        candidates = [p["question_id"] for p, omit in zip(plans, omitted, strict=True) if omit == recipient]
        for qid in rng.sample(candidates, 10):
            assignments[qid].append(recipient)
    planned = {p["question_id"]: p for p in plans}
    used_leaves, used_cuts, cases = set(), set(), []
    for recipient in range(3):
        qids = [qid for qid in assignments if recipient in assignments[qid]]
        rng.shuffle(qids)
        tones = [tone for i, tone in enumerate(TONES) for _ in range(16 if i == recipient else 17)]
        rng.shuffle(tones)
        for qid, tone in zip(qids, tones, strict=True):
            route = route_from_plan(planned[qid])
            candidates = {}
            for leaf in route.leaves:
                if (qid, tone, leaf.id) in used_leaves:
                    continue
                cuts = [
                    n
                    for n in route.path(leaf.id)
                    if n.depth in (2, 3, 4) and n.sender == recipient and (qid, tone, n.id) not in used_cuts
                ]
                if cuts:
                    candidates[leaf.id] = cuts
            if not candidates:
                raise ValueError(f"No eligible unused prefix for {qid}/{tone}/{recipient}; do not select by outcome")
            leaf = rng.choice(sorted(candidates))
            node = rng.choice(candidates[leaf])
            used_leaves.add((qid, tone, leaf))
            used_cuts.add((qid, tone, node.id))
            cases.append(
                {
                    "question_id": qid,
                    "tone": tone,
                    "source_leaf": leaf,
                    "cut_node": node.id,
                    "anchor_node": node.parent,
                    "recipient": recipient,
                    "challenger": node.receiver,
                    "feedback_T": node.depth,
                    "receiver_T": node.depth + 1,
                    "recipient_participation": sum(n.receiver == recipient for n in route.path(node.id)[:-1]) + 1,
                }
            )
    rng.shuffle(cases)
    # First six form a prespecified engineering gate, covering all ordered model pairs.
    first = []
    for recipient in range(3):
        for challenger in range(3):
            if recipient == challenger:
                continue
            index = next(
                i for i, c in enumerate(cases) if c["recipient"] == recipient and c["challenger"] == challenger
            )
            first.append(cases.pop(index))
    cases = [dict(c, id=f"case-{i:03d}") for i, c in enumerate(first + cases, 1)]
    assert Counter(c["recipient"] for c in cases) == {0: 50, 1: 50, 2: 50}
    assert Counter(c["tone"] for c in cases) == dict.fromkeys(TONES, 50)
    assert set(Counter(c["question_id"] for c in cases).values()) == {2, 3}
    assert len(used_leaves) == len(used_cuts) == len(cases) == 150
    return cases


def prepare(source: Path, *, mock=False, seed=20260926):
    source = source.resolve()
    raw_manifest = (source / "manifest.json").read_bytes()
    old = json.loads(raw_manifest)
    report = json.loads((source / "report.json").read_text())
    if old.get("prompt_version") != prompts.PROMPT_VERSION or old.get("agreement_rubric") != agreement_manifest():
        raise ValueError("Expected the completed leaning-label source, not legacy data")
    if old["kind"] != "paid_main_study" or report["status"] != "completed":
        raise ValueError("Source must be the completed real main study")
    config = PilotConfig(**old["config"])
    config.validate()
    if config.roster != "mixed_family" or config.judge_model != "gemini-3.8-flash":
        raise ValueError("The source roster and C judge must remain unchanged")
    cases = select_cases(old, seed)
    questions = {q["id"]: q for q in old["questions"]}
    contexts, hashes = {}, {}
    for plan in old["plans"]:
        qid = plan["question_id"]
        raw = (source / "questions" / f"{qid}.json").read_bytes()
        hashes[qid] = hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        if record["status"] != "completed" or len(record["questions"]) != 1:
            raise ValueError("Source question is not complete")
        q = record["questions"][0]
        if q["question_id"] != qid or q["question_fingerprint"] != plan["question_fingerprint"]:
            raise ValueError("Source question identity mismatch")
        contexts[qid] = {
            "question": questions[qid],
            "plan": plan,
            "initial_answers": q["initial_answers"],
            "formal_replies": q["formal_replies"],
        }
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_forced_feedback_supplement",
        "protocol_version": VERSION,
        "base_prompt_version": prompts.PROMPT_VERSION,
        "implementation_sha256": implementation_hash(),
        "force_instruction": FORCE_INSTRUCTION,
        "agreement_rubric": agreement_manifest(),
        "config": asdict(config),
        "probability_readout": dict(TOPK_ZERO_FILL_POLICY),
        "source": {
            "directory": str(source),
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "question_sha256": hashes,
            "contexts_sha256": digest(contexts),
        },
        "sampling": {
            "seed": seed,
            "cases": 150,
            "questions": 60,
            "cut_depths": [2, 3, 4],
            "per_recipient": 50,
            "per_tone": 50,
            "question_cases": "2 or 3",
            "rule": "Random routes and cuts only; one source trajectory per case; shared cuts deduplicated",
            "independence": "Repeated cases from one question are not independent questions",
            "preflight_case_ids": [c["id"] for c in cases[:6]],
        },
        "design": {
            "arms": list(ARMS),
            "natural_feedback": "Archived unforced reply at the exact cut; no outcome-based selection",
            "forced_feedback": "New reply with only the challenger privately assigned an opposing stance",
            "return": "Fresh recipient reply in BOTH arms; fixed return to the immediately preceding speaker",
            "baseline": "One fresh pre-feedback own-position read, shared across arms; never added to debate history",
            "measurements": "Recipient self-label; C option/text; open-model D choice and own-text-versus-filler",
            "grouping": "Recipient's new self-label, not the forced challenger's label; retain T and member",
            "conditioning": "Self-label subgroups are post-treatment descriptions, not randomized causal strata",
            "excluded": ["human feedback", "B external judge", "E synthesis", "continuing old future branches"],
            "selection": "Retain all format-valid outputs regardless of label or observed change",
        },
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "spend_cap_usd": None,
            "failure_policy": FORMAT_RETRY_POLICY,
            "max_inflight_requests": 64,
            "question_workers": 8,
        },
        "cases": cases,
        "planned_counts": {
            "cases": 150,
            "recipient_responses": 300,
            "forced_feedback": 150,
            "archived_natural_feedback": 150,
            "C_readings": 450,
            "C_judgments": 300,
            "D_cases": 100,
            "D_choice_readings": 300,
            "D_text_pairs": 200,
            "logical_calls": 1600,
        },
    }
    return json.loads(canonical(manifest)), contexts


def case_route(context, case):
    source = route_from_plan(context["plan"])
    cut = source.get(case["cut_node"])
    if cut.parent != case["anchor_node"] or cut.sender != case["recipient"] or cut.receiver != case["challenger"]:
        raise ValueError("Cut metadata does not match the source route")
    returned = Node("feedback_return", cut.id, cut.receiver, cut.sender, cut.depth + 1)
    # Deliberately a new truncated route: never call the main study's six-leaf shape validator.
    return Route((*source.path(cut.id), returned))


class FeedbackGraph(QuestionGraph):
    """Independent supplementary cases grouped by question for the existing ready-node scheduler."""

    def __init__(self, context, cases, manifest):
        self.question = Question.from_dict(context["question"])
        self.config = PilotConfig(**manifest["config"])
        self.context, self.cases, self.manifest = context, cases, manifest
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted = set()
        self.initial = {int(k): v for k, v in context["initial_answers"].items()}
        for case in cases:
            self._add_case(case)

    def key(self, case, suffix):
        return f"{self.question.id}/{VERSION}/{case['id']}/{suffix}"

    def history(self, case, values, arm, *, include_return=False):
        route = case_route(self.context, case)
        source = self.context["formal_replies"][case["tone"]]
        history = {n.id: source[n.id] for n in route.path(case["cut_node"])[:-1]}
        if arm == "natural":
            history[case["cut_node"]] = source[case["cut_node"]]
        elif arm == "forced":
            history[case["cut_node"]] = values[self.key(case, "forced/feedback")]
        elif arm != "prefix":
            raise ValueError("Unknown feedback arm")
        if include_return:
            history["feedback_return"] = values[self.key(case, f"{arm}/reply")]
        return history

    def _add_case(self, case):
        question, config = self.question, self.config
        route = case_route(self.context, case)
        model = config.members[case["recipient"]]
        labels = question.labels if model in OPEN_MODELS else ()

        def add(suffix, deps, purpose, target, builder, effort, tokens, schema=None, labels=(), parse=None):
            key = self.key(case, suffix)
            self.tasks[key] = Task(
                key,
                frozenset(self.key(case, d) for d in deps),
                lambda values, count: Request(
                    key,
                    purpose,
                    target,
                    builder(values, count),
                    effort,
                    tokens,
                    schema,
                    config.cache and purpose in ("debate", "position"),
                    candidate_labels=labels,
                ),
                parse or (lambda text: prompts.parse_json(text, schema)),
                purpose,
                target,
            )

        before = {"tone": case["tone"], "member": case["recipient"], "node_id": case["anchor_node"]}
        after = {**before, "node_id": "feedback_return"}

        add(
            "before",
            (),
            "position",
            model,
            lambda v, count: prompts.position_messages(
                question, before, self.initial, route, self.history(case, v, "prefix")
            ),
            "none",
            config.position_tokens,
            labels=labels,
            parse=lambda text: prompts.parse_position(text, question),
        )

        def forced_messages(values, count):
            messages = prompts.debate_messages(
                question, case["tone"], self.initial, route, self.history(case, values, "prefix"), case["cut_node"]
            )
            return (Message("developer", messages[0].text + "\n" + FORCE_INSTRUCTION), *messages[1:])

        add(
            "forced/feedback",
            (),
            "debate",
            config.members[case["challenger"]],
            forced_messages,
            config.debate_effort,
            config.debate_tokens,
            prompts.REPLY_SCHEMA,
        )
        for arm in ARMS:
            feedback_deps = ["forced/feedback"] if arm == "forced" else []
            add(
                f"{arm}/reply",
                feedback_deps,
                "debate",
                model,
                lambda v, count, a=arm: prompts.debate_messages(
                    question, case["tone"], self.initial, route, self.history(case, v, a), "feedback_return"
                ),
                config.debate_effort,
                config.debate_tokens,
                prompts.REPLY_SCHEMA,
            )
            add(
                f"{arm}/position",
                [f"{arm}/reply"],
                "position",
                model,
                lambda v, count, a=arm: prompts.position_messages(
                    question, after, self.initial, route, self.history(case, v, a, include_return=True)
                ),
                "none",
                config.position_tokens,
                labels=labels,
                parse=lambda text: prompts.parse_position(text, question),
            )
            add(
                f"{arm}/judge_c",
                ["before", f"{arm}/position"],
                "judge_c",
                config.judge_model,
                lambda v, count, a=arm: prompts.c_messages(
                    question, v[self.key(case, "before")]["position"], v[self.key(case, f"{a}/position")]["position"]
                ),
                "low",
                config.judge_tokens,
                prompts.C_SCHEMA,
            )
            if labels:
                for treatment in ("argument", "control"):

                    def d_messages(values, count, a=arm, t=treatment):
                        history = self.history(case, values, a)
                        incoming = history[case["cut_node"]]["reply"]
                        if t == "control":
                            repeats = max(1, round(count(model, incoming) / count(model, prompts.FILLER_SENTENCE)))
                            incoming = " ".join([prompts.FILLER_SENTENCE] * repeats)
                        fixed = values[self.key(case, "before")]["position"]
                        return prompts.d_text_messages(question, after, self.initial, route, history, fixed, incoming)

                    add(
                        f"{arm}/d_text/{treatment}",
                        ["before", *feedback_deps],
                        "d_text",
                        model,
                        d_messages,
                        "none",
                        config.probability_tokens,
                        labels=RATING_LABELS,
                        parse=prompts.parse_rating,
                    )

    def case_report(self, case, count):
        def get(suffix):
            return self.values.get(self.key(case, suffix))

        def position(value):
            return {k: v for k, v in value.items() if k != "_readout"} if value else None

        def dist(value, labels):
            return (
                distribution(value["_readout"], labels, missing_as_zero=True) if value and "_readout" in value else None
            )

        before = get("before")
        before_dist = dist(before, self.question.labels)
        arms = {}
        for arm in ARMS:
            peer = (
                self.context["formal_replies"][case["tone"]][case["cut_node"]]
                if arm == "natural"
                else get("forced/feedback")
            )
            reply, after = get(f"{arm}/reply"), get(f"{arm}/position")
            after_dist = dist(after, self.question.labels)
            choice_changed = (
                before["choice"] != after["choice"]
                if before and after and before["choice"] is not None and after["choice"] is not None
                else None
            )
            rating = {t: dist(get(f"{arm}/d_text/{t}"), RATING_LABELS) for t in ("argument", "control")}
            d_text = None
            if all(rating.values()):
                text = peer["reply"]
                model = self.config.members[case["recipient"]]
                repeats = max(1, round(count(model, text) / count(model, prompts.FILLER_SENTENCE)))
                d_text = {
                    **rating,
                    "fixed_full_position": before["position"],
                    "filler_repetitions": repeats,
                    "argument_tokens": count(model, text),
                    "filler_tokens": count(model, " ".join([prompts.FILLER_SENTENCE] * repeats)),
                    "mean_own_agreement_argument_minus_control": own_position_difference(
                        rating["argument"], rating["control"]
                    ),
                }
            arms[arm] = {
                "feedback": peer,
                "receiver_reply": reply,
                "receiver_self_label": reply["agreement"] if reply else None,
                "position_after": position(after),
                "C_choice_changed": choice_changed,
                "C_text": get(f"{arm}/judge_c"),
                "D_choice_after": after_dist,
                "D_choice_pp": choice_change(before_dist, after_dist, before["choice"])
                if before_dist and after_dist
                else None,
                "D_text": d_text,
            }
        keys = {k for k in self.tasks if k.startswith(self.key(case, ""))}
        failures = {k: v for k, v in self.failed.items() if k in keys}
        blocked = {k: v for k, v in self.blocked.items() if k in keys}
        done = sum(k in self.values or k in failures or k in blocked for k in keys)
        return {
            "case": case,
            "question": asdict(self.question),
            "status": "completed"
            if keys <= self.values.keys()
            else "completed_with_failures"
            if done == len(keys)
            else "partial",
            "position_before": position(before),
            "D_choice_before": before_dist,
            "arms": arms,
            "failed_tasks": failures,
            "blocked_tasks": blocked,
            "request_keys": sorted(keys),
        }
