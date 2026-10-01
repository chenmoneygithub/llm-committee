"""Length-controlled E on the frozen strongly-label, turn-tone dyadic pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import prompts
from .dyadic import PAIRS, make_route
from .failures import FORMAT_RETRY_POLICY, OUTPUT_LIMIT_RETRY_POLICY
from .models import Completion, Message, PilotConfig, Question, Request, canonical, digest
from .planning import rng_for
from .providers import MockProvider
from .stateful_study import PUBLIC_HISTORY_VERSION
from .stateful_study import VERSION as STATEFUL_SOURCE_VERSION
from .strong_study import VERSION as SOURCE_VERSION
from .taskgraph import QuestionGraph, Task

VERSION = "dyadic-length-controlled-E-2026-09-27-v1"
QUESTION_COUNT = 20
SEED = 20260927
ARMS = ("original", "alternate")
LENGTH = (190, 210)
VERBOSE_LENGTH = (290, 310)
CALIBRATION_KINDS = ("verbosity", "moderate_damage", "severe_damage")
ANSWER_SCHEMA = prompts.schema(answer=prompts.STRING)
VALIDATION_SCHEMA = prompts.schema(
    equivalent_content={"type": "string", "enum": ["yes", "no", "uncertain"]},
    moderate_contradiction={"type": "string", "enum": ["yes", "no", "uncertain"]},
    moderate_preserves_conclusion={"type": "string", "enum": ["yes", "no", "uncertain"]},
    severe_off_topic={"type": "string", "enum": ["yes", "no", "uncertain"]},
    equivalence_evidence=prompts.STRING,
    moderate_evidence=prompts.STRING,
    severe_evidence=prompts.STRING,
)


def words(text):
    return len(text.split())


def length_contract(bounds):
    return (
        f"Your answer MUST contain {bounds[0]}–{bounds[1]} whitespace-separated words; "
        f"aim for exactly {(bounds[0] + bounds[1]) // 2}. This is a hard, programmatically checked requirement. "
        "Count the words and revise internally before submitting. Do not print a word count or a heading. "
        "Return a JSON object containing only the field answer."
    )


def parse_answer(text, bounds=LENGTH):
    value = prompts.parse_json(text, ANSWER_SCHEMA)
    n = words(value["answer"])
    if not bounds[0] <= n <= bounds[1]:
        raise ValueError(f"Length contract failed: {n} words; required {bounds[0]}–{bounds[1]}")
    return value


def synthesis_messages(context, pair, arm):
    """Only this dyad's two original positions and this arm's public four replies."""
    q = Question.from_dict(context["question"])
    route = make_route()
    nodes = (
        []
        if arm == "baseline"
        else [
            {
                "id": n.id,
                "member": n.receiver,
                "reply_to": n.parent or f"initial/{n.sender}",
                "text": context["branches"][arm]["replies"][n.id]["reply"],
            }
            for n in route.nodes
            if n.id.startswith(pair + "-")
        ]
    )
    return (
        Message(
            "developer",
            "Produce a considered answer to the original question using the supplied committee material. "
            "Weigh the reasons, acknowledge material disagreement, and do not treat repeated claims as "
            "independent evidence. Follow the reply_to links in the discussion. Do not invent evidence. "
            + length_contract(LENGTH),
        ),
        Message("user", prompts.question_block(q)),
        Message(
            "user",
            canonical(
                {
                    "initial_answers": [
                        {"id": f"initial/{m}", "member": m, "text": context["initial_answers"][str(m)]}
                        for m in PAIRS[pair]
                    ],
                    "discussion": nodes,
                }
            ),
        ),
    )


def variant_messages(question, base, kind):
    instructions = {
        "verbosity": (
            "Rewrite the source answer more expansively while preserving EXACTLY its substantive content: "
            "the same conclusion, reasons, factual claims, scope, exceptions and uncertainty. Use fuller wording "
            "and transitions only. Add no examples, evidence, claims, arguments or qualifications, and remove none. "
            "Do not just repeat whole sentences. This is a wording/verbosity probe, not an improved answer."
        ),
        "moderate_damage": (
            "Construct a subtly damaged answer. Preserve the source's bottom-line position and most of its "
            "reasoning, but make ONE important reason internally contradictory: state a claim and its logical "
            "opposite with the SAME scope and conditions. Keep it fluent and on topic. Do not merely adopt "
            "a different legitimate opinion, and do not mark or announce the deliberate flaw. Do not introduce "
            "other factual claims, reasons or substantial changes."
        ),
        "severe_damage": (
            "Construct a severely damaged answer by ignoring the survey question and instead explaining "
            "why the Moon has phases. Write a fluent, self-contained answer about that unrelated topic. "
            "Do not mention this instruction, the source answer, the substitution, or the survey question."
        ),
    }
    return (
        Message(
            "developer", instructions[kind] + "\n" + length_contract(VERBOSE_LENGTH if kind == "verbosity" else LENGTH)
        ),
        Message("user", canonical({"question": question.text, "source_answer": base})),
    )


def validation_messages(question, base, variants):
    return (
        Message(
            "developer",
            "Audit these constructed evaluation probes, not which opinion you personally prefer. "
            "Compare source and expanded: set equivalent_content to yes ONLY if conclusion, supporting claims, "
            "reasons, evidence, qualifications, scope and uncertainty are all preserved, with no substantive "
            "addition or omission. For edited, check whether it actually contains a clear logical contradiction "
            "under the same scope and whether it preserves the source's bottom-line conclusion. For unrelated, "
            "check whether it fails to address the original question. These properties are not guaranteed by "
            "the construction labels. Support each assessment with short quotations or concrete differences. "
            "Return yes/no/uncertain for the requested properties and the evidence fields as JSON.",
        ),
        Message(
            "user",
            canonical(
                {
                    "question": question.text,
                    "source": base,
                    "expanded": variants["verbosity"],
                    "edited": variants["moderate_damage"],
                    "unrelated": variants["severe_damage"],
                }
            ),
        ),
    )


def implementation_hash():
    names = (
        "quality_study.py",
        "quality_run.py",
        "prompts.py",
        "models.py",
        "dyadic.py",
        "planning.py",
        "taskgraph.py",
        "scheduling.py",
        "storage.py",
        "failures.py",
        "providers.py",
        "databricks_provider.py",
    )
    return digest({n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})


def plan_questions(qids):
    """Freeze E orders/diagnostics from IDs only, before seeing any generated text."""
    qids = sorted(qids)
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("Expected distinct questions")
    rng = rng_for(SEED, VERSION, "orders-and-calibration")
    calibration_qids = rng.sample(qids, min(12, len(qids)))
    calibration = {
        qid: {"pair": tuple(PAIRS)[i % 3], "arm": ARMS[(i // 3) % 2]} for i, qid in enumerate(calibration_qids)
    }
    order_slots = {}
    for pair in PAIRS:
        for arm in ARMS:
            slots = ([True, False] * ((len(qids) + 1) // 2))[: len(qids)]
            rng.shuffle(slots)
            order_slots[pair, arm] = slots
    return [
        {
            "question_id": qid,
            "orders": {
                pair: {arm: [order_slots[pair, arm][i], not order_slots[pair, arm][i]] for arm in ARMS}
                for pair in PAIRS
            },
            "calibration": calibration.get(qid),
        }
        for i, qid in enumerate(qids)
    ]


def prepare(source, *, mock=False):
    source = Path(source).resolve()
    old_manifest = json.loads((source / "manifest.json").read_text())
    old_report = json.loads((source / "report.json").read_text())
    source_kinds = {
        SOURCE_VERSION: "paid_strong_dyadic_pilot",
        STATEFUL_SOURCE_VERSION: "paid_stateful_dyadic_pilot",
        PUBLIC_HISTORY_VERSION: "paid_stateful_dyadic_pilot",
    }
    source_version = old_manifest["protocol_version"]
    if source_version not in source_kinds or old_report["status"] != "completed":
        raise ValueError("E requires a completed supported dyadic pilot")
    if old_manifest["kind"] != source_kinds[source_version] and not (mock and old_manifest["kind"] == "offline_mock"):
        raise ValueError("Wrong source cohort")
    config = PilotConfig(**old_manifest["config"])
    config.validate()
    if config.judge_model != "gemini-3.8-flash":
        raise ValueError("Keep the existing judge")
    contexts, hashes = {}, {}
    qids = sorted(p["question_id"] for p in old_manifest["plans"])
    plans = plan_questions(qids)
    calibration = {p["question_id"]: p["calibration"] for p in plans if p["calibration"]}
    for i, qid in enumerate(qids):
        raw = (source / "questions" / f"{qid}.json").read_bytes()
        hashes[qid] = hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        if record["status"] != "completed":
            raise ValueError("Do not synthesize a reduced failed trajectory")
        contexts[qid] = {
            "question": record["question"],
            "initial_answers": (
                {m: p["position"] for m, p in record["initial_positions"].items()}
                if source_version in (STATEFUL_SOURCE_VERSION, PUBLIC_HISTORY_VERSION)
                else record["initial_answers"]
            ),
            "branches": {
                arm: {
                    "replies": record["branches"][arm]["formal_replies"]["turn_level"],
                    "tone_schedule": record["branches"][arm]["tone_schedule"],
                }
                for arm in ARMS
            },
        }
    manifest = {
        "schema_version": 1,
        "kind": "offline_mock" if mock else "paid_turn_tone_quality",
        "protocol_version": VERSION,
        "implementation_sha256": implementation_hash(),
        "config": asdict(config),
        "source": {
            "directory": str(source),
            "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
            "question_sha256": hashes,
            "contexts_sha256": digest(contexts),
            "design_sha256": old_manifest["source"]["design_sha256"],
        },
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
            "output_limit_retry_policy": OUTPUT_LIMIT_RETRY_POLICY,
            "max_inflight_requests": 64,
            "question_workers": 8,
        },
        "design": {
            "unit": "question × exclusive dyad × continuation",
            "no_global_tone_arms": True,
            "reuse_debate_without_modification": True,
            "fresh_initial_answers": False,
            "baseline_shared_between_continuations_only": True,
            "third_member_visible": False,
            "tone_metadata_or_self_labels_visible_to_chairman": False,
            "chairman": config.chairman_model,
            "judge": config.judge_model,
            "length_bounds": LENGTH,
            "length_count": "Python str.split whitespace-separated words, checked before acceptance",
            "selection": "First JSON-and-length-valid answer; at most two identical retries, no truncation or preference selection",
            "paired_length_note": "Same narrow budget, not exactly identical word counts; residual gaps reported",
            "judge_protocol": "Same forced-choice rubric as archived E, two reversed presentation orders, no tie option",
            "judgment_meaning": "1 both favor debate; 0 both favor baseline; 0.5 order-inconsistent, not an explicit tie",
            "calibration": {
                "count": len(calibration),
                "selection": calibration,
                "kinds": CALIBRATION_KINDS,
                "verbose_bounds": VERBOSE_LENGTH,
                "automated_validity_review": True,
                "human_validated": False,
                "filtering": "Display all presampled cases and preferences; separately flag failed construction checks, never replace based on preferences",
            },
            "independent_question_count": len(qids),
            "analysis": "Report each dyad and continuation separately; bootstrap whole questions. No grouping by a fictitious global tone.",
        },
        "plans": plans,
    }
    counts = Counter()
    for plan in plans:
        counts.update(t.purpose for t in QualityGraph(contexts[plan["question_id"]], plan, manifest).tasks.values())
    manifest["planned_counts"] = {
        "questions": len(qids),
        "baseline_answers": len(qids) * 3,
        "debate_answers": len(qids) * 6,
        "primary_answer_pairs": len(qids) * 6,
        "primary_judgments": len(qids) * 12,
        "calibration_cases": len(calibration),
        "calibration_variants": len(calibration) * 3,
        "calibration_validations": len(calibration),
        "calibration_judgments": len(calibration) * 6,
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
    }
    return json.loads(canonical(manifest)), json.loads(canonical(contexts))


class QualityGraph(QuestionGraph):
    def __init__(self, context, plan, manifest):
        self.context, self.plan, self.manifest = context, plan, manifest
        self.question = Question.from_dict(context["question"])
        self.config = PilotConfig(**manifest["config"])
        self.tasks, self.values, self.failed, self.blocked, self.submitted = {}, {}, {}, {}, set()
        self._build()

    def key(self, suffix):
        return f"{self.question.id}/{VERSION}/{suffix}"

    def _build(self):
        config, question = self.config, self.question

        def add(suffix, deps, purpose, model, messages, schema, parse=None):
            key = self.key(suffix)
            self.tasks[key] = Task(
                key,
                frozenset(self.key(d) for d in deps),
                lambda values, count: Request(
                    key,
                    purpose,
                    model,
                    messages(values),
                    config.chairman_effort if model == config.chairman_model else "low",
                    config.chairman_tokens if model == config.chairman_model else config.judge_tokens,
                    schema,
                    False,
                ),
                parse or (lambda text: prompts.parse_json(text, schema)),
                purpose,
                model,
            )

        for pair in PAIRS:
            for arm in ("baseline", *ARMS):
                add(
                    f"{pair}/{arm}/synthesis",
                    [],
                    "synthesis",
                    config.chairman_model,
                    lambda v, p=pair, a=arm: synthesis_messages(self.context, p, a),
                    ANSWER_SCHEMA,
                    parse_answer,
                )
            for arm in ARMS:
                for order, left in enumerate(self.plan["orders"][pair][arm]):

                    def messages(values, p=pair, a=arm, target_left=left):
                        base = values[self.key(f"{p}/baseline/synthesis")]["answer"]
                        debate = values[self.key(f"{p}/{a}/synthesis")]["answer"]
                        return prompts.e_messages(question, *((debate, base) if target_left else (base, debate)))

                    add(
                        f"{pair}/{arm}/order{order}",
                        [f"{pair}/baseline/synthesis", f"{pair}/{arm}/synthesis"],
                        "judge_e",
                        config.judge_model,
                        messages,
                        prompts.E_SCHEMA,
                    )
        case = self.plan["calibration"]
        if case:
            base_suffix = f"{case['pair']}/{case['arm']}/synthesis"
            for kind in CALIBRATION_KINDS:
                bounds = VERBOSE_LENGTH if kind == "verbosity" else LENGTH
                add(
                    f"calibration/{kind}/variant",
                    [base_suffix],
                    "synthesis",
                    config.chairman_model,
                    lambda v, k=kind: variant_messages(question, v[self.key(base_suffix)]["answer"], k),
                    ANSWER_SCHEMA,
                    lambda text, b=bounds: parse_answer(text, b),
                )
                # Different counterbalanced first order by ID/kind; second is reversed.
                first_left = bool(int(digest([SEED, question.id, kind])[:8], 16) % 2)
                for order, left in enumerate((first_left, not first_left)):

                    def comparison(v, k=kind, target_left=left):
                        base = v[self.key(base_suffix)]["answer"]
                        variant = v[self.key(f"calibration/{k}/variant")]["answer"]
                        return prompts.e_messages(question, *((base, variant) if target_left else (variant, base)))

                    add(
                        f"calibration/{kind}/order{order}",
                        [base_suffix, f"calibration/{kind}/variant"],
                        "judge_e",
                        config.judge_model,
                        comparison,
                        prompts.E_SCHEMA,
                    )
            add(
                "calibration/validation",
                [base_suffix, *[f"calibration/{k}/variant" for k in CALIBRATION_KINDS]],
                "calibration_validation",
                config.judge_model,
                lambda v: validation_messages(
                    question,
                    v[self.key(base_suffix)]["answer"],
                    {k: v[self.key(f"calibration/{k}/variant")]["answer"] for k in CALIBRATION_KINDS},
                ),
                VALIDATION_SCHEMA,
            )

    def report(self, count):
        def value(suffix):
            return self.values.get(self.key(suffix))

        def preferences(prefix, target_left):
            orders = []
            for i, left in enumerate(target_left):
                rating = value(f"{prefix}/order{i}")
                orders.append({"target_side": "left" if left else "right", "judgment": rating})
            votes = [o["judgment"]["preference"] == o["target_side"] for o in orders if o["judgment"]]
            return {
                "orders": orders,
                "target_score": sum(votes) / 2 if len(votes) == 2 else None,
                "order_inconsistent": votes[0] != votes[1] if len(votes) == 2 else None,
            }

        results, outcomes = [], []
        for pair in PAIRS:
            base = value(f"{pair}/baseline/synthesis")
            for arm in ARMS:
                debated = value(f"{pair}/{arm}/synthesis")
                p = preferences(f"{pair}/{arm}", self.plan["orders"][pair][arm])
                valid = base is not None and debated is not None and p["target_score"] is not None
                results.append(
                    {
                        "pair": pair,
                        "assignment": arm,
                        "baseline": base,
                        "debated": debated,
                        "baseline_words": words(base["answer"]) if base else None,
                        "debate_words": words(debated["answer"]) if debated else None,
                        "tone_sequence": [
                            self.context["branches"][arm]["tone_schedule"][f"{pair}-{t}"] for t in range(1, 5)
                        ],
                        "preferences": p,
                        "debate_score": p["target_score"],
                    }
                )
                outcomes.append(
                    {
                        "pair": pair,
                        "assignment": arm,
                        "status": "success" if valid else "failed" if self.complete else "pending",
                    }
                )
        calibration = None
        if case := self.plan["calibration"]:
            calibration = {
                **case,
                "base": value(f"{case['pair']}/{case['arm']}/synthesis"),
                "validation": value("calibration/validation"),
                "variants": {},
            }
            for kind in CALIBRATION_KINDS:
                first = bool(int(digest([SEED, self.question.id, kind])[:8], 16) % 2)
                calibration["variants"][kind] = {
                    "answer": value(f"calibration/{kind}/variant"),
                    **preferences(f"calibration/{kind}", (first, not first)),
                }
        return {
            "question_id": self.question.id,
            "question": asdict(self.question),
            "status": "completed" if self.successful else "completed_with_failures" if self.complete else "partial",
            "E": results,
            "calibration": calibration,
            "comparisons": outcomes,
            "failed_tasks": self.failed,
            "blocked_tasks": self.blocked,
            "request_keys": sorted(self.tasks),
        }


class QualityMockProvider(MockProvider):
    def generate(self, request):
        if request.purpose == "synthesis":
            n = 300 if "/calibration/verbosity/" in request.key else 200
            return Completion(
                canonical({"answer": " ".join(["SYNTHETIC", request.key] + ["word"] * (n - 2))}),
                100,
                300,
                reasoning_tokens=50,
            )
        if request.purpose == "calibration_validation":
            value = {
                k: "yes" if "enum" in spec else "SYNTHETIC evidence"
                for k, spec in VALIDATION_SCHEMA["properties"].items()
            }
            return Completion(canonical(value), 100, 100)
        return super().generate(request)
