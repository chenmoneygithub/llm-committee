"""Two-reply interventions on frozen public-history dyads, with matched A–E reads."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from llm_committee.pivot import prompts, stateful_prompts, strong_prompts
from llm_committee.pivot.dyadic import PAIRS, make_route
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY, OUTPUT_LIMIT_RETRY_POLICY
from llm_committee.pivot.models import OPEN_MODELS, Message, PilotConfig, Question, Request, canonical, digest
from llm_committee.pivot.planning import rng_for
from llm_committee.pivot.probabilities import RATING_LABELS, TOPK_ZERO_FILL_POLICY, choice_change, distribution
from llm_committee.pivot.quality_study import ANSWER_SCHEMA, QualityMockProvider, parse_answer
from llm_committee.pivot.quality_study import synthesis_messages as original_synthesis
from llm_committee.pivot.stateful_study import PUBLIC_HISTORY_VERSION, StatefulMockProvider, rating_metrics
from llm_committee.pivot.taskgraph import QuestionGraph, Task
from scripts.scale_public_history import ROOT, read, sha

VERSION = "public-history-forced-feedback-2026-09-28-v1"
SOURCE = ROOT / "runs/public-history-extension30-20260928/combined/mixed_family/dyadic"
QUALITY = SOURCE.parent / "quality"
OUTPUT = ROOT / "runs/forced-feedback-public-history-300-20260928"
SEED = 20260928
ARMS = ("natural", "forced")
FORCE = (
    "For this reply only, take the opposing side to the central conclusion of the immediately preceding "
    "incoming_peer_message. This assigned stance overrides agreement-seeking instructions, while retaining "
    "this turn's assigned delivery tone. Clearly identify the conclusion you reject and defend a substantively "
    "opposing position with concrete reasons. Do not merely criticize wording or add minor qualifications. "
    "Do not invent facts or citations, and do not mention this private assignment in your reply. "
    "Use the usual four-field response format and length limits. Report the agreement label that best "
    "describes your actual reply; no particular label is required."
)


def request_identity(document):
    """Only the local journal key may differ when reusing an existing measurement."""
    return digest({k: v for k, v in document.items() if k != "key"})


def implementation_hashes():
    paths = [
        *sorted((ROOT / "llm_committee/pivot").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_forced_feedback_public_history.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def source_archives(paths):
    """Select first archived success by storage key, never by label or preference."""
    indexes = defaultdict(dict)
    for source in paths:
        with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            for row in db.execute("SELECT * FROM calls WHERE status='completed' ORDER BY key"):
                record = dict(row)
                request = json.loads(record["request"])
                if request["purpose"] not in ("judge_b", "judge_c", "d_choice", "d_text", "synthesis"):
                    continue
                identity = request_identity(request)
                qid = request["key"].split("/", 1)[0]
                indexes[qid].setdefault(
                    identity,
                    {
                        "request": request,
                        "value": json.loads(record["parsed"]),
                        "provenance": {
                            "directory": str(source),
                            "storage_key": record["key"],
                            "row_sha256": digest(record),
                        },
                    },
                )
    return dict(indexes)


def select_cases(qids):
    """All 150 distinct early prefixes; one outcome-blind late continuation per dyad."""
    ordered = sorted(qids)
    if len(ordered) != 50 or len(set(ordered)) != 50:
        raise ValueError("Expected the frozen fifty-question source")
    choices = {}
    for pair in PAIRS:
        assignments = ["original"] * 25 + ["alternate"] * 25
        rng_for(SEED, VERSION, pair, "late-continuation").shuffle(assignments)
        choices[pair] = dict(zip(ordered, assignments, strict=True))
    plans = []
    for qid in ordered:
        cases = []
        for pair, (recipient, challenger) in PAIRS.items():
            for cut in (1, 3):
                cases.append(
                    {
                        "id": f"{pair}-T{cut}",
                        "question_id": qid,
                        "pair": pair,
                        "cut_T": cut,
                        "return_T": cut + 1,
                        "cut_node": f"{pair}-{cut}",
                        "return_node": f"{pair}-{cut+1}",
                        "recipient": recipient,
                        "challenger": challenger,
                        "source_assignment": "original" if cut == 1 else choices[pair][qid],
                        "forced_left_first": bool(int(digest([SEED, qid, pair, cut, "E-order"])[:8], 16) % 2),
                    }
                )
        plans.append({"question_id": qid, "cases": cases})
    return plans


def prepare(*, mock=False):
    old, quality = read(SOURCE / "manifest.json"), read(QUALITY / "manifest.json")
    if old["protocol_version"] != PUBLIC_HISTORY_VERSION or old["config"] != quality["config"]:
        raise ValueError("Wrong source protocol or inconsistent configuration")
    cfg = PilotConfig(**old["config"])
    cfg.validate()
    if cfg.roster != "mixed_family" or cfg.judge_model != "gemini-3.8-flash":
        raise ValueError("Only the existing mixed-family roster and judge are in scope")
    if any(read(p / "report.json")["status"] != "completed" for p in (SOURCE, QUALITY)):
        raise ValueError("Expected completed source debates and quality answers")
    archives = {} if mock else source_archives((SOURCE, QUALITY))
    plans = select_cases(p["question_id"] for p in old["plans"])
    contexts, preservation = {}, {}
    for source in (SOURCE, QUALITY):
        for name in ("manifest.json", "report.json", "requests.sqlite3", "source-contexts.json"):
            preservation[str(source / name)] = sha(source / name)
    for plan in plans:
        qid = plan["question_id"]
        path = SOURCE / "questions" / f"{qid}.json"
        q = read(path)
        if q["status"] != "completed":
            raise ValueError("A source discussion is incomplete")
        for pair in PAIRS:
            for t in (1, 2):
                nid = f"{pair}-{t}"
                a, b = (q["branches"][arm] for arm in ("original", "alternate"))
                if (
                    a["formal_replies"]["turn_level"][nid] != b["formal_replies"]["turn_level"][nid]
                    or a["tone_schedule"][nid] != b["tone_schedule"][nid]
                ):
                    raise ValueError("Expected the same original/alternate early prefix")
        contexts[qid] = {"record": q, "archive": archives.get(qid, {})}
        preservation[str(path)] = sha(path)
        other = QUALITY / "questions" / f"{qid}.json"
        preservation[str(other)] = sha(other)
    manifest = {
        "kind": "offline_mock" if mock else "paid_public_history_forced_feedback",
        "protocol_version": VERSION,
        "config": asdict(cfg),
        "plans": plans,
        "implementation_sha256": digest(implementation_hashes()),
        "implementation_files": implementation_hashes(),
        "force_instruction": FORCE,
        "agreement_rubric": old["agreement_rubric"],
        "probability_readout": dict(TOPK_ZERO_FILL_POLICY),
        "source": {"directory": str(SOURCE), "contexts_sha256": digest(contexts), "preservation": preservation},
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
            "output_limit_retry_policy": OUTPUT_LIMIT_RETRY_POLICY,
            "max_inflight_requests": 64,
            "question_workers": 8,
        },
        "design": {
            "samples": 300,
            "independent_questions": 50,
            "early": 150,
            "late": 150,
            "sampling": "One unique early prefix and one late continuation per question/dyad; balanced late source schedules, no outcome selection",
            "stop": "T1 intervention ends at T2; T3 intervention ends at T4",
            "tone": "Existing per-turn schedule, identical within natural/forced pairs",
            "formal_outputs": ["reply", "agreement", "choice", "position"],
            "private_position_feedback": False,
            "force_label": False,
            "natural_control": "Archived, endpoint-matched public replies and positions; no successful debate regeneration",
            "reuse": "Exact model/settings/messages/schema match excluding journal key; deterministic first archived success",
            "C_D": "Recipient's own previous participation -> return; condition on challenger and recipient actual labels",
            "D_scope": "Mixed-family open-weight recipients only; neutral, no-reasoning, independent side reads",
            "E": "Natural vs forced synthesis at same endpoint, Terra 190–210 words, original Gemini rubric, both orders",
            "E_calibration": "No new battery; previous diagnostics are contextual evidence, not new validation",
            "inference": "Question-clustered; label strata are descriptive post-intervention groups",
        },
    }
    counts = Counter()
    for plan in plans:
        counts.update(t.purpose for t in FeedbackGraph(contexts[plan["question_id"]], plan, manifest).tasks.values())
    manifest["planned_counts"] = {
        "cases": 300,
        "formal_replies": 600,
        "by_purpose_including_reuse": dict(counts),
        "logical_tasks_including_reuse": sum(counts.values()),
    }
    return json.loads(canonical(manifest)), contexts


def verify_preservation(manifest):
    for path, expected in manifest["source"]["preservation"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    if implementation_hashes() != manifest["implementation_files"]:
        raise ValueError("Frozen implementation changed")


class FeedbackGraph(QuestionGraph):
    def __init__(self, context, plan, manifest):
        self.context, self.plan, self.manifest = context, plan, manifest
        self.record = context["record"]
        self.question = Question.from_dict(self.record["question"])
        self.config = PilotConfig(**manifest["config"])
        self.route = make_route()
        self.initial = {int(k): v for k, v in self.record["initial_positions"].items()}
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted, self.eligible_reuse = set(), set()
        self.reused, self.cache_after, self.case_keys, self.measurements = {}, {}, {}, {}
        self._build()
        reachable = set()
        while more := {t.key for t in self.tasks.values() if t.dependencies <= reachable} - reachable:
            reachable.update(more)
        if reachable != self.tasks.keys():
            raise ValueError("Invalid feedback dependency graph")

    def key(self, suffix):
        return f"{self.question.id}/{VERSION}/{suffix}"

    def archived(self, case):
        return self.record["branches"][case["source_assignment"]]["formal_replies"]["turn_level"]

    def history(self, case, values, arm):
        archived = self.archived(case)
        history = {n.id: archived[n.id] for n in self.route.path(case["cut_node"])[:-1]}
        for role, nid in (("feedback", case["cut_node"]), ("return", case["return_node"])):
            if arm == "natural":
                history[nid] = archived[nid]
            elif self.key(f"{case['id']}/forced/{role}") in values:
                history[nid] = values[self.key(f"{case['id']}/forced/{role}")]
        return history

    def before(self, case):
        return self.initial[case["recipient"]] if case["cut_T"] == 1 else self.archived(case)[f"{case['pair']}-2"]

    def synthesis(self, case, values, arm):
        # Reuse the exact chairman task; only the endpoint-limited public nodes vary.
        template = original_synthesis(
            {
                "question": self.record["question"],
                "initial_answers": {str(m): v["position"] for m, v in self.initial.items()},
            },
            case["pair"],
            "baseline",
        )
        payload = json.loads(template[-1].text)
        history = self.history(case, values, arm)
        payload["discussion"] = [
            {
                "id": n.id,
                "member": n.receiver,
                "reply_to": n.parent or f"initial/{n.sender}",
                "text": history[n.id]["reply"],
            }
            for n in self.route.path(case["return_node"])
        ]
        return (*template[:-1], Message("user", canonical(payload)))

    def _build(self):
        cfg, q = self.config, self.question

        def add(suffix, deps, purpose, model, builder, effort, tokens, schema=None, labels=(), parse=None, reuse=False):
            key = self.key(suffix)
            if key not in self.tasks:
                self.tasks[key] = Task(
                    key,
                    frozenset(deps),
                    lambda v, c: Request(
                        key, purpose, model, builder(v, c), effort, tokens, schema, False, candidate_labels=labels
                    ),
                    parse or (lambda s: prompts.parse_json(s, schema)),
                    purpose,
                    model,
                )
            if reuse:
                self.eligible_reuse.add(key)
            return key

        for case in self.plan["cases"]:
            cid = case["id"]
            schedule = self.record["branches"][case["source_assignment"]]["tone_schedule"]

            def debate(v, count, c=case, s=schedule, role="feedback"):
                nid = c["cut_node"] if role == "feedback" else c["return_node"]
                messages = stateful_prompts.debate_messages(
                    q, s[nid], self.initial, self.route, self.history(c, v, "forced"), nid, explicit_position=False
                )
                if role == "feedback":
                    messages = (*messages[:-1], Message("user", messages[-1].text + "\n\n" + FORCE))
                return messages

            feedback = add(
                f"{cid}/forced/feedback",
                [],
                "debate",
                cfg.members[case["challenger"]],
                debate,
                cfg.debate_effort,
                cfg.debate_tokens,
                stateful_prompts.reply_schema(q),
            )
            returned = add(
                f"{cid}/forced/return",
                [feedback],
                "debate",
                cfg.members[case["recipient"]],
                lambda v, count, fn=debate: fn(v, count, role="return"),
                cfg.debate_effort,
                cfg.debate_tokens,
                stateful_prompts.reply_schema(q),
            )
            keys = {feedback, returned}
            meta = {"arms": {}}
            model = cfg.members[case["recipient"]]
            before_node = None if case["cut_T"] == 1 else f"{case['pair']}-2"
            if model in OPEN_MODELS:
                prior = add(
                    f"D1-reference/{before_node or 'initial-' + str(case['recipient'])}",
                    [],
                    "d_choice",
                    model,
                    lambda v, count, c=case, n=before_node: stateful_prompts.choice_messages(
                        q,
                        c["recipient"],
                        self.initial,
                        self.route,
                        self.history(c, {}, "natural"),
                        n,
                        explicit_position=False,
                    ),
                    "none",
                    cfg.probability_tokens,
                    labels=q.labels,
                    parse=lambda s: stateful_prompts.parse_choice(s, q),
                    reuse=True,
                )
                keys.add(prior)
                meta["D1_before"] = prior
            for arm in ARMS:
                own = {}
                for role in ("feedback", "return"):
                    deps = [] if arm == "natural" else ([feedback] if role == "feedback" else [feedback, returned])

                    def b_messages(v, count, c=case, a=arm, r=role):
                        h = self.history(c, v, a)
                        nid = c["cut_node"] if r == "feedback" else c["return_node"]
                        node = self.route.get(nid)
                        peer = h[node.parent]["reply"] if node.parent else self.initial[node.sender]["position"]
                        return strong_prompts.b_messages(q, peer, h[nid]["reply"])

                    own[f"B_{role}"] = add(
                        f"{cid}/{arm}/B-{role}",
                        deps,
                        "judge_b",
                        cfg.judge_model,
                        b_messages,
                        "low",
                        cfg.judge_tokens,
                        strong_prompts.B_SCHEMA,
                        reuse=arm == "natural",
                    )
                own["C"] = add(
                    f"{cid}/{arm}/C",
                    [] if arm == "natural" else [returned],
                    "judge_c",
                    cfg.judge_model,
                    lambda v, count, c=case, a=arm: prompts.c_messages(
                        q, self.before(c)["position"], self.history(c, v, a)[c["return_node"]]["position"]
                    ),
                    "low",
                    cfg.judge_tokens,
                    prompts.C_SCHEMA,
                    reuse=arm == "natural",
                )
                if model in OPEN_MODELS:
                    suffix = (
                        f"D1-reference/{case['return_node']}"
                        + (f"-{case['source_assignment']}" if case["cut_T"] == 3 else "")
                        if arm == "natural"
                        else f"{cid}/forced/D1"
                    )
                    own["D1"] = add(
                        suffix,
                        [] if arm == "natural" else [feedback],
                        "d_choice",
                        model,
                        lambda v, count, c=case, a=arm: stateful_prompts.choice_messages(
                            q,
                            c["recipient"],
                            self.initial,
                            self.route,
                            self.history(c, v, a),
                            c["return_node"],
                            explicit_position=False,
                        ),
                        "none",
                        cfg.probability_tokens,
                        labels=q.labels,
                        parse=lambda s: stateful_prompts.parse_choice(s, q),
                        reuse=arm == "natural",
                    )
                    for treatment in ("argument", "control"):

                        def d2(v, count, c=case, a=arm, t=treatment, m=model):
                            h = self.history(c, v, a)
                            incoming = h[c["cut_node"]]["reply"]
                            if t == "control":
                                n = max(1, round(count(m, incoming) / count(m, prompts.FILLER_SENTENCE)))
                                incoming = " ".join([prompts.FILLER_SENTENCE] * n)
                            return stateful_prompts.text_messages(
                                q, self.initial, self.route, h, c["return_node"], incoming, explicit_position=False
                            )

                        key = add(
                            f"{cid}/{arm}/D2-{treatment}",
                            [] if arm == "natural" else [feedback],
                            "d_text",
                            model,
                            d2,
                            "none",
                            cfg.probability_tokens,
                            labels=RATING_LABELS,
                            parse=prompts.parse_rating,
                            reuse=arm == "natural",
                        )
                        self.cache_after[key] = {own["D1"] if treatment == "argument" else own["D2_argument"]}
                        own[f"D2_{treatment}"] = key
                own["E_answer"] = add(
                    f"{cid}/{arm}/synthesis",
                    [] if arm == "natural" else [feedback, returned],
                    "synthesis",
                    cfg.chairman_model,
                    lambda v, count, c=case, a=arm: self.synthesis(c, v, a),
                    cfg.chairman_effort,
                    cfg.chairman_tokens,
                    ANSWER_SCHEMA,
                    parse=parse_answer,
                    reuse=arm == "natural",
                )
                keys.update(own.values())
                meta["arms"][arm] = own
            orders = []
            for order, target_left in enumerate((case["forced_left_first"], not case["forced_left_first"])):
                natural, forced = (meta["arms"][a]["E_answer"] for a in ARMS)
                key = add(
                    f"{cid}/E-order{order}",
                    [natural, forced],
                    "judge_e",
                    cfg.judge_model,
                    lambda v, count, n=natural, f=forced, left=target_left: prompts.e_messages(
                        q, *((v[f]["answer"], v[n]["answer"]) if left else (v[n]["answer"], v[f]["answer"]))
                    ),
                    "low",
                    cfg.judge_tokens,
                    prompts.E_SCHEMA,
                )
                orders.append({"key": key, "target_side": "left" if target_left else "right"})
                keys.add(key)
            meta["E_orders"] = orders
            self.measurements[cid], self.case_keys[cid] = meta, keys

    def ready(self):
        terminal = self.values.keys() | self.failed.keys() | self.blocked.keys()
        return next(
            (
                t
                for t in self.tasks.values()
                if t.key not in self.submitted
                and t.key not in self.blocked
                and t.dependencies <= self.values.keys()
                and self.cache_after.get(t.key, set()) <= terminal
            ),
            None,
        )

    def restore(self, journal, offline_provider):
        # Load static exact-match source measurements before replaying this journal.
        saved = {k for (k,) in journal.db.execute("SELECT key FROM calls")}
        for key in sorted(self.eligible_reuse):
            task = self.tasks[key]
            if task.dependencies:
                raise ValueError("Archived reuse must be independent of new generations")
            request = task.build({}, offline_provider.token_count)
            archived = self.context["archive"].get(request_identity(request.document()))
            if archived:
                if key in saved:
                    raise ValueError("A reused source measurement was also dispatched")
                self.accept(task, request, archived["value"])
                self.reused[key] = archived["provenance"]
        super().restore(journal, offline_provider)

    def report(self, count):
        def dist(key, labels):
            value = self.values.get(key)
            return distribution(value["_readout"], labels, missing_as_zero=True) if value else None

        cases, trajectories = [], []
        for case in self.plan["cases"]:
            cid, meta = case["id"], self.measurements[case["id"]]
            before = self.before(case)
            before_dist = dist(meta["D1_before"], self.question.labels) if "D1_before" in meta else None
            arms = {}
            for arm in ARMS:
                keys = meta["arms"][arm]
                history = self.history(case, self.values, arm)
                feedback, returned = (history.get(case[k]) for k in ("cut_node", "return_node"))
                after_dist = dist(keys["D1"], self.question.labels) if "D1" in keys else None
                d2 = (
                    {t: dist(keys[f"D2_{t}"], RATING_LABELS) for t in ("argument", "control")}
                    if "D2_argument" in keys
                    else {}
                )
                arms[arm] = {
                    "feedback": feedback,
                    "return": returned,
                    "challenger_label": feedback["agreement"] if feedback else None,
                    "recipient_label": returned["agreement"] if returned else None,
                    "B_feedback": self.values.get(keys["B_feedback"]),
                    "B_return": self.values.get(keys["B_return"]),
                    "C_choice_changed": (
                        before["choice"] != returned["choice"]
                        if returned and before["choice"] is not None and returned["choice"] is not None
                        else None
                    ),
                    "C_text": self.values.get(keys["C"]),
                    "position_before": before,
                    "D1_before": before_dist,
                    "D1_after": after_dist,
                    "D1_prior_choice_change_pp": (
                        choice_change(before_dist, after_dist, before["choice"]) if before_dist and after_dist else None
                    ),
                    "D2": (
                        {**d2, **rating_metrics(d2["argument"], d2["control"]), "fixed_position": before["position"]}
                        if d2 and all(d2.values())
                        else None
                    ),
                    "synthesis": self.values.get(keys["E_answer"]),
                    "requests": keys,
                }
            orders = [{**o, "judgment": self.values.get(o["key"])} for o in meta["E_orders"]]
            votes = [o["judgment"]["preference"] == o["target_side"] for o in orders if o["judgment"]]
            failed = self.case_keys[cid] & (self.failed.keys() | self.blocked.keys())
            formal_ok = arms["forced"]["feedback"] is not None and arms["forced"]["return"] is not None
            status = (
                "completed_with_failures"
                if failed
                else "completed" if self.case_keys[cid] <= self.values.keys() else "partial"
            )
            cases.append(
                {
                    **case,
                    "status": status,
                    "model": self.config.members[case["recipient"]],
                    "tones": {
                        n.id: self.record["branches"][case["source_assignment"]]["tone_schedule"][n.id]
                        for n in self.route.path(case["return_node"])
                    },
                    "arms": arms,
                    "E_orders": orders,
                    "E_forced_wins": sum(votes) / 2 if len(votes) == 2 else None,
                    "E_order_inconsistent": votes[0] != votes[1] if len(votes) == 2 else None,
                    "failed_keys": sorted(failed),
                    "measurement_keys": meta,
                }
            )
            trajectories.append(
                {"case": cid, "status": "success" if formal_ok else "failed" if self.complete else "pending"}
            )
        return {
            "question_id": self.question.id,
            "question": self.record["question"],
            "protocol_version": VERSION,
            "status": "completed" if self.successful else "completed_with_failures" if self.complete else "partial",
            "cases": cases,
            "trajectories": trajectories,
            "initial_positions": self.record["initial_positions"],
            "reused_measurements": self.reused,
            "request_keys": sorted(self.tasks),
            "failed_tasks": self.failed,
            "blocked_tasks": self.blocked,
        }


class FeedbackMockProvider(StatefulMockProvider):
    def generate(self, request):
        if request.purpose == "synthesis":
            return QualityMockProvider().generate(request)
        return super().generate(request)
