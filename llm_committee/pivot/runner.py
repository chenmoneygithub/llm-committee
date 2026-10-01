"""Mixed-family A–E orchestration over a fixed tree; measurements are side reads."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import prompts
from .agreement import agreement_manifest
from .models import AGREEMENT, MAIN_STUDY_REUSE_POLICY, OPEN_MODELS, TONES, PilotConfig, Question, Request, canonical
from .planning import plan_question, read_id, rng_for, route_from_plan
from .probabilities import (
    RATING_LABELS,
    TOPK_ZERO_FILL_POLICY,
    choice_change,
    distribution,
    merge_scored_readout,
    own_position_difference,
    uses_topk_zero_fill,
)
from .providers import Provider
from .storage import Journal, ProbabilityReadBlocked, RunBlocked


def execution_blockers(config: PilotConfig) -> list[str]:
    if config.roster not in ("same_family", "mixed_family"):
        return ["This pilot entry point implements the same-family and mixed-family workflows only"]
    return []


def require_executable(config: PilotConfig) -> None:
    blockers = execution_blockers(config)
    if blockers:
        raise RunBlocked("Pilot execution is not ready: " + "; ".join(blockers))


def probability_plan(plan: dict, config: PilotConfig) -> dict:
    """Declare D's work without inventing probabilities or changing the shared sample."""
    members = (1, 2) if config.roster == "mixed_family" else ()
    return {
        "status": "planned" if members else "not_applicable",
        "members": [{"member": m, "model": config.members[m]} for m in members],
        "choice_reading_ids": [r["id"] for r in plan["readings"] if r["member"] in members],
        "text_pairs": [
            {
                "event_id": e["id"],
                "member": e["member"],
                "fixed_full_position_reading": e["previous_reading"],
                "arms": ["argument", "control"],
                "context_rule": "Same pre-peer context; exclude the receiver's subsequent formal reply",
            }
            for e in plan["events"]
            if e["member"] in members
        ],
    }


def manifest_for(questions: tuple[Question, ...], config: PilotConfig, *, mock: bool) -> dict:
    config.validate()
    if not questions or len({q.fingerprint for q in questions}) != len(questions):
        raise ValueError("Pilot needs distinct questions")
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    plans = [plan_question(q, config) for q in questions]
    for plan in plans:
        plan["D"] = probability_plan(plan, config)
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    versions = {}
    for name in (
        "openai",
        "httpx",
        "databricks-sdk",
        "tinker",
        "tinker-cookbook",
        "tml-renderers",
        "transformers",
        "tokenizers",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    manifest = {
        "schema_version": 2,
        "kind": "offline_mock" if mock else "paid_engineering_pilot",
        "prompt_version": prompts.PROMPT_VERSION,
        "agreement_rubric": agreement_manifest(),
        "implementation_sha256": code.hexdigest(),
        "runtime_packages": versions,
        "cost_accounting": {
            "kind": "token_based_estimate_not_provider_invoice",
            "price_date": "2026-09-25",
            "databricks_closed_model_multiplier": 1.1 if config.closed_provider == "databricks" else None,
            "databricks_note": "Standard list rates at assumed USD 0.07/DBU, conservatively allowing 10% regional uplift",
        },
        "config": asdict(config),
        "probability_readout": dict(TOPK_ZERO_FILL_POLICY),
        "questions": [asdict(q) for q in questions],
        "question_fingerprints": [q.fingerprint for q in questions],
        "main_study_reuse": dict(MAIN_STUDY_REUSE_POLICY),
        "plans": plans,
        "execution": {
            "implemented": not execution_blockers(config),
            "blockers": execution_blockers(config),
            "live_verified": False,
            "acceptance": "Real pilot must verify direct reads, original top-k probability capture and billing; mocks cannot",
        },
        "excluded": [
            *([] if config.roster == "mixed_family" else ["D: roster has no open-weight members"]),
            "A instruction ablation",
            "human annotation",
            "main-study inference",
        ],
    }
    return json.loads(canonical(manifest))  # Canonical list/tuple shape must survive a CLI restart.


def expected_counts(manifest: dict) -> dict:
    plans = manifest["plans"]
    judge = manifest["config"]["judge_model"] is not None
    result = {
        "questions": len(plans),
        "debates": len(plans) * 3,
        "initial_answers": len(plans) * 3,
        "formal_replies": sum(len(p["route"]) * 3 for p in plans),
        "sampled_events": len(plans) * 8,
        "C_readings": sum(len(p["readings"]) for p in plans),
        "B_judgments": len(plans) * 8 if judge else 0,
        "C_judgments": sum(len(p["text_pairs"]) for p in plans) if judge else 0,
        "syntheses": len(plans) * 4,
        "E_judgments": len(plans) * 6 if judge else 0,
        "D_choice_readings": sum(len(p["D"]["choice_reading_ids"]) for p in plans),
        "D_text_pairs": sum(len(p["D"]["text_pairs"]) for p in plans),
    }
    # New runs use original top-k values only. Keep historical scoring allowances
    # when planning/replaying an explicitly unchanged archived manifest.
    result["D_calls"] = 2 * result["D_text_pairs"]
    zero_fill = uses_topk_zero_fill(manifest)
    result["D_choice_supplemental_scoring_allowance"] = 0 if zero_fill else result["D_choice_readings"]
    result["D_text_supplemental_scoring_allowance"] = 0 if zero_fill else result["D_calls"]
    result["logical_calls"] = sum(
        result[k]
        for k in (
            "initial_answers",
            "formal_replies",
            "C_readings",
            "B_judgments",
            "C_judgments",
            "syntheses",
            "E_judgments",
            "D_calls",
        )
    )
    result["logical_calls_with_scoring_allowance"] = (
        result["logical_calls"]
        + result["D_choice_supplemental_scoring_allowance"]
        + result["D_text_supplemental_scoring_allowance"]
    )
    return result


def run_pilot(
    questions: tuple[Question, ...], config: PilotConfig, manifest: dict, journal: Journal, provider: Provider
) -> dict:
    require_executable(config)  # Also protects direct callers before any model request.
    measurement_failures = []
    zero_fill = uses_topk_zero_fill(manifest)

    def read_distribution(key, model, labels, reading):
        readout = reading["_readout"]
        if not zero_fill and set(readout["candidate_logprobs"]) != set(labels):
            request = Request(
                key + "/candidate_scores",
                "candidate_scores",
                model,
                (),
                "none",
                1,
                candidate_labels=labels,
                scoring=readout,
            )
            try:
                readout = journal.call(
                    request, provider, lambda text: merge_scored_readout(readout, json.loads(text), labels)
                )
            except ProbabilityReadBlocked as exc:
                if not manifest["execution"].get("record_probability_failures", False):
                    raise
                failure = {"key": key, "scoring_key": exc.key, "model": model, "reason": exc.reason}
                measurement_failures.append(failure)
                if journal.progress:
                    journal.progress({"event": "measurement_failed_no_retry", **failure})
                return None
        try:
            return distribution(readout, labels, missing_as_zero=zero_fill)
        except (ValueError, KeyError, TypeError) as exc:
            raise RunBlocked(f"Invalid probability read for {key}; original response remains in the journal") from exc

    def invoke(key, purpose, model, messages, effort, tokens, output_schema=None):
        request = Request(
            key,
            purpose,
            model,
            messages,
            effort,
            tokens,
            output_schema,
            config.cache and purpose in ("debate", "position"),
        )
        return journal.call(request, provider, lambda text: prompts.parse_json(text, output_schema))

    results = []
    for question, plan in zip(questions, manifest["plans"], strict=True):
        failure_start = len(measurement_failures)
        route = route_from_plan(plan)
        prefix = f"{question.id}/{config.roster}"
        initial = {}
        for member, model in enumerate(config.members):
            initial[member] = invoke(
                f"{prefix}/initial/{member}",
                "initial",
                model,
                prompts.initial_messages(question, member),
                config.debate_effort,
                config.initial_tokens,
                prompts.INITIAL_SCHEMA,
            )["position"]
        # These dictionaries are separate from replies and are never passed to debate/synthesis builders.
        replies: dict[str, dict[str, dict]] = {}
        positions, choice_distributions = {}, {}

        def read_position(
            reading,
            *,
            question=question,
            initial=initial,
            route=route,
            replies=replies,
            prefix=prefix,
            positions=positions,
            choice_distributions=choice_distributions,
        ):
            request = Request(
                f"{prefix}/C/{reading['id']}",
                "position",
                config.members[reading["member"]],
                prompts.position_messages(question, reading, initial, route, replies.get(reading["tone"], {})),
                "none",
                config.position_tokens,
                cache=config.cache,
                candidate_labels=question.labels if config.members[reading["member"]] in OPEN_MODELS else (),
            )
            positions[reading["id"]] = journal.call(
                request, provider, lambda text, q=question: prompts.parse_position(text, q)
            )
            if request.candidate_labels:
                measured = read_distribution(
                    request.key, request.model, request.candidate_labels, positions[reading["id"]]
                )
                if measured is not None:
                    choice_distributions[reading["id"]] = measured

        # Check the planned baseline direct reads before paying for all debate turns.
        # This changes scheduling only: same nodes, contexts and measurements, no new retry arm.
        for reading in plan["readings"]:
            if reading["node_id"] is None:
                read_position(reading)
        for tone in TONES:
            replies[tone] = {}
            for node in route.nodes:
                replies[tone][node.id] = invoke(
                    f"{prefix}/{tone}/debate/{node.id}",
                    "debate",
                    config.members[node.receiver],
                    prompts.debate_messages(question, tone, initial, route, replies[tone], node.id),
                    config.debate_effort,
                    config.debate_tokens,
                    prompts.REPLY_SCHEMA,
                )
        for reading in plan["readings"]:
            if reading["node_id"] is not None:
                read_position(reading)

        text_distributions = {}
        for event in plan["events"]:
            model = config.members[event["member"]]
            if model not in OPEN_MODELS:
                continue
            node = route.get(event["node_id"])
            branch = replies[event["tone"]]
            peer = branch[node.parent]["reply"] if node.parent else initial[node.sender]
            fixed_position = positions[event["previous_reading"]]["position"]
            argument_tokens = provider.token_count(model, peer)
            sentence_tokens = provider.token_count(model, prompts.FILLER_SENTENCE)
            repeats = max(1, round(argument_tokens / sentence_tokens))
            filler = " ".join([prompts.FILLER_SENTENCE] * repeats)
            arms = {}
            for arm, incoming in (("argument", peer), ("control", filler)):
                request = Request(
                    f"{prefix}/Dtext/{event['id']}/{arm}",
                    "d_text",
                    model,
                    prompts.d_text_messages(question, event, initial, route, branch, fixed_position, incoming),
                    "none",
                    config.probability_tokens,
                    candidate_labels=RATING_LABELS,
                )
                reading = journal.call(request, provider, prompts.parse_rating)
                measured = read_distribution(request.key, model, RATING_LABELS, reading)
                arms[arm] = {
                    **(
                        measured if measured is not None else {"status": "invalid_probability_read", "key": request.key}
                    ),
                    "sampled_rating": reading["rating"],
                }
            text_distributions[event["id"]] = {
                **arms,
                "reference_reading": event["previous_reading"],
                "fixed_full_position": fixed_position,
                "argument_tokens": argument_tokens,
                "control_tokens": provider.token_count(model, filler),
                "filler_repetitions": repeats,
                "status": "completed" if all("probabilities" in a for a in arms.values()) else "incomplete",
                "mean_own_agreement_argument_minus_control": (
                    own_position_difference(arms["argument"], arms["control"])
                    if all("probabilities" in a for a in arms.values())
                    else None
                ),
            }

        b_judgments, c_judgments = {}, {}
        if config.judge_model:
            for event in plan["events"]:
                node = route.get(event["node_id"])
                branch = replies[event["tone"]]
                peer = branch[node.parent]["reply"] if node.parent else initial[node.sender]
                b_judgments[event["id"]] = invoke(
                    f"{prefix}/B/{event['id']}",
                    "judge_b",
                    config.judge_model,
                    prompts.b_messages(question, peer, branch[node.id]["reply"]),
                    "low",
                    config.judge_tokens,
                    prompts.B_SCHEMA,
                )
            for pair in plan["text_pairs"]:
                c_judgments[pair["id"]] = invoke(
                    f"{prefix}/Cjudge/{pair['id']}",
                    "judge_c",
                    config.judge_model,
                    prompts.c_messages(
                        question, positions[pair["before"]]["position"], positions[pair["after"]]["position"]
                    ),
                    "low",
                    config.judge_tokens,
                    prompts.C_SCHEMA,
                )

        baseline = invoke(
            f"{prefix}/E/baseline",
            "synthesis",
            config.chairman_model,
            prompts.synthesis_messages(question, initial, route, None),
            config.chairman_effort,
            config.chairman_tokens,
            prompts.SYNTHESIS_SCHEMA,
        )["answer"]
        syntheses, preferences = {}, {}
        for tone in TONES:
            syntheses[tone] = invoke(
                f"{prefix}/E/{tone}/synthesis",
                "synthesis",
                config.chairman_model,
                prompts.synthesis_messages(question, initial, route, replies[tone]),
                config.chairman_effort,
                config.chairman_tokens,
                prompts.SYNTHESIS_SCHEMA,
            )["answer"]
            if config.judge_model:
                first_left = bool(
                    rng_for(config.seed, question.fingerprint, config.roster, tone, "answer_order").randrange(2)
                )
                votes = []
                orders = []
                for order in range(2):
                    debate_left = first_left if order == 0 else not first_left
                    left, right = (syntheses[tone], baseline) if debate_left else (baseline, syntheses[tone])
                    verdict = invoke(
                        f"{prefix}/E/{tone}/order{order}",
                        "judge_e",
                        config.judge_model,
                        prompts.e_messages(question, left, right),
                        "low",
                        config.judge_tokens,
                        prompts.E_SCHEMA,
                    )
                    votes.append(verdict["preference"] == ("left" if debate_left else "right"))
                    orders.append({"debate_side": "left" if debate_left else "right", **verdict})
                preferences[tone] = {
                    "orders": orders,
                    "debate_score": sum(votes) / 2,
                    "order_inconsistent": votes[0] != votes[1],
                }

        event_results = []
        for event in plan["events"]:
            before, start, current = (
                positions[event[k]] for k in ("previous_reading", "initial_reading", "current_reading")
            )

            def changed(a, b):
                return a["choice"] != b["choice"] if a["choice"] is not None and b["choice"] is not None else None

            event_results.append(
                {
                    **event,
                    "A": replies[event["tone"]][event["node_id"]]["agreement"],
                    "B": b_judgments.get(event["id"]),
                    "C_choice_adjacent_changed": changed(before, current),
                    "C_choice_initial_changed": changed(start, current),
                    "C_text_adjacent": c_judgments.get(event["adjacent_pair"]),
                    "C_text_final": c_judgments.get(event["final_pair"]),
                    "D_choice_adjacent_pp": (
                        choice_change(
                            choice_distributions[event["previous_reading"]],
                            choice_distributions[event["current_reading"]],
                            before["choice"],
                        )
                        if all(event[k] in choice_distributions for k in ("current_reading", "previous_reading"))
                        else None
                    ),
                    "D_choice_initial_pp": (
                        choice_change(
                            choice_distributions[event["initial_reading"]],
                            choice_distributions[event["current_reading"]],
                            start["choice"],
                        )
                        if all(event[k] in choice_distributions for k in ("current_reading", "initial_reading"))
                        else None
                    ),
                    "D_text": text_distributions.get(event["id"]),
                }
            )

        # Ragged trajectory/member/actual participation-index view. Shared readings are REFERENCES,
        # not additional observations; unsampled participations are not invented or interpolated.
        trajectory_positions = {}
        for tone in TONES:
            trajectory_positions[tone] = {}
            for leaf in route.leaves:
                by_member = {str(m): [] for m in range(3)}
                for member in range(3):
                    key = f"initial/{member}"
                    if key in positions:
                        by_member[str(member)].append({"time_index": 0, "T": 0, "reading_id": key})
                for node in route.path(leaf.id):
                    key = read_id(tone, node, node.receiver)
                    if key in positions:
                        by_member[str(node.receiver)].append(
                            {"time_index": route.participation(node.id), "T": node.depth, "reading_id": key}
                        )
                trajectory_positions[tone][leaf.id] = by_member

        a_counts = {}
        for tone in TONES:
            counts = Counter(item["agreement"] for item in replies[tone].values())
            a_counts[tone] = {
                "n_replies": len(replies[tone]),
                "unreported": counts[None],
                "counts": {label: counts[label] for label in AGREEMENT},
            }
        results.append(
            {
                "question_id": question.id,
                "question_fingerprint": question.fingerprint,
                "initial_answers": initial,
                "formal_replies": replies,
                "A": a_counts,
                "C_readings": positions,
                "positions": trajectory_positions,
                "C_text_pairs": [{**pair, "judgment": c_judgments.get(pair["id"])} for pair in plan["text_pairs"]],
                "sampled_events": event_results,
                "E": {"baseline": baseline, "debated": syntheses, "preferences": preferences},
                "D": (
                    {
                        "status": (
                            "incomplete"
                            if len(measurement_failures) > failure_start
                            else "synthetic" if manifest["kind"] == "offline_mock" else "completed"
                        ),
                        "measurement_failures": measurement_failures[failure_start:],
                        "choice_readings": choice_distributions,
                        "text_events": text_distributions,
                        "interpretation": "Choice: change in probability of reference choice (pp). "
                        "Text: argument-minus-control mean agreement with own prior full position; "
                        "not automatically movement toward opponent or lasting belief change.",
                    }
                    if config.roster == "mixed_family"
                    else {"status": "not_applicable", "reason": "same-family roster has no open-weight members"}
                ),
            }
        )
    return {
        "kind": manifest["kind"],
        "status": "completed_with_measurement_failures" if measurement_failures else "completed",
        "measurement_failures": measurement_failures,
        "counts": expected_counts(manifest),
        "questions": results,
        "charged_or_reserved_usd": journal.charged_usd,
        "prior_continuation_cost_usd": manifest.get("continuation", {}).get("prior_charged_or_reserved_usd", 0.0),
        "interpretation": "Engineering pilot only; no significance tests or main-study claims. "
        "Mock outputs, when selected, are synthetic test data, not model results.",
    }
