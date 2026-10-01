"""Missingness-aware reports. Failed measurements never become labels, zeros or ties."""

from __future__ import annotations

from collections import Counter

from . import prompts
from .models import AGREEMENT, OPEN_MODELS, TONES
from .planning import read_id, rng_for, route_from_plan
from .probabilities import RATING_LABELS, choice_change, distribution, own_position_difference
from .runner import expected_counts


def trajectory_outcomes(graph):
    route = route_from_plan(graph.plan)
    prefix = f"{graph.question.id}/{graph.config.roster}"
    outcomes = []
    for tone in TONES:
        for leaf in route.leaves:
            keys = [f"{prefix}/{tone}/debate/{n.id}" for n in route.path(leaf.id)]
            failed = [key for key in keys if key in graph.failed]
            blocked = [key for key in keys if key in graph.blocked]
            status = (
                "success"
                if all(key in graph.values for key in keys)
                else "failed"
                if failed
                else "blocked"
                if blocked
                else "pending"
            )
            outcomes.append(
                {
                    "question_id": graph.question.id,
                    "roster": graph.config.roster,
                    "tone": tone,
                    "trajectory_id": leaf.id,
                    "status": status,
                    "node_keys": keys,
                    "completed_nodes": sum(key in graph.values for key in keys),
                    "failed_node_keys": failed,
                    "blocked_node_keys": blocked,
                    "failure_causes": sorted(set(failed).union(*(graph.blocked[k]["causes"] for k in blocked))),
                }
            )
    return outcomes


def outcome_summary(outcomes):
    counts = Counter(item["status"] for item in outcomes)
    return {"planned": len(outcomes), **{s: counts[s] for s in ("success", "failed", "blocked", "pending")}}


def task_support(graph):
    support = {}
    for key, task in graph.tasks.items():
        counts = support.setdefault(
            task.purpose, dict.fromkeys(("planned", "completed", "failed", "blocked", "pending"), 0)
        )
        status = (
            "completed"
            if key in graph.values
            else "failed"
            if key in graph.failed
            else "blocked"
            if key in graph.blocked
            else "pending"
        )
        counts["planned"] += 1
        counts[status] += 1
    return support


def partial_report(graph, journal, token_count):
    """Build the usual data views from valid outputs only, without replaying missing requests."""
    question, config, plan = graph.question, graph.config, graph.plan
    prefix = f"{question.id}/{config.roster}"
    route = route_from_plan(plan)

    def value(suffix):
        return graph.values.get(f"{prefix}/{suffix}")

    def missing(suffix):
        key = f"{prefix}/{suffix}"
        return {
            "status": "missing",
            "key": key,
            "failure": graph.failed.get(key) or graph.blocked.get(key) or {"status": "pending"},
        }

    initials = {m: value(f"initial/{m}")["position"] for m in range(3) if value(f"initial/{m}") is not None}
    replies = {
        tone: {n.id: value(f"{tone}/debate/{n.id}") for n in route.nodes if value(f"{tone}/debate/{n.id}") is not None}
        for tone in TONES
    }
    positions = {r["id"]: value(f"C/{r['id']}") for r in plan["readings"] if value(f"C/{r['id']}") is not None}
    choices = {
        rid: distribution(v["_readout"], question.labels, missing_as_zero=True)
        for rid, v in positions.items()
        if "_readout" in v
    }
    text_events = {}
    for event in plan["events"]:
        model = config.members[event["member"]]
        if model not in OPEN_MODELS:
            continue
        arms = {}
        for arm in ("argument", "control"):
            suffix = f"Dtext/{event['id']}/{arm}"
            reading = value(suffix)
            arms[arm] = (
                {
                    **distribution(reading["_readout"], RATING_LABELS, missing_as_zero=True),
                    "sampled_rating": reading["rating"],
                }
                if reading is not None
                else missing(suffix)
            )
        complete = all("probabilities" in arm for arm in arms.values())
        entry = {
            **arms,
            "reference_reading": event["previous_reading"],
            "fixed_full_position": positions.get(event["previous_reading"], {}).get("position"),
            "status": "completed" if complete else "incomplete",
            "mean_own_agreement_argument_minus_control": own_position_difference(arms["argument"], arms["control"])
            if complete
            else None,
        }
        node = route.get(event["node_id"])
        peer = replies[event["tone"]].get(node.parent, {}).get("reply") if node.parent else initials.get(node.sender)
        if peer is not None:
            n = token_count(model, peer)
            repeats = max(1, round(n / token_count(model, prompts.FILLER_SENTENCE)))
            entry.update(
                argument_tokens=n,
                filler_repetitions=repeats,
                control_tokens=token_count(model, " ".join([prompts.FILLER_SENTENCE] * repeats)),
            )
        text_events[event["id"]] = entry

    def changed(before, after):
        a, b = positions.get(before, {}).get("choice"), positions.get(after, {}).get("choice")
        return a != b if a is not None and b is not None else None

    def probability_change(before, after):
        return (
            choice_change(choices[before], choices[after], positions[before]["choice"])
            if before in choices and after in choices
            else None
        )

    events = [
        {
            **e,
            "A": replies[e["tone"]].get(e["node_id"], {}).get("agreement"),
            "B": value(f"B/{e['id']}"),
            "C_choice_adjacent_changed": changed(e["previous_reading"], e["current_reading"]),
            "C_choice_initial_changed": changed(e["initial_reading"], e["current_reading"]),
            "C_text_adjacent": value(f"Cjudge/{e['adjacent_pair']}"),
            "C_text_final": value(f"Cjudge/{e['final_pair']}"),
            "D_choice_adjacent_pp": probability_change(e["previous_reading"], e["current_reading"]),
            "D_choice_initial_pp": probability_change(e["initial_reading"], e["current_reading"]),
            "D_text": text_events.get(e["id"]),
        }
        for e in plan["events"]
    ]
    trajectories, a_counts = {}, {}
    for tone in TONES:
        counts = Counter(r["agreement"] for r in replies[tone].values())
        a_counts[tone] = {
            "n_replies": len(replies[tone]),
            "planned_replies": len(route.nodes),
            "missing_replies": len(route.nodes) - len(replies[tone]),
            "unreported": counts[None],
            "counts": {label: counts[label] for label in AGREEMENT},
        }
        trajectories[tone] = {}
        for leaf in route.leaves:
            by_member = {str(m): [] for m in range(3)}
            for member in range(3):
                rid = f"initial/{member}"
                if rid in positions:
                    by_member[str(member)].append({"time_index": 0, "T": 0, "reading_id": rid})
            for node in route.path(leaf.id):
                rid = read_id(tone, node, node.receiver)
                if rid in positions:
                    by_member[str(node.receiver)].append(
                        {"time_index": route.participation(node.id), "T": node.depth, "reading_id": rid}
                    )
            trajectories[tone][leaf.id] = by_member

    preferences = {}
    if config.judge_model:
        for tone in TONES:
            first_left = bool(
                rng_for(config.seed, question.fingerprint, config.roster, tone, "answer_order").randrange(2)
            )
            orders, votes = [], []
            for order in range(2):
                left = first_left if order == 0 else not first_left
                suffix = f"E/{tone}/order{order}"
                verdict = value(suffix)
                orders.append({"debate_side": "left" if left else "right", **(verdict or missing(suffix))})
                if verdict is not None:
                    votes.append(verdict["preference"] == ("left" if left else "right"))
            preferences[tone] = {
                "orders": orders,
                "status": "completed" if len(votes) == 2 else "missing",
                "debate_score": sum(votes) / 2 if len(votes) == 2 else None,
                "order_inconsistent": votes[0] != votes[1] if len(votes) == 2 else None,
            }
    missing_tasks = {**graph.failed, **graph.blocked}
    result = {
        "question_id": question.id,
        "question_fingerprint": question.fingerprint,
        "initial_answers": initials,
        "formal_replies": replies,
        "A": a_counts,
        "C_readings": positions,
        "positions": trajectories,
        "C_text_pairs": [{**p, "judgment": value(f"Cjudge/{p['id']}")} for p in plan["text_pairs"]],
        "sampled_events": events,
        "E": {
            "baseline": (value("E/baseline") or {}).get("answer"),
            "debated": {t: (value(f"E/{t}/synthesis") or {}).get("answer") for t in TONES},
            "preferences": preferences,
        },
        "D": {
            "status": "incomplete"
            if any(v["purpose"] in ("position", "d_text") and v["model"] in OPEN_MODELS for v in missing_tasks.values())
            else "completed",
            "choice_readings": choices,
            "text_events": text_events,
        },
    }
    return {
        "status": "completed_with_failures",
        "kind": graph.manifest["kind"],
        "questions": [result],
        "counts": expected_counts({**graph.manifest, "plans": [plan]}),
        "task_failures": graph.failed,
        "blocked_tasks": graph.blocked,
        "task_support": task_support(graph),
        "trajectory_outcomes": trajectory_outcomes(graph),
        "logical_task_outputs": graph.values,
        "missingness_note": "Null/missing results are not agreement, no-change, zero or tie. Retained prefixes are partial observations; use task support and trajectory outcomes for denominators.",
        "charged_or_reserved_usd": journal.charged_usd,
        "prior_continuation_cost_usd": graph.manifest.get("continuation", {}).get("prior_charged_or_reserved_usd", 0),
    }
