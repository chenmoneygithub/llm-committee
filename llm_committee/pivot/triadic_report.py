"""Offline A/B/E report: three-member trees, with explicit denominators and full cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .forced_feedback_report import esc, rate, table
from .models import TONES, PilotConfig
from .quality_report import KIND_NAMES, calibration_case, construction_valid, preference_text
from .quality_study import CALIBRATION_KINDS
from .results_report import CSS, ci_cell, question_estimate
from .strong_agreement import AGREEMENT, AGREEMENT_DEFINITIONS
from .study import atomic_json
from .triadic_study import ARMS, VERSION, route_for


def label(value):
    return value.replace("_", " ").capitalize() if value else "Not reported"


def collect(records):
    formal = []
    for q in records:
        for arm, branch in q["branches"].items():
            for n in q["route"]:
                if arm == "alternate" and n["depth"] <= 2:
                    continue
                if reply := branch["replies"].get(n["id"]):
                    formal.append(
                        {
                            "question_id": q["question_id"],
                            "assignment": arm,
                            "member": n["receiver"],
                            "node_id": n["id"],
                            "tone": branch["tones"][n["id"]],
                            **reply,
                        }
                    )
    return formal


def summarize(records):
    formal = collect(records)
    b = [{"question_id": q["question_id"], **r} for q in records for r in q["B"]]
    endpoints = [{"question_id": q["question_id"], **r} for q in records for r in q["E1"]]
    quality = [{"question_id": q["question_id"], **r} for q in records for r in q["E2"]]
    e1 = []
    for member in range(3):
        group = [r for r in endpoints if r["member"] == member]
        choices = [r for r in group if r["choice_changed"] is not None]
        texts = [r for r in group if r["text_judgment"] and r["text_judgment"]["label"] != "unjudgeable"]
        counts = Counter(r["text_judgment"]["label"] for r in texts)
        e1.append(
            {
                "member": member,
                "planned_endpoints": len(group),
                "participated": sum(r["participated"] for r in group),
                "choice_compared": len(choices),
                "option_changed": sum(r["choice_changed"] for r in choices),
                "text_sampled": sum(r["text_sampled"] for r in group),
                "text_compared": len(texts),
                "reasons_adjusted": counts["adjusted"],
                "conclusion_changed": counts["conclusion_changed"],
            }
        )
    e2 = []
    for arm in ARMS:
        rows = [r for r in quality if r["assignment"] == arm]
        complete = [r for r in rows if r["debate_score"] is not None]
        counts = Counter(r["debate_score"] for r in complete)
        e2.append(
            {
                "assignment": arm,
                "planned": len(rows),
                "completed": len(complete),
                "debate_both": counts[1],
                "baseline_both": counts[0],
                "inconsistent": counts[0.5],
                "estimate": question_estimate(rows, "debate_score"),
                "baseline_words": mean(r["baseline_words"] for r in complete) if complete else None,
                "debate_words": mean(r["debate_words"] for r in complete) if complete else None,
                "max_word_gap": max((abs(r["debate_words"] - r["baseline_words"]) for r in complete), default=None),
            }
        )
    calibrations = []
    for kind in CALIBRATION_KINDS:
        for checked_only in (False, True):
            cases = [q["calibration"] for q in records if q["calibration"]]
            selected = [c for c in cases if not checked_only or construction_valid(c, kind)]
            scores = [
                c["variants"][kind]["target_score"] for c in selected if c["variants"][kind]["target_score"] is not None
            ]
            counts = Counter(scores)
            calibrations.append(
                {
                    "kind": kind,
                    "checked_only": checked_only,
                    "selected": len(selected),
                    "judged": len(scores),
                    "construction_passed": sum(construction_valid(c, kind) is True for c in selected),
                    "source_both": counts[1],
                    "variant_both": counts[0],
                    "inconsistent": counts[0.5],
                    "source_wins": mean(scores) if scores else None,
                }
            )
    return {
        "formal": formal,
        "B": b,
        "E1_endpoints": endpoints,
        "E1": e1,
        "E2_rows": quality,
        "E2": e2,
        "calibration_summary": calibrations,
    }


def full_case(q, members):
    route = route_for(q)
    options = table(["Option", "Text"], [[chr(65 + i), text] for i, text in enumerate(q["question"]["options"])])
    body = [f"<h3>{esc(q['question']['text'])}</h3>", options, "<h4>Initial independent answers</h4>"]
    for member, initial in q["initial_positions"].items():
        body.append(
            f"<p><b>{esc(members[int(member)])} · {esc(initial['choice'])}</b><br>{esc(initial['position'])}</p>"
        )
    body.append("<details><summary>Frozen routing and tone assignments</summary>")
    body.append(
        table(
            ["Node", "Replies to", "Sender → receiver", "T", "Original tone", "Alternate tone"],
            [
                [
                    n.id,
                    n.parent or f"initial/{n.sender}",
                    f"{n.sender} → {n.receiver}",
                    n.depth,
                    q["branches"]["original"]["tones"][n.id],
                    q["branches"]["alternate"]["tones"][n.id],
                ]
                for n in route.nodes
            ],
        )
    )
    body.append(
        table(
            ["Terminal path", "Member sequence"],
            [
                [
                    leaf.id,
                    " → ".join([str(route.path(leaf.id)[0].sender), *[str(n.receiver) for n in route.path(leaf.id)]]),
                ]
                for leaf in route.leaves
            ],
        )
    )
    body.append("</details>")
    for heading, arm, shared in (
        ("Shared T1/T2", "original", True),
        ("Original continuation: T3–T5", "original", False),
        ("Alternate continuation: T3–T5", "alternate", False),
    ):
        body.append(f"<details><summary>{heading}</summary>")
        for n in route.nodes:
            if (n.depth <= 2) != shared:
                continue
            r = q["branches"][arm]["replies"].get(n.id)
            if not r:
                body.append(f"<p>{n.id}: unavailable</p>")
                continue
            body.append(
                f"<article><h4>{n.id} · {esc(members[n.receiver])} replies to {esc(n.parent or f'initial/{n.sender}')} · {esc(q['branches'][arm]['tones'][n.id])}</h4><p><b>Public reply · {label(r['agreement'])}</b><br>{esc(r['reply'])}</p><p><b>Recorded own position · option {esc(r['choice'])}</b><br>{esc(r['position'])}</p></article>"
            )
        body.append("</details>")
    body.append("<details><summary>B: sampled text judgments</summary>")
    body.append(
        table(
            ["Continuation / node", "Self-label", "Judge label", "Evidence"],
            [
                [
                    f"{r['assignment']} / {r['node_id']}",
                    label(r["self_label"]),
                    label((r["judgment"] or {}).get("label")),
                    (r["judgment"] or {}).get("evidence", "Unavailable"),
                ]
                for r in q["B"]
            ],
        )
    )
    body.append("</details><details><summary>E1: sampled initial → last-participation text judgments</summary>")
    body.append(
        table(
            ["Continuation / path", "Member", "Last node", "Text change", "Evidence"],
            [
                [
                    f"{r['assignment']} / {r['leaf']}",
                    members[r["member"]],
                    r["last_node"],
                    label((r["text_judgment"] or {}).get("label")),
                    (r["text_judgment"] or {}).get("evidence", "Unavailable"),
                ]
                for r in q["E1"]
                if r["text_sampled"]
            ],
        )
    )
    body.append("</details><details><summary>E2: final answers and both-order judgments</summary>")
    base = q["E2"][0]["baseline"]
    body.append(f"<h4>Shared no-debate answer</h4><p>{esc(base['answer'] if base else 'Unavailable')}</p>")
    for r in q["E2"]:
        body.append(
            f"<h4>{r['assignment'].title()} debate answer</h4><p>{esc(r['debated']['answer'] if r['debated'] else 'Unavailable')}</p>"
            + preference_text(r["preferences"], "Debate answer")
        )
    body.append("</details>")
    return f"<details class='question-case'><summary>{esc(q['question_id'])} — {esc(q['question']['text'])}</summary>{''.join(body)}</details>"


def publish(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] != VERSION or report["status"] not in (
        "completed",
        "completed_with_failures",
        "preflight_completed",
    ):
        raise ValueError("Need a completed phase of the triadic study")
    records = [
        json.loads((source / "questions" / f"{p['question_id']}.json").read_text())
        for p in manifest["plans"]
        if (source / "questions" / f"{p['question_id']}.json").exists()
    ]
    cfg = PilotConfig(**manifest["config"])
    members = cfg.members
    summary = {
        "protocol_version": VERSION,
        "source_directory": str(source),
        "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "run_report": report,
        **summarize(records),
    }
    formal, b = summary["formal"], summary["B"]
    a_rows = []
    for tone in TONES:
        group = [r for r in formal if r["tone"] == tone]
        counts = Counter(r["agreement"] for r in group)
        a_rows.append([tone, len(group), *[rate(counts[k], len(group)) for k in AGREEMENT]])
    b_rows = []
    b_excluded = sum(not r["judgment"] or r["judgment"]["label"] not in AGREEMENT for r in b)
    for current in AGREEMENT:
        group = [r for r in b if r["self_label"] == current]
        counts = Counter((r["judgment"] or {}).get("label") for r in group)
        n = sum(counts[k] for k in AGREEMENT)
        b_rows.append([label(current), len(group), *[rate(counts[k], n) for k in AGREEMENT]])
    e1_rows = [
        [
            members[r["member"]],
            f"{r['participated']}/{r['planned_endpoints']}",
            rate(r["option_changed"], r["choice_compared"]),
            rate(r["reasons_adjusted"], r["text_compared"]),
            rate(r["conclusion_changed"], r["text_compared"]),
        ]
        for r in summary["E1"]
    ]
    e2_rows = [
        [
            r["assignment"].title(),
            f"{r['completed']}/{r['planned']}",
            rate(r["debate_both"], r["completed"]),
            rate(r["baseline_both"], r["completed"]),
            rate(r["inconsistent"], r["completed"]),
            ci_cell(r["estimate"], scale=100, digits=1, signed=False),
        ]
        for r in summary["E2"]
    ]
    length_rows = [
        [
            r["assignment"].title(),
            f"{r['baseline_words']:.1f}" if r["baseline_words"] is not None else "—",
            f"{r['debate_words']:.1f}" if r["debate_words"] is not None else "—",
            r["max_word_gap"],
        ]
        for r in summary["E2"]
    ]

    def calibration_table(checked):
        return table(
            [
                "Diagnostic",
                "Judged / sampled",
                "Construction passed",
                "Source wins both orders",
                "Variant wins both orders",
                "Order-inconsistent",
                "Source wins (%)",
            ],
            [
                [
                    KIND_NAMES[r["kind"]],
                    f"{r['judged']}/{r['selected']}",
                    r["construction_passed"],
                    r["source_both"],
                    r["variant_both"],
                    r["inconsistent"],
                    f"{100 * r['source_wins']:.1f}" if r["source_wins"] is not None else "—",
                ]
                for r in summary["calibration_summary"]
                if r["checked_only"] == checked
            ],
        )

    calibration_cases = []
    for q in records:
        if c := q["calibration"]:
            calibration_cases.append(
                calibration_case({**q, "calibration": {**c, "base": c["source"], "pair": "whole three-member tree"}})
            )
    triadic_paths = sum(len(set(t["member_sequence"])) == 3 for q in records for t in q["trajectories"])
    path_count = sum(len(q["trajectories"]) for q in records)
    mock = (
        '<p class="warning">OFFLINE MOCK — synthetic fixtures, not scientific findings.</p>'
        if manifest["kind"] == "offline_mock"
        else ""
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Three-member public-history debate · A/B/E</title><style>{CSS}
article{{border-left:3px solid #cbd5e1;padding:0 16px}}details{{margin:14px 0;padding:12px;border:1px solid #ddd}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}td,th{{vertical-align:top}}
</style></head><body><main><h1>Three-member debate · turn-level tone · A/B/E</h1>{mock}
<nav><a href="#a">A · Self-labels</a> · <a href="#b">B · Text check</a> · <a href="#e1">E1 · Member outcomes</a> · <a href="#e2">E2 · Final-answer quality</a> · <a href="#cases">Full debates</a></nav>
<p>{len(records)}/{len(manifest["plans"])} questions · {esc(" / ".join(members))}. Archived question-specific routing is unchanged: two starting exchanges, six terminal paths, five replies per path. Two turn-tone assignments share T1/T2 and differ at T3–T5. Shared nodes count once; this report contains {len(formal)} unique replies and {path_count} branch paths.</p>
<p>Each reasoning-enabled turn jointly generates reply, agreement, choice and position. Only public replies enter the branch history; updated private choice/position fields are not reinjected. Original independent answers remain visible according to the same branch-context rule as the dyadic pilot. Historical labels, tone instructions and judgments are not transmitted.</p>
<p><b>Local C/D are intentionally not collected.</b> E1 below is initial-to-final outcome analysis, not a pooled turn-level C/D analysis. Some random paths use only two members: {triadic_paths}/{path_count} paths include all three member identities (including the initial sender). No path was selected or removed based on its outcomes.</p>
<p>This is a three-member extension, not an isolated committee-size comparison: the tree contains more replies than the four-turn dyads. The two continuations share initial answers/prefixes and are not independent replications. All uncertainty is based on questions, not treating branch endpoints as new questions.</p>
<p><a href="turn-tone-dyadic-public-history-2026-09-27.html">Separate two-member reference report</a> — results are not pooled.</p>
<details><summary>Four agreement labels</summary>{table(["Label", "Meaning"], [[label(k), AGREEMENT_DEFINITIONS[k]] for k in AGREEMENT])}</details>
<section id="a"><h2>A · What label does each reply report?</h2><p>Agreement is with the incoming peer message, not the original survey question. Rows use the current turn's tone, not a global tone condition. All generated turns, including T1, are included; shared replies count once.</p>{table(["Current tone", "Replies", *[label(k) for k in AGREEMENT]], a_rows)}<p>{sum(r["agreement"] is None for r in formal)} replies have no reported agreement direction.</p></section>
<section id="b"><h2>B · Does the reply text support its self-label?</h2><p>One Gemini 3.8 Flash judge reads the question, incoming message and reply, without the self-label or tone instruction. Each row asks: among replies self-labeled X, what label does the judge assign from the text? Eight base nodes per question were presampled at T2–T5; their alternate suffixes are also judged, with shared prefixes judged once.</p>{table(["Self-label", "Sampled replies", *[f"Gemini: {label(k)}" for k in AGREEMENT]], b_rows)}<p>{b_excluded} missing or outside-rubric judgments are excluded from the cell denominators. Agreement with one model judge is not human-validated accuracy.</p></section>
<section id="e1"><h2>E1 · Members' initial → last-participation positions</h2><p>For each member on each terminal path, compare its original choice/full position with its last formal participation. Last participation may precede the path's final turn; no extra end-of-path re-answer is requested. Do not attribute the net change to the final reply's label or tone.</p><p>Option changes cover all comparable participating-member endpoints. Text changes use two presampled leaves per question in both continuations; identical endpoint pairs share one judgment. Members who never replied on a path retain their initial-only history and are excluded from change denominators, not counted as observed unchanged outcomes.</p>{table(["Member", "Participating / planned endpoints", "Option changed", "Reasons adjusted", "Conclusion changed"], e1_rows)}<p>Reasons adjusted means changed reasoning/qualifications without a changed main conclusion; conclusion changed means a changed substantive answer. These text categories are disjoint. Option and text counts must not be added. Text cells have their own sampled denominators; these repeated endpoints are not independent questions.</p></section>
<section id="e2"><h2>E2 · Does the full discussion improve the chairman's answer?</h2><p><b>No debate:</b> Terra synthesizes the three original independent answers. <b>Debate:</b> the same Terra chairman receives those exact answers plus the complete six-leaf public discussion tree, with shared nodes listed once and explicit parent links. Each question has one baseline and two debate syntheses. Private positions, self-labels and tone metadata are absent.</p><p>Both sides must contain 190–210 words. One Gemini 3.8 Flash judge compares the answers blindly in both presentation orders. Order-inconsistent means one decision favors each answer; it is not an explicit tie.</p>{table(["Continuation", "Complete / planned pairs", "Debate in both orders", "Baseline in both orders", "Order-inconsistent", "Debate wins (%) [95% interval]"], e2_rows)}<p><b>Debate wins (%)</b> = decisions choosing debate ÷ all decisions × 100. A 20-pair row has 40 decisions. This is a judge preference rate, not a percentage quality improvement. Intervals are question-level percentile bootstraps (10,000 resamples), not multiplicity-adjusted claims.</p>
<h3>Output-length check</h3>{table(["Continuation", "Baseline mean words", "Debate mean words", "Largest paired word gap"], length_rows)}<p>The common narrow budget reduces length differences; it does not guarantee freedom from residual style or length preferences.</p>
<h3>Small judge diagnostics</h3><p>Twelve questions were selected before generation. Each contributes an expanded-wording variant, a version with one intended contradictory reason, and an off-topic variant. All are judged in both orders. The same model separately checks whether each construction meets its intended specification; this is not independent human validation.</p>{calibration_table(False)}<p>Source wins counts decisions preferring the unmodified answer. The expanded-wording probe has no assumed correct winner; it tests presentation sensitivity only when substantive content is preserved. Damage detection does not certify sensitivity to subtle natural quality differences.</p><details><summary>Constructions passing the automatic check only</summary>{calibration_table(True)}</details>{"".join(calibration_cases)}</section>
<section id="cases"><h2>Qualitative review · complete questions, trees and outputs</h2><p>Member IDs 0/1/2 correspond to {esc(" / ".join(members))}. Sibling nodes are separate branches, not a concatenated conversation. Shared T1/T2 are shown once per question.</p>{"".join(full_case(q, members) for q in records)}</section>
<details><summary>Execution and failures</summary><pre>{esc(json.dumps(report, ensure_ascii=False, indent=2))}</pre></details></main></body></html>"""
    output.write_text(document)
    atomic_json(output.with_suffix(".json"), summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(publish(args.source, args.output))
