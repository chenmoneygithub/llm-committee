"""Offline, question-aware results tables for a completed mixed-committee study.

No providers, credentials, journal mutation, generation, or manuscript editing.
Run: python -m llm_committee.pivot.results_report RUN_DIRECTORY OUTPUT.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .agreement import LEGACY_AGREEMENT as AGREEMENT
from .agreement import LEGACY_PROMPT_VERSION
from .models import TEXT_SHIFT, TONES
from .probabilities import choice_change, own_position_difference

MODEL_NAMES = ("GPT-5.6 Terra", "Qwen3.8-27B", "Inkling")
LABEL_NAMES = ("Fully agreed", "Partially agreed", "Partially disagreed", "Fully disagreed")
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260926


def question_estimate(records, key):
    """Average within question first; percentile bootstrap whole question means.

    Unavailable values are excluded, not replaced with zeros. This describes the
    presampled events, not an inverse-probability estimate of all debate turns.
    """
    by_question = defaultdict(list)
    for row in records:
        if row.get(key) is not None:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError("Nonfinite measurement")
            by_question[row["question_id"]].append(value)
    values = np.array([mean(by_question[q]) for q in sorted(by_question)])
    if not len(values):
        return {"n": 0, "questions": 0, "mean": None, "ci": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    return {
        "n": sum(map(len, by_question.values())),
        "questions": len(values),
        "mean": float(values.mean()),
        "ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])] if len(values) > 1 else None,
    }


def categories(rows, key, labels):
    counts = Counter(row.get(key) for row in rows)
    return {
        "n": len(rows),
        "counts": {label: counts[label] for label in labels},
        "other": sum(n for label, n in counts.items() if label not in labels),
    }


def changes(rows, key):
    counts = Counter(row[key] for row in rows)
    return {"n": len(rows), "valid": counts[True] + counts[False], "changed": counts[True], "unavailable": counts[None]}


def validate_question(q, plan, question, *, agreement_labels=AGREEMENT):
    """Fail closed on mismatched inputs, duplicate events, or inconsistent saved metrics."""
    if q["question_id"] != plan["question_id"] or q["question_fingerprint"] != plan["question_fingerprint"]:
        raise ValueError("Question/plan mismatch")
    nodes = {node["id"]: node for node in plan["route"]}
    for tone in TONES:
        replies = q["formal_replies"][tone]
        assert set(replies) == set(nodes)
        counts = Counter(r["agreement"] for r in replies.values())
        assert set(counts).issubset((*agreement_labels, None))
        assert q["A"][tone]["counts"] == {label: counts[label] for label in agreement_labels}
        assert q["A"][tone]["unreported"] == counts[None]
        assert q["A"][tone]["n_replies"] == len(nodes)
    events = q["sampled_events"]
    assert len(events) == len({e["id"] for e in events}) == 8
    assert {e["id"] for e in events} == {e["id"] for e in plan["events"]}
    pairs = {p["id"]: p for p in q["C_text_pairs"]}
    assert len(pairs) == len(q["C_text_pairs"])
    assert set(pairs) == {p["id"] for p in plan["text_pairs"]}
    readings = q["C_readings"]
    assert set(readings) == {r["id"] for r in plan["readings"]}
    distributions = q["D"]["choice_readings"]
    labels = {chr(65 + i) for i in range(len(question["options"]))}
    for distribution in list(distributions.values()) + [
        e["D_text"][arm] for e in events if e["D_text"] for arm in ("argument", "control")
    ]:
        assert math.isclose(sum(distribution["probabilities"].values()), 1, abs_tol=1e-9)
        assert all(0 <= x <= 1 for x in distribution["probabilities"].values())
        assert distribution["read_temperature"] == 1
        assert distribution["source"] == "sample_topk"
    for e in events:
        assert e["A"] == q["formal_replies"][e["tone"]][e["node_id"]]["agreement"]
        assert e["B"]["label"] in (*agreement_labels, "no_position", "unjudgeable")
        for kind, ref in (("adjacent", "previous_reading"), ("initial", "initial_reading")):
            before, after = readings[e[ref]]["choice"], readings[e["current_reading"]]["choice"]
            assert before is None or before in labels
            assert after is None or after in labels
            expected = before != after if before is not None and after is not None else None
            assert e[f"C_choice_{kind}_changed"] == expected
            if e["member"] in (1, 2):
                expected_d = choice_change(distributions[e[ref]], distributions[e["current_reading"]], before)
                saved = e[f"D_choice_{kind}_pp"]
                assert (saved is expected_d is None) or math.isclose(saved, expected_d, abs_tol=1e-10)
        assert e["C_text_adjacent"] == pairs[e["adjacent_pair"]]["judgment"]
        assert e["C_text_adjacent"]["label"] in TEXT_SHIFT
        if e["final_pair"]:
            assert e["C_text_final"] == pairs[e["final_pair"]]["judgment"]
            assert e["final_trajectories"]
        else:
            assert e["C_text_final"] is None
        if e["D_text"]:
            d = e["D_text"]
            assert d["fixed_full_position"] == readings[e["previous_reading"]]["position"]
            assert math.isclose(
                d["mean_own_agreement_argument_minus_control"],
                own_position_difference(d["argument"], d["control"]),
                abs_tol=1e-10,
            )
    for result in q["E"]["preferences"].values():
        assert len(result["orders"]) == 2
        assert {o["debate_side"] for o in result["orders"]} == {"left", "right"}
        votes = [o["preference"] == o["debate_side"] for o in result["orders"]]
        assert result["debate_score"] == mean(votes)
        assert result["order_inconsistent"] == (votes[0] != votes[1])


def analyze(source: Path, *, agreement_labels=AGREEMENT, label_names=LABEL_NAMES, prompt_version=LEGACY_PROMPT_VERSION):
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    status = json.loads((source / "report.json").read_text())
    if (
        status["status"] != "completed"
        or manifest["kind"] != "paid_main_study"
        or manifest["config"]["roster"] != "mixed_family"
        or len(manifest["questions"]) != 60
    ):
        raise ValueError("This report requires the completed, real 60-question mixed-family study")
    if manifest["prompt_version"] != prompt_version:
        raise ValueError("Report protocol mismatch; do not relabel data from another rubric")
    plans = {p["question_id"]: p for p in manifest["plans"]}
    bank = {q["id"]: q for q in manifest["questions"]}
    questions, file_hashes = [], {}
    assert set(plans) == set(bank)
    for qid in bank:
        path = source / "questions" / f"{qid}.json"
        raw = path.read_bytes()
        file_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        assert record["status"] == "completed"
        assert len(record["questions"]) == 1
        q = record["questions"][0]
        validate_question(q, plans[qid], bank[qid], agreement_labels=agreement_labels)
        questions.append(q)

    replies, events, quality, pairs, readouts = [], [], [], [], []
    for q in questions:
        qid = q["question_id"]
        nodes = {n["id"]: n for n in plans[qid]["route"]}
        reading_meta = {r["id"]: r for r in plans[qid]["readings"]}
        for tone in TONES:
            replies.extend(
                {
                    "question_id": qid,
                    "tone": tone,
                    "T": nodes[n]["depth"],
                    "member": nodes[n]["receiver"],
                    "A": r["agreement"],
                }
                for n, r in q["formal_replies"][tone].items()
            )
            quality.append(
                {
                    "question_id": qid,
                    "tone": tone,
                    **q["E"]["preferences"][tone],
                    "baseline_words": len(q["E"]["baseline"].split()),
                    "debate_words": len(q["E"]["debated"][tone].split()),
                }
            )
        for e in q["sampled_events"]:
            row = {
                "question_id": qid,
                **e,
                "B_label": e["B"]["label"],
                "text_adjacent": e["C_text_adjacent"]["label"],
                "text_final": e["C_text_final"]["label"] if e["C_text_final"] else None,
                "D_text_delta": e["D_text"]["mean_own_agreement_argument_minus_control"] if e["D_text"] else None,
            }
            if e["D_text"]:
                for arm in ("argument", "control"):
                    dist = e["D_text"][arm]
                    row[f"D_text_{arm}"] = sum((i + 1) * dist["probabilities"][k] for i, k in enumerate("ABCDEFG"))
                    readouts.append({"member": e["member"], "kind": "Own-position rating", **dist})
            events.append(row)
        pairs.extend({"question_id": qid, **p} for p in q["C_text_pairs"])
        for rid, dist in q["D"]["choice_readings"].items():
            readouts.append({"member": reading_meta[rid]["member"], "kind": "Survey choice", **dist})

    observed = {
        "questions": len(questions),
        "debates": len(quality),
        "formal_replies": len(replies),
        "sampled_events": len(events),
        "C_readings": sum(len(q["C_readings"]) for q in questions),
        "C_judgments": len(pairs),
        "B_judgments": len(events),
        "D_choice_readings": sum(r["kind"] == "Survey choice" for r in readouts),
        "D_text_pairs": sum(e["D_text"] is not None for e in events),
        "E_judgments": 2 * len(quality),
    }
    assert all(v == status["planned_counts"][k] for k, v in observed.items())
    assert status["fully_successful_questions"] == len(questions)
    assert status["trajectories"]["success"] == 18 * len(questions)

    a_rows = []
    for scope in ("First replies (T = 1)", "All five depths"):
        for tone in TONES:
            chosen = [r for r in replies if r["tone"] == tone and (scope == "All five depths" or r["T"] == 1)]
            a_rows.append({"scope": scope, "tone": tone, **categories(chosen, "A", agreement_labels)})
    model_a = [
        {
            "model": MODEL_NAMES[m],
            "tone": tone,
            **categories([r for r in replies if r["member"] == m and r["tone"] == tone], "A", agreement_labels),
        }
        for m in range(3)
        for tone in TONES
    ]
    b_rows = [
        {"group": name, **categories(rows, "B_label", (*agreement_labels, "no_position", "unjudgeable"))}
        for name, rows in [("All sampled replies", events)]
        + [(MODEL_NAMES[m], [e for e in events if e["member"] == m]) for m in range(3)]
    ]
    b_tones = [
        {
            "group": tone.title(),
            **categories(
                [e for e in events if e["tone"] == tone], "B_label", (*agreement_labels, "no_position", "unjudgeable")
            ),
        }
        for tone in TONES
    ]
    ab = [e for e in events if e["A"] in agreement_labels and e["B_label"] in agreement_labels]
    ab_counts = Counter(
        "same"
        if e["A"] == e["B_label"]
        else "more_disagreeing"
        if agreement_labels.index(e["B_label"]) > agreement_labels.index(e["A"])
        else "less_disagreeing"
        for e in ab
    )
    ab_confusion = [
        [sum(e["A"] == a and e["B_label"] == b for e in ab) for b in agreement_labels] for a in agreement_labels
    ]
    final_events = [e for e in events if e["final_pair"]]

    def c_row(name, rows, choice_key="C_choice_adjacent_changed", text_key="text_adjacent"):
        return {
            "group": name,
            "questions": len({e["question_id"] for e in rows}),
            "choice": changes(rows, choice_key),
            "text": categories(rows, text_key, tuple(TEXT_SHIFT)),
            "choice_estimate": question_estimate(rows, choice_key),
        }

    c_main = [
        c_row("Previous own position → current", events),
        c_row("Initial → sampled last own position", final_events, "C_choice_initial_changed", "text_final"),
    ]
    c_models = [c_row(MODEL_NAMES[m], [e for e in events if e["member"] == m]) for m in range(3)]
    c_agreement = [
        c_row(name, [e for e in events if e["A"] == label])
        for name, label in zip(label_names, agreement_labels, strict=True)
    ]
    c_depth = [
        dict(
            c_row(f"T = {t}", [e for e in events if e["T"] == t]),
            initial_choice=changes([e for e in events if e["T"] == t], "C_choice_initial_changed"),
        )
        for t in range(1, 6)
    ]
    c_participation = [
        c_row(f"Participation {t}", [e for e in events if e["participation_index"] == t])
        for t in sorted({e["participation_index"] for e in events})
    ]
    d_rows, d_by_a = [], []
    for member in (1, 2):
        selected = [e for e in events if e["member"] == member]
        d_rows.append(
            {
                "model": MODEL_NAMES[member],
                "n": len(selected),
                **{
                    k: question_estimate(selected, k)
                    for k in (
                        "D_choice_adjacent_pp",
                        "D_choice_initial_pp",
                        "D_text_delta",
                        "D_text_argument",
                        "D_text_control",
                    )
                },
                "D_choice_final_pp": question_estimate([e for e in selected if e["final_pair"]], "D_choice_initial_pp"),
            }
        )
        for name, label in zip(label_names, agreement_labels, strict=True):
            subset = [e for e in selected if e["A"] == label]
            d_by_a.append(
                {
                    "model": MODEL_NAMES[member],
                    "A": name,
                    "n": len(subset),
                    **{k: question_estimate(subset, k) for k in ("D_choice_adjacent_pp", "D_text_delta")},
                }
            )

    def quality_row(name, rows):
        counts = Counter(r["debate_score"] for r in rows)
        return {
            "tone": name,
            "n": len(rows),
            "debate": counts[1],
            "baseline": counts[0],
            "split": counts[0.5],
            "estimate": question_estimate(rows, "debate_score"),
            "baseline_words": mean(r["baseline_words"] for r in rows),
            "debate_words": mean(r["debate_words"] for r in rows),
        }

    e_rows = [quality_row(tone.title(), [e for e in quality if e["tone"] == tone]) for tone in TONES]
    e_rows.append(quality_row("All tones", quality))
    a_contrast = []
    for q in questions:
        values = {tone: q["A"][tone]["counts"][agreement_labels[0]] / q["A"][tone]["n_replies"] for tone in TONES}
        a_contrast.append({"question_id": q["question_id"], "diff": 100 * (values["friendly"] - values["hostile"])})
    probability_audit = []
    for member in (1, 2):
        for kind in ("Survey choice", "Own-position rating"):
            rows = [r for r in readouts if r["member"] == member and r["kind"] == kind]
            masses = [r["candidate_mass"] for r in rows]
            probability_audit.append(
                {
                    "model": MODEL_NAMES[member],
                    "kind": kind,
                    "n": len(rows),
                    "zero_fill": sum(r["zero_fill_applied"] for r in rows),
                    "min_mass": min(masses),
                    "median_mass": float(np.median(masses)),
                    "candidate_mass_below_half": sum(x < 0.5 for x in masses),
                    "max_probability_ge_99": sum(max(r["probabilities"].values()) >= 0.99 for r in rows),
                }
            )
    summary = {
        "schema": "mixed_results_summary_v1",
        "source_directory": str(source),
        "completed_at_utc": status["updated_at_utc"],
        "counts": observed,
        "planned_counts": status["planned_counts"],
        "trajectories": status["trajectories"],
        "config": manifest["config"],
        "request_retries": status["request_retries"],
        "transport_retries": status["transport_retries"],
        "format_retries": status["format_retries"],
        "charged_or_reserved_usd": status["prior_charged_or_reserved_usd"] + status["new_charged_or_reserved_usd"],
        "final_backfill_usd": status["new_charged_or_reserved_usd"],
        "unselectable_readings": sum(r["choice"] is None for q in questions for r in q["C_readings"].values()),
        "A": a_rows,
        "A_by_model": model_a,
        "A_friendly_minus_hostile_pp": question_estimate(a_contrast, "diff"),
        "B": b_rows,
        "B_by_tone": b_tones,
        "AB": {"n": len(ab), "counts": dict(ab_counts), "matrix": ab_confusion},
        "C": c_main,
        "C_by_model": c_models,
        "C_by_self_label": c_agreement,
        "C_by_depth": c_depth,
        "C_by_participation": c_participation,
        "C_initial_to_current": changes(events, "C_choice_initial_changed"),
        "C_final_distinct_pairs": len({(e["question_id"], e["final_pair"]) for e in final_events}),
        "D": d_rows,
        "D_by_self_label": d_by_a,
        "probability_audit": probability_audit,
        "E": e_rows,
        "inference": {
            "resampling_unit": "question",
            "bootstrap_draws": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "interval": "pointwise percentile 95%; descriptive/post-collection, no multiplicity adjustment",
            "event_weighting": "sample-conditional, no inverse inclusion weighting",
        },
        "source_files_sha256": file_hashes,
        "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
    }
    assert summary["C_final_distinct_pairs"] == len(final_events)
    return summary, bank, questions, plans


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "—"


def fraction(n, d):
    return f"{n}/{d} ({pct(n, d)})" if d else "— (no eligible observations)"


def ci_cell(estimate, scale=1, digits=2, signed=True):
    if estimate["mean"] is None:
        return "—"
    spec = f"{'+' if signed else ''}.{digits}f"
    center = format(scale * estimate["mean"], spec)
    if estimate["ci"] is None:
        return center + " (one question; no interval)"
    low, high = [format(scale * x, spec) for x in estimate["ci"]]
    return f"{center} [{low}, {high}]"


def table(title, headers, rows, note=""):
    def esc(value):
        return html.escape(str(value))

    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f'<th scope="row">{esc(v)}</th>' if i == 0 else f"<td>{esc(v)}</td>" for i, v in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table><caption>{esc(title)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        + (f'<p class="table-note">{esc(note)}</p>' if note else "")
    )


def agreement_cells(row):
    return [row["n"], *[pct(row["counts"][k], row["n"]) for k in AGREEMENT]]


def c_cells(row):
    c, t = row["choice"], row["text"]
    return [
        row["group"],
        t["n"],
        fraction(c["changed"], c["valid"]),
        c["unavailable"],
        *[f"{t['counts'][k]} ({pct(t['counts'][k], t['n'])})" for k in ("unchanged", "adjusted", "conclusion_changed")],
        t["counts"]["unjudgeable"],
    ]


CSS = """
:root { color-scheme: light; --ink:#202c35; --muted:#586671; --line:#ced7dd; --accent:#126459; }
* { box-sizing:border-box; }
body { margin:0; background:#f2f5f6; color:var(--ink); font:16px/1.65 system-ui,-apple-system,sans-serif; }
main { max-width:1180px; margin:32px auto; padding:48px 52px; background:white; border:1px solid #dce3e7; }
h1 { font:700 38px/1.2 Georgia,serif; margin:10px 0 16px; max-width:900px; }
h2 { font:700 27px/1.3 Georgia,serif; margin:0 0 16px; }
h3 { font-size:18px; margin:24px 0 12px; }
p { margin:12px 0; } a { color:var(--accent); text-underline-offset:3px; }
.eyebrow { color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }
.muted,.table-note { color:var(--muted); } .table-note { font-size:13px; margin:8px 0 22px; }
.lede { font-size:18px; max-width:960px; } .status { padding:14px 18px; background:#eaf5ef; border-left:4px solid var(--accent); }
nav { display:flex; gap:12px 22px; flex-wrap:wrap; margin:24px 0; padding:14px 0; border-block:1px solid var(--line); }
section { margin-top:40px; padding-top:10px; scroll-margin-top:20px; }
.findings { padding:18px 24px; background:#f7f9fa; border:1px solid var(--line); }
.findings li { padding:5px 0; } .findings ul { margin:0; padding-left:22px; }
.table-wrap { width:100%; overflow-x:auto; margin-top:24px; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; line-height:1.45; }
caption { caption-side:top; text-align:left; font-size:15px; font-weight:650; margin-bottom:12px; }
thead { border-top:2px solid var(--ink); border-bottom:1px solid var(--ink); }
tbody { border-bottom:2px solid var(--ink); }
th,td { padding:10px 11px; text-align:right; vertical-align:top; }
th { font-weight:600; } td { white-space:normal; }
thead th { font-size:12px; } tbody th,thead th:first-child { text-align:left; }
tbody tr:nth-child(even) { background:#f7f9fa; }
tbody tr+tr { border-top:1px solid #e6ecef; }
details { border:1px solid var(--line); padding:12px 18px; margin:20px 0; }
summary { cursor:pointer; font-weight:600; } details .table-wrap { margin-top:16px; }
.formula { padding:15px 20px; font:17px/1.8 Georgia,serif; background:#f7f9fa; }
.callout { border-left:3px solid #ba8137; padding:4px 18px; background:#fffaf0; }
.example { padding:10px 18px; margin-top:18px; border-left:3px solid #9ebeb7; }
.example blockquote { margin:10px 0; font-size:14px; white-space:pre-wrap; }
code { font-size:12px; overflow-wrap:anywhere; } .inventory td { text-align:left; }
footer { border-top:1px solid var(--line); margin-top:40px; padding-top:18px; font-size:12px; color:var(--muted); }
@media(max-width:760px) { main { margin:0; padding:24px 18px; } h1 { font-size:29px; } th,td { padding:8px; } }
@media print { body { background:white; } main { margin:0; border:none; max-width:none; padding:0; }
nav { display:none; } .table-wrap { overflow:visible; } table { font-size:10px; } tr { break-inside:avoid; }
h2,h3,caption { break-after:avoid; } section { margin-top:22px; } a { color:inherit; } }
"""


def render(summary, bank, questions, plans):
    s = summary
    esc = html.escape
    n = s["counts"]
    a_all = s["A"][3:]
    c, cfinal = s["C"]
    eall = s["E"][-1]
    b = s["AB"]
    parts = [
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Mixed committee — 60-question results, September 26, 2026</title>"
        f"<style>{CSS}</style></head><body><main>",
        '<header><div class="eyebrow">GlobalOpinionQA · Completed run · 26 September 2026</div>'
        "<h1>What changed during committee debate?</h1>"
        '<p class="lede">Results from GPT-5.6 Terra, Qwen3.8-27B and Inkling on 60 questions, '
        "each discussed under friendly, neutral and hostile instructions.</p>"
        f'<p class="status"><strong>Complete:</strong> 60/60 questions · 180/180 debates · '
        f"1,080/1,080 planned paths · {n['formal_replies']:,} unique replies. "
        "All failed requests have been recovered; no technical task remains missing.</p>"
        '<p class="muted">Chairman: GPT-5.6 Terra. External B/C/E judge: Gemini 3.8 Flash. '
        "This is the new mixed-family experiment only, not the old paper’s dataset or its old Layer C/D definitions.</p>"
        '</header><nav aria-label="Report sections">'
        '<a href="#findings">Summary</a><a href="#a">A · Self-labels</a>'
        '<a href="#b">B · Reply text</a><a href="#c">C · Position change</a>'
        '<a href="#d">D · Probabilities</a><a href="#e">E · Final answers</a>'
        '<a href="#methods">Scope & accounting</a></nav>',
        '<section id="findings"><h2>Main observations</h2><div class="findings"><ul>',
        f"<li><strong>Tone changes reported agreement substantially.</strong> Full agreement is "
        f"{pct(a_all[0]['counts']['fully_agreed'], a_all[0]['n'])} under friendly instructions, "
        f"{pct(a_all[1]['counts']['fully_agreed'], a_all[1]['n'])} under neutral instructions and "
        f"{pct(a_all[2]['counts']['fully_agreed'], a_all[2]['n'])} under hostile instructions. "
        "These labels describe agreement with the incoming peer message, not a changed answer to the question.</li>",
        f"<li><strong>The text judge often sees more disagreement than the self-label reports.</strong> "
        f"On the same {b['n']} sampled replies, ratings match in {pct(b['counts'].get('same', 0), b['n'])}; "
        f"the judge is more disagreeing in {pct(b['counts'].get('more_disagreeing', 0), b['n'])} "
        f"and less disagreeing in {pct(b['counts'].get('less_disagreeing', 0), b['n'])}. "
        "This is one judge’s assessment, not validated ground truth.</li>",
        f"<li><strong>Most sampled positions remain stable.</strong> Relative to the member’s previous own position, "
        f"the selected option changes in {fraction(c['choice']['changed'], c['choice']['valid'])}; "
        f"the text judge finds {c['text']['counts']['adjusted']}/{c['text']['n']} adjustments "
        f"and {c['text']['counts']['conclusion_changed']}/{c['text']['n']} changed conclusions. "
        "These are separate signals; a reply can object to a peer while retaining its own position.</li>",
        f"<li><strong>The probability probe detects modest average reductions in endorsement of the prior text.</strong> "
        f"Argument minus filler is {s['D'][0]['D_text_delta']['mean']:+.3f} points for Qwen "
        f"and {s['D'][1]['D_text_delta']['mean']:+.3f} for Inkling on the 1–7 scale "
        "(question-weighted). This does not by itself establish movement toward the peer or lasting belief change.</li>",
        f"<li><strong>Final-answer preference is not uniformly neutral in this run.</strong> "
        f"The judge prefers the debate synthesis in both orders on {eall['debate']}/{eall['n']} pairs, "
        f"the no-debate synthesis on {eall['baseline']}/{eall['n']}, and changes its preference after reversal "
        f"on {eall['split']}/{eall['n']}. The hostile arm has the strongest observed preference for debate. "
        "The old paper’s all-tie result does not describe this new judge protocol.</li></ul></div>"
        "<p>Tables A and B follow the paper’s four-category presentation. The old one-round/three-round arms "
        "were not rerun: this study has branching paths of five turns. “First replies” below means depth 1 within "
        "these same debates, not a separate one-round experiment.</p></section>",
    ]

    parts.append(
        '<section id="a"><h2>A · What members say about the peer’s message</h2>'
        "<p>Each reply carries a self-reported agreement label. A shared prefix reply is counted once, "
        "even if several paths later pass through it. Each tone contains the same 60 questions and routing plans.</p>"
    )
    parts.append(
        table(
            "Table A1. Self-reported agreement (%)",
            ["Replies included", "Tone", "Replies", *LABEL_NAMES],
            [[r["scope"], r["tone"].title(), *agreement_cells(r)] for r in s["A"]],
            "Each label percentage divides by all replies in its row; the four categories sum to 100% "
            "up to rounding. No self-label is missing. First replies have no prior debate reply on their branch.",
        )
    )
    pooled_gap = 100 * (
        a_all[0]["counts"]["fully_agreed"] / a_all[0]["n"] - a_all[2]["counts"]["fully_agreed"] / a_all[2]["n"]
    )
    parts.append(
        f"<p>The pooled friendly-minus-hostile difference in full agreement is {pooled_gap:.1f} percentage "
        f"points (pp). Averaging each question’s difference equally gives "
        f"{ci_cell(s['A_friendly_minus_hostile_pp'], digits=1)} pp (95% interval). "
        "This is an observed tone contrast in one roster, not evidence that tone dominates model selection.</p>"
    )
    parts.append(
        "<details><summary>Model-by-tone breakdown</summary>"
        + table(
            "Table A2. All unique replies, by responding member (%)",
            ["Responding member", "Tone", "Replies", *LABEL_NAMES],
            [[r["model"], r["tone"].title(), *agreement_cells(r)] for r in s["A_by_model"]],
            "All three members participate in the same mixed committee; this is not a comparison of different committee rosters.",
        )
        + "</details></section>"
    )

    parts.append(
        '<section id="b"><h2>B · What the reply text actually says</h2>'
        "<p>Gemini reads the question, incoming peer contribution and reply, without the structured self-label "
        "or tone instruction. B/C/D share 8 presampled events per question across tones and depths. "
        "The following percentages describe those sampled events; they are not estimates reweighted to all replies.</p>"
    )
    b_headers = ["Sample", "Replies", *LABEL_NAMES, "No position / unjudgeable"]
    parts.append(
        table(
            "Table B1. External judge’s agreement ratings (%)",
            b_headers,
            [
                [r["group"], *agreement_cells(r), r["counts"]["no_position"] + r["counts"]["unjudgeable"]]
                for r in s["B"]
            ],
            "Four agreement categories use all sampled replies in the row. The last column is a count. "
            "A more disagreeing rating is not automatically a more accurate rating.",
        )
    )
    parts.append(
        table(
            "Table B2. Self-label versus external reading of exactly the same replies",
            ["Self-label ↓ / Judge →", *LABEL_NAMES, "Row total"],
            [[name, *row, sum(row)] for name, row in zip(LABEL_NAMES, b["matrix"], strict=True)],
            "Cell entries are counts. The diagonal means an exact four-level match. "
            "Do not compare full-run A percentages with sampled B percentages as if they used the same replies.",
        )
    )
    parts.append(
        "<details><summary>Sampled tone breakdown</summary>"
        + table(
            "Table B3. External ratings by tone (%)",
            b_headers,
            [
                [r["group"], *agreement_cells(r), r["counts"]["no_position"] + r["counts"]["unjudgeable"]]
                for r in s["B_by_tone"]
            ],
            "Tone is shown as a diagnostic breakdown; the main B/C/D summaries pool tones.",
        )
        + "</details></section>"
    )

    c_headers = [
        "Comparison / group",
        "Events",
        "Option changed / comparable",
        "Option unavailable",
        "Text unchanged",
        "Text adjusted",
        "Conclusion changed",
        "Text unjudgeable",
    ]
    parts.append(
        '<section id="c"><h2>C · Did the member’s own position change?</h2>'
        "<p>We separately ask the member to choose one original survey option and write its current full position. "
        "The selected letter gives a programmatic comparison; Gemini independently compares the two full texts. "
        "“Adjusted” means changed reasons, scope or qualifications without changing the main conclusion.</p>"
        "<p><strong>Previous → current</strong> compares the same member’s latest earlier participation on the "
        "same branch with its sampled current participation (using the initial position if this is its first turn). "
        "<strong>Initial → last</strong> uses a sampled position only when it is that member’s last participation "
        "on at least one complete path. A member can finish before the path’s fifth turn. These are sampled "
        "member endpoints, not all final positions or the chairman’s final answer.</p>"
    )
    parts.append(
        table(
            "Table C1. Option changes and full-text changes",
            c_headers,
            [c_cells(r) for r in s["C"]],
            "Text percentages divide by Events. Option percentages exclude unselectable before/after answers, "
            "with the excluded count shown explicitly. Comparisons may overlap: one pair can answer both questions. "
            "Across them there are 673 unique judged text pairs, not 480 + 345 independent pairs.",
        )
    )
    initial = s["C_initial_to_current"]
    parts.append(
        f"<p>Across all 480 sampled current positions, the initial-to-current option-change rate is "
        f"{fraction(initial['changed'], initial['valid'])} ({initial['unavailable']} unavailable). "
        "Initial-to-current text was judged only for sampled member endpoints, so no all-480 text rate is invented.</p>"
    )
    parts.append(
        table(
            "Table C2. Position change conditioned on the same turn’s self-reported peer agreement",
            c_headers,
            [c_cells(r) for r in s["C_by_self_label"]],
            "All rows compare previous own position → current own position. These are associations, "
            "not randomized groups. In particular, very small label groups cannot support a general conclusion.",
        )
    )
    parts.append("<details><summary>Changes by member, debate depth and participation count</summary>")
    parts.append(
        table("Table C3. Previous → current, by responding member", c_headers, [c_cells(r) for r in s["C_by_model"]])
    )
    parts.append(
        table(
            "Table C4. Changes over the debate path",
            [
                "Depth",
                "Events",
                "Previous → current option change",
                "Initial → current option change",
                "Text adjusted",
                "Text conclusion changed",
            ],
            [
                [
                    r["group"],
                    r["text"]["n"],
                    fraction(r["choice"]["changed"], r["choice"]["valid"]),
                    fraction(r["initial_choice"]["changed"], r["initial_choice"]["valid"]),
                    fraction(r["text"]["counts"]["adjusted"], r["text"]["n"]),
                    fraction(r["text"]["counts"]["conclusion_changed"], r["text"]["n"]),
                ]
                for r in s["C_by_depth"]
            ],
            "T is the turn’s index along its branch, not a round where all members speak. Text columns compare "
            "previous own → current. Samples at successive depths are not a fully observed longitudinal panel; "
            "differences between these rows do not isolate an effect of longer debate.",
        )
    )
    parts.append(
        table(
            "Table C5. Previous → current, by the member’s participation count on that branch",
            c_headers,
            [c_cells(r) for r in s["C_by_participation"]],
            "Participation 1 means this member’s first reply on the branch; its earlier reference is its initial position.",
        )
        + "</details>"
    )

    # Deterministic illustrative cases: never selected to estimate their frequency.
    examples = []
    for title, predicate in (
        (
            "Same selected option; text judged adjusted",
            lambda e: e["C_choice_adjacent_changed"] is False and e["C_text_adjacent"]["label"] == "adjusted",
        ),
        ("Selected option changed", lambda e: e["C_choice_adjacent_changed"] is True),
    ):
        for q in sorted(questions, key=lambda q: q["question_id"]):
            match = next((e for e in sorted(q["sampled_events"], key=lambda e: e["id"]) if predicate(e)), None)
            if match:
                before = q["C_readings"][match["previous_reading"]]
                after = q["C_readings"][match["current_reading"]]
                question = bank[q["question_id"]]

                def option(reading, question=question):
                    c = reading["choice"]
                    return f"{c} — {question['options'][ord(c) - 65]}" if c else "No selected option"

                examples.append(
                    f'<div class="example"><h3>{esc(title)}</h3>'
                    f'<p>{esc(question["text"])}</p><p class="muted">'
                    f"{MODEL_NAMES[match['member']]} · {match['tone']} · T={match['T']} · "
                    f"peer self-label: {esc(match['A'])}</p>"
                    f"<p><strong>Before: {esc(option(before))}</strong></p>"
                    f"<blockquote>{esc(before['position'])}</blockquote>"
                    f"<p><strong>After: {esc(option(after))}</strong></p>"
                    f"<blockquote>{esc(after['position'])}</blockquote>"
                    f"<p><strong>Judge:</strong> {esc(match['C_text_adjacent']['label'])}. "
                    f"{esc(match['C_text_adjacent']['evidence'])}</p></div>"
                )
                break
    parts.append(
        "<details><summary>Two actual before/after examples</summary><p>For illustration only: "
        "the first matching event in question-ID/event-ID order for each stated category. "
        "These examples are not representative estimates.</p>" + "".join(examples) + "</details></section>"
    )

    parts.append(
        '<section id="d"><h2>D · Probability changes at the same sampled events</h2>'
        "<p>D covers Qwen and Inkling in the same committee. It does not run another debate or use another question bank. "
        "Choice probabilities are taken from the same direct read that produces C’s option; the text probe adds "
        "two separate one-letter readings, without adding their outputs to the debate.</p>"
        "<h3>D-choice: probability of the member’s reference option</h3>"
        '<div class="formula">Δchoice = 100 × [p<sub>current</sub>(reference option) '
        "− p<sub>reference</sub>(reference option)]</div>"
        "<p>The reference option is the member’s previously selected option, or its initial selected option "
        "in the initial comparison. Negative values mean less probability on that same option; they need not "
        "mean a different sampled choice. The options retain each question’s original meaning; we do not "
        "average option letters across questions as an ordinal stance scale.</p>"
    )
    d_choice_rows = []
    for r in s["D"]:
        for label, key in (
            ("Previous → current", "D_choice_adjacent_pp"),
            ("Initial → current", "D_choice_initial_pp"),
            ("Initial → sampled last", "D_choice_final_pp"),
        ):
            est = r[key]
            d_choice_rows.append([r["model"], label, est["n"], est["questions"], ci_cell(est)])
    parts.append(
        table(
            "Table D1. Change in probability of the reference option (pp)",
            ["Member", "Reference comparison", "Comparisons", "Questions", "Mean [95% interval]"],
            d_choice_rows,
            "First average available comparisons within each question, then give supported questions equal weight. "
            "Unselectable reference options have no D-choice metric. A current unselectable choice may still have "
            "a normalized option distribution; its D-choice change can remain defined.",
        )
    )
    parts.append(
        "<h3>D-text: endorsement of the member’s own earlier full position</h3>"
        "<p>Fix the member’s full previous position text. In two otherwise matched copies of the pre-reply context, "
        "add either the actual incoming peer message or a repeated filler sentence: "
        "<q>The moderator noted the time and reminded participants that further remarks would follow in due course.</q> "
        "The filler approximately matches message length without arguing about the topic. Ask how much the "
        "member now endorses its own fixed text, from completely disagree (1) to completely agree (7).</p>"
        '<div class="formula">Δown-text = Σ<sub>k=1…7</sub> k × '
        "[p<sub>argument</sub>(k) − p<sub>filler</sub>(k)]</div>"
        "<p>Negative values mean the peer message lowers endorsement of the prior text relative to filler. "
        "The peer is not necessarily an opponent: its actual agreement label is retained below. "
        "This is a matched incoming-message test, not C’s post-reply before/after comparison.</p>"
    )
    parts.append(
        table(
            "Table D2. Mean agreement with own prior text (1–7 scale)",
            [
                "Member",
                "Message/control pairs",
                "Questions",
                "With argument",
                "With filler",
                "Argument − filler [95% interval]",
            ],
            [
                [
                    r["model"],
                    r["n"],
                    r["D_text_delta"]["questions"],
                    f"{r['D_text_argument']['mean']:.3f}",
                    f"{r['D_text_control']['mean']:.3f}",
                    ci_cell(r["D_text_delta"], digits=3),
                ]
                for r in s["D"]
            ],
            "Question-weighted means and paired differences. The fixed text is the previous own position, "
            "which is the initial position on a first participation. No separate fixed-initial-text probe was run "
            "for every later event. No top-two margin or entropy-matched adjustment is used.",
        )
    )
    parts.append(
        "<details><summary>Probability changes paired with self-reported agreement; readout diagnostics</summary>"
    )
    parts.append(
        table(
            "Table D3. Same-turn self-label and quantitative changes",
            [
                "Member",
                "Self-label about peer",
                "Events",
                "D-choice comparisons / questions",
                "Δchoice pp [95% interval]",
                "D-text pairs / questions",
                "Δown-text [95% interval]",
            ],
            [
                [
                    r["model"],
                    r["A"],
                    r["n"],
                    f"{r['D_choice_adjacent_pp']['n']} / {r['D_choice_adjacent_pp']['questions']}",
                    ci_cell(r["D_choice_adjacent_pp"]),
                    f"{r['D_text_delta']['n']} / {r['D_text_delta']['questions']}",
                    ci_cell(r["D_text_delta"], digits=3),
                ]
                for r in s["D_by_self_label"]
            ],
            "D-choice uses previous → current here. Label groups are observational and sometimes very small; "
            "intervals are descriptive and not multiplicity-adjusted.",
        )
    )
    parts.append(
        table(
            "Table D4. Probability-readout diagnostics",
            [
                "Member",
                "Read type",
                "Readings",
                "Missing candidates set to zero",
                "Min. returned candidate mass",
                "Median candidate mass",
                "Max normalized p ≥ 0.99",
            ],
            [
                [
                    r["model"],
                    r["kind"],
                    r["n"],
                    fraction(r["zero_fill"], r["n"]),
                    f"{r['min_mass']:.6f}",
                    f"{r['median_mass']:.6f}",
                    fraction(r["max_probability_ge_99"], r["n"]),
                ]
                for r in s["probability_audit"]
            ],
            "Top-20 scores only, read temperature 1. Absent candidate letters are approximated as zero and "
            "returned candidate probabilities are normalized. Reported candidate mass is before normalization; "
            "it is a lower bound when candidates are absent. A missing top-20 candidate need not have true probability zero.",
        )
        + '</details><p class="callout">These are context-conditioned output-token probabilities, not direct access '
        "to beliefs or calibrated confidence. C/D direct reads have reasoning disabled; formal debate has reasoning enabled. "
        "High concentration and the top-20 approximation limit how much fine-grained movement this readout can reveal. "
        "In six Inkling choice reads, returned option mass is below 0.5 (minimum 0.00438); all six emitted "
        "UNJUDGEABLE. Their normalized distributions are conditional on the listed options, not unconditional "
        "endorsement of those options. They are retained under the agreed normalization rule, not silently filtered.</p></section>"
    )

    parts.append(
        '<section id="e"><h2>E · Did the final answer improve?</h2>'
        "<p>The same Terra chairman synthesizes either the three original independent answers, or those exact "
        "answers plus the completed debate tree. The no-debate synthesis is reused across the three tones for "
        "each question. Gemini sees each answer pair twice in opposite orders and must choose one answer each time.</p>"
    )
    parts.append(
        table(
            "Table E1. Blind preference: debate synthesis versus no-debate synthesis",
            [
                "Tone",
                "Pairs",
                "Debate in both orders",
                "No debate in both orders",
                "Preference changes after reversal",
                "Debate preference score (%) [95% interval]",
            ],
            [
                [
                    r["tone"],
                    r["n"],
                    fraction(r["debate"], r["n"]),
                    fraction(r["baseline"], r["n"]),
                    fraction(r["split"], r["n"]),
                    ci_cell(r["estimate"], scale=100, digits=1, signed=False),
                ]
                for r in s["E"]
            ],
            "A pair scores 1 for debate in both orders, 0 for no debate in both, and 0.5 for inconsistent choices. "
            "The 0.5 cases are not explicit ties. All pairs remain included. Intervals resample questions; "
            "the pooled row averages the three tones within each question first.",
        )
    )
    hostile = s["E"][2]
    parts.append(
        f"<p>For hostile debate, the preference score is {100 * hostile['estimate']['mean']:.1f}%, "
        f"with {hostile['debate']} consistent debate preferences versus {hostile['baseline']} consistent "
        f"baseline preferences. Friendly and neutral show weaker or differently directed contrasts. "
        "Treat this as a result from this judge, rubric and roster: the study has not yet established "
        "human quality gains or replicated the effect across committees.</p>"
        "<details><summary>Answer lengths and interpretation</summary>"
    )
    parts.append(
        table(
            "Table E2. Mean synthesis length (whitespace-separated words)",
            ["Tone", "No debate", "With debate"],
            [[r["tone"], f"{r['baseline_words']:.1f}", f"{r['debate_words']:.1f}"] for r in s["E"][:3]],
            "The judge is instructed not to reward length itself, but this instruction does not prove the absence "
            "of length or style bias. No new damaged-answer calibration or human comparison was run.",
        )
        + "</details></section>"
    )

    parts.append(
        '<section id="methods"><h2>Scope, denominators and completion</h2>'
        "<p>For each question, three initial answers are shared across tones. The preplanned debate tree has "
        "two initial sender–receiver pairs and six five-turn paths; shared prefixes are generated only once. "
        "Different branches never share an evolving member-position state. The same routing is reused across tones.</p>"
        "<p>B/C/D sampling takes one event at each of the five depths, then three more with depth weights "
        "1:1:1:1:1.5. It is fixed before responses are generated and pools tones. Tables describe this sampling "
        "distribution without inverse-probability weighting. Events, paths and judgments are not independent questions.</p>"
        "<p>Percentages in A/B/C are descriptive pooled counts with explicit denominators. D means first average "
        "within each supported question. The A tone contrast and E preference score also weight questions equally. "
        "Shown 95% intervals use 10,000 question-level percentile bootstrap resamples with a fixed seed. "
        "These are post-collection descriptive analyses, not preregistered or multiplicity-adjusted significance tests.</p>"
    )
    count_rows = [
        ["Questions / debates", "60 / 180", "One mixed roster; 3 tones; no independent repetitions"],
        ["Completed / failed paths", "1,080 / 0", "Six five-turn paths per question × tone; overlapping prefixes"],
        ["Unique debate replies", n["formal_replies"], "Not 1,080 × 5 independently generated replies"],
        ["Sampled B/C events", n["sampled_events"], "8 per question, across tones"],
        [
            "C position readings",
            n["C_readings"],
            f"{s['unselectable_readings']} valid responses do not select an original option",
        ],
        ["Unique C text comparisons", n["C_judgments"], "Shared adjacent/endpoint comparisons judged once"],
        [
            "D-choice readings / D-text pairs",
            f"{n['D_choice_readings']} / {n['D_text_pairs']}",
            "Qwen and Inkling only",
        ],
        ["Syntheses / E judgments", "240 / 360", "60 shared baselines + 180 debate syntheses; two orders per pair"],
        [
            "Successful scientific calls",
            s["planned_counts"]["logical_calls"],
            "Retries are not new scientific observations",
        ],
        [
            "Recovered format / transport requests",
            "32 / 17",
            "49 logical requests; 50 additional attempts; none exhausted",
        ],
    ]
    parts.append(table("Table S1. Data collection and analysis units", ["Item", "Count", "Meaning"], count_rows))
    parts.append(
        f"<p>The final backfill retried 17 connection failures and completed 15 previously blocked tasks "
        f"(32 calls), reusing 6,864 valid results. All requests are complete. The {s['unselectable_readings']} "
        "unselectable C responses are a semantic outcome, not failed API calls; their text is still usable. "
        "Each numeric comparison reports its own available denominator.</p>"
        f"<p>The cumulative local ledger records <strong>US${s['charged_or_reserved_usd']:.2f}</strong> "
        "in token-based estimated charges plus unresolved reservations, including inherited pilot/retry accounting; "
        f"the final backfill adds about US${s['final_backfill_usd']:.2f}. "
        "This is not a provider invoice or a reconciled actual-spend total.</p>"
        '<p class="callout">What is not established here: a causal claim that every C change was caused by '
        "peer content (there is no matched no-peer C resampling arm); human-validated B/C/E ratings; "
        "lasting beliefs; or a comparison between same-model, same-family and mixed-family committees. "
        "Only the mixed-family roster was run. No instruction-removal ablation or supplementary benchmark was newly run.</p>"
    )

    inventory = []
    for index, q in enumerate(questions, 1):
        qid = q["question_id"]
        question = bank[qid]
        options = "; ".join(f"{chr(65 + i)}: {v}" for i, v in enumerate(question["options"]))
        inventory.append(
            [
                index,
                question["text"],
                options,
                len(plans[qid]["route"]) * 3,
                sum(e["C_choice_adjacent_changed"] is True for e in q["sampled_events"]),
            ]
        )
    parts.append(
        '<details class="inventory"><summary>The 60 questions and original answer options</summary>'
        + table(
            "Question inventory (screened, user-approved bank)",
            ["#", "Question", "Original options", "Unique replies, all tones", "Observed adjacent option changes"],
            inventory,
            "The final column is a raw count among the question’s eight sampled events, not a rate or quality score.",
        )
        + "</details>"
    )
    parts.append(
        "<details><summary>Report provenance and reproducibility</summary>"
        "<p>Generated offline from the final completed checkpoint. No model calls, new judgments, "
        "prompt changes or source-result edits were made for this report. The companion JSON contains "
        "unrounded table values, denominators, analysis settings and SHA-256 hashes of all input question files.</p>"
        f"<p>Source: <code>{esc(s['source_directory'])}</code><br>"
        f"Run completed: <code>{esc(s['completed_at_utc'])}</code><br>"
        f"Manifest SHA-256: <code>{s['manifest_sha256']}</code></p>"
        "<p>Generator: <code>python -m llm_committee.pivot.results_report RUN_DIRECTORY OUTPUT.html</code></p>"
        '<p>Table presentation reference: <a href="https://arxiv.org/pdf/2609.08016">arXiv:2609.08016</a>, '
        "Tables 2–5. Its older data and metrics have not been merged into this report.</p></details></section>"
        "<footer>Research summary · 60 screened opinion questions · one mixed committee · "
        "data-derived tables, not publication-ready claims. All styles and content are embedded; no network is needed to read this file.</footer>"
        "</main></body></html>"
    )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".html":
        parser.error("Output must end in .html")
    source = args.source.resolve()
    output = args.output.resolve()
    if source in output.parents:
        parser.error("Keep generated analysis outside the immutable run directory")
    summary, bank, questions, plans = analyze(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(summary, bank, questions, plans), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(output.with_suffix(".json"))


if __name__ == "__main__":
    main()
