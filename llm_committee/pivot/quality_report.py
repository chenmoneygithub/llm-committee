"""Append length-controlled E to the same HTML without changing A–D results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean

from .dyadic import PAIRS
from .forced_feedback_report import esc, rate, table
from .quality_study import ARMS, CALIBRATION_KINDS, LENGTH, VERSION, words
from .results_report import ci_cell, question_estimate

KIND_NAMES = {
    "verbosity": "Expanded wording (same content intended)",
    "moderate_damage": "One contradictory reason (moderate intended damage)",
    "severe_damage": "Off-topic answer (severe intended damage)",
}


def load_run(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] != VERSION or report["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Need a finished current-protocol E follow-up")
    parent = Path(manifest["source"]["directory"])
    assert hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest() == manifest["source"]["manifest_sha256"]
    records = []
    for plan in manifest["plans"]:
        qid = plan["question_id"]
        assert (
            hashlib.sha256((parent / "questions" / f"{qid}.json").read_bytes()).hexdigest()
            == manifest["source"]["question_sha256"][qid]
        )
        record = json.loads((source / "questions" / f"{qid}.json").read_text())
        assert record["question_id"] == qid
        assert len(record["E"]) == 6
        for row in record["E"]:
            votes = [
                o["judgment"]["preference"] == o["target_side"] for o in row["preferences"]["orders"] if o["judgment"]
            ]
            assert row["debate_score"] == (sum(votes) / 2 if len(votes) == 2 else None)
            for field in ("baseline", "debated"):
                if row[field]:
                    assert LENGTH[0] <= words(row[field]["answer"]) <= LENGTH[1]
        records.append(record)
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        ledger = db.execute("SELECT status,COUNT(*),SUM(charge) FROM calls GROUP BY status").fetchall()
        errors = db.execute("SELECT error,COUNT(*) FROM calls WHERE status='invalid' GROUP BY error").fetchall()
    report["cost_accounting"] = {
        "received_response_estimate_usd": sum(v for s, n, v in ledger if s in ("completed", "received", "invalid")),
        "unresolved_reservations_usd": sum(v for s, n, v in ledger if s not in ("completed", "received", "invalid")),
        "unresolved_attempts": sum(n for s, n, v in ledger if s not in ("completed", "received", "invalid")),
    }
    report["invalid_output_reasons"] = [{"reason": error, "attempts": n} for error, n in errors]
    return manifest, report, records


def construction_valid(case, kind):
    check = case["validation"]
    if not check:
        return None
    fields = {
        "verbosity": ["equivalent_content"],
        "moderate_damage": ["moderate_contradiction", "moderate_preserves_conclusion"],
        "severe_damage": ["severe_off_topic"],
    }[kind]
    return all(check[field] == "yes" for field in fields)


def summarize(records):
    rows = [{"question_id": q["question_id"], **r} for q in records for r in q["E"]]
    main = []
    for pair in PAIRS:
        for arm in ARMS:
            group = [r for r in rows if r["pair"] == pair and r["assignment"] == arm]
            valid = [r for r in group if r["debate_score"] is not None]
            counts = Counter(r["debate_score"] for r in valid)
            gaps = [r["debate_words"] - r["baseline_words"] for r in valid]
            main.append(
                {
                    "pair": pair,
                    "assignment": arm,
                    "planned": len(group),
                    "valid": len(valid),
                    "debate_both": counts[1],
                    "baseline_both": counts[0],
                    "inconsistent": counts[0.5],
                    "estimate": question_estimate(group, "debate_score"),
                    "baseline_words": mean(r["baseline_words"] for r in valid) if valid else None,
                    "debate_words": mean(r["debate_words"] for r in valid) if valid else None,
                    "mean_word_gap": mean(gaps) if gaps else None,
                    "max_absolute_word_gap": max(map(abs, gaps), default=None),
                }
            )
    calibration_rows = []
    for q in records:
        if case := q["calibration"]:
            for kind, variant in case["variants"].items():
                calibration_rows.append(
                    {
                        "question_id": q["question_id"],
                        "kind": kind,
                        "score": variant["target_score"],
                        "construction_valid": construction_valid(case, kind),
                        "source_words": words(case["base"]["answer"]) if case["base"] else None,
                        "variant_words": words(variant["answer"]["answer"]) if variant["answer"] else None,
                    }
                )
    calibration = []
    for kind in CALIBRATION_KINDS:
        all_rows = [r for r in calibration_rows if r["kind"] == kind]
        valid = [r for r in all_rows if r["score"] is not None]
        checked = [r for r in valid if r["construction_valid"]]
        for name, group in (("All presampled cases", valid), ("Construction-check passed only", checked)):
            counts = Counter(r["score"] for r in group)
            calibration.append(
                {
                    "kind": kind,
                    "subset": name,
                    "planned": len(all_rows),
                    "judged": len(group),
                    "construction_valid": sum(r["construction_valid"] is True for r in group),
                    "source_both": counts[1],
                    "variant_both": counts[0],
                    "inconsistent": counts[0.5],
                    "source_score": mean(r["score"] for r in group) if group else None,
                }
            )
    return {
        "primary_rows": rows,
        "primary_summary": main,
        "calibration_rows": calibration_rows,
        "calibration_summary": calibration,
    }


def decimal(value):
    return "—" if value is None else f"{value:.1f}"


def preference_text(value, target):
    score = value["target_score"]
    if score is None:
        verdict = "Missing complete two-order comparison"
    elif score == 1:
        verdict = f"{target} preferred in both orders"
    elif score == 0:
        verdict = "Other answer preferred in both orders"
    else:
        verdict = "Preference changed when order reversed (not an explicit tie)"
    explanations = "".join(
        f"<p>Order {i + 1} ({esc(target)} shown on the {esc(o['target_side'])}; "
        f"judge selects {esc((o['judgment'] or {}).get('preference', 'neither: missing'))}): "
        f"{esc((o['judgment'] or {}).get('evidence', 'Missing'))}</p>"
        for i, o in enumerate(value["orders"])
    )
    return f"<p><strong>{esc(verdict)}</strong></p><details><summary>Both blind judge explanations</summary>{explanations}</details>"


def full_case(record, pair):
    group = {r["assignment"]: r for r in record["E"] if r["pair"] == pair}
    base = group["original"]["baseline"]
    panels = []
    for arm in ARMS:
        r = group[arm]
        panels.append(
            f"<article><h4>{arm.title()} continuation</h4><p>Turn tones: {esc(' → '.join(r['tone_sequence']))}</p>"
            f"<p>{esc(r['debated']['answer'] if r['debated'] else 'Missing answer')}</p><p>Words: {esc(r['debate_words'])}</p>"
            + preference_text(r["preferences"], "Debate answer")
            + "</article>"
        )
    options = "".join(f"<li>{chr(65 + i)}: {esc(v)}</li>" for i, v in enumerate(record["question"]["options"]))
    return (
        f"<details class='e-case'><summary>{esc(record['question_id'])} · {pair} — {esc(record['question']['text'])}</summary>"
        f"<ul>{options}</ul><h4>Shared no-debate synthesis ({esc(group['original']['baseline_words'])} words)</h4>"
        f"<p class='shared-text'>{esc(base['answer'] if base else 'Missing answer')}</p>"
        f"<div class='fork-grid'>{''.join(panels)}</div></details>"
    )


def calibration_case(record):
    case = record["calibration"]
    if not case:
        return ""
    body = f"<h4>Unmodified synthesis</h4><p>{esc(case['base']['answer'] if case['base'] else 'Missing')}</p>"
    check = case["validation"]
    body += "<h4>Automatic construction review (not human validation)</h4>"
    if check:
        body += table(
            ["Property", "Assessment", "Supporting detail"],
            [
                ["Expansion preserves substantive content", check["equivalent_content"], check["equivalence_evidence"]],
                [
                    "Edited version has an internal contradiction",
                    check["moderate_contradiction"],
                    check["moderate_evidence"],
                ],
                [
                    "Edited version retains main conclusion",
                    check["moderate_preserves_conclusion"],
                    "See contradiction explanation above",
                ],
                ["Severe version is off topic", check["severe_off_topic"], check["severe_evidence"]],
            ],
        )
    else:
        body += "<p>Construction review unavailable.</p>"
    for kind in CALIBRATION_KINDS:
        variant = case["variants"][kind]
        body += f"<h4>{esc(KIND_NAMES[kind])}</h4><p>{esc(variant['answer']['answer'] if variant['answer'] else 'Missing')}</p>"
        validity = construction_valid(case, kind)
        verdict = "passed" if validity else "failed or uncertain" if validity is False else "unavailable"
        body += f"<p>Construction check: {verdict}; words: {words(variant['answer']['answer']) if variant['answer'] else 'missing'}.</p>"
        body += preference_text(variant, "Unmodified source")
    return f"<details class='e-calibration-case'><summary>{esc(record['question_id'])} · {case['pair']} · {case['arm']} — {esc(record['question']['text'])}</summary>{body}</details>"


def render(manifest, report, records):
    summary = summarize(records)
    primary = summary["primary_summary"]
    rows = [
        [
            r["pair"],
            r["assignment"].title(),
            f"{r['valid']}/{r['planned']}",
            rate(r["debate_both"], r["valid"]),
            rate(r["baseline_both"], r["valid"]),
            rate(r["inconsistent"], r["valid"]),
            ci_cell(r["estimate"], scale=100, digits=1, signed=False),
        ]
        for r in primary
    ]
    lengths = [
        [
            r["pair"],
            r["assignment"].title(),
            decimal(r["baseline_words"]),
            decimal(r["debate_words"]),
            decimal(r["mean_word_gap"]),
            r["max_absolute_word_gap"],
        ]
        for r in primary
    ]

    def cal_table(subset):
        return table(
            [
                "Probe",
                "Judged / presampled",
                "Construction check passed",
                "Source in both orders",
                "Variant in both orders",
                "Order-inconsistent",
                "Source preference score (%)",
            ],
            [
                [
                    KIND_NAMES[r["kind"]],
                    f"{r['judged']}/{r['planned']}",
                    r["construction_valid"],
                    rate(r["source_both"], r["judged"]),
                    rate(r["variant_both"], r["judged"]),
                    rate(r["inconsistent"], r["judged"]),
                    decimal(100 * r["source_score"]) if r["source_score"] is not None else "—",
                ]
                for r in summary["calibration_summary"]
                if r["subset"] == subset
            ],
        )

    cost = report["cost_accounting"]
    billing = f"New E calls: token-estimated US${cost['received_response_estimate_usd']:.2f}, not a provider invoice."
    if cost["unresolved_attempts"]:
        billing += f" {cost['unresolved_attempts']} attempts have unknown billing; US${cost['unresolved_reservations_usd']:.2f} remains reserved, not confirmed spending."
    retries = report["request_retries"]
    mock = '<p class="callout">OFFLINE MOCK — not scientific results.</p>' if manifest["kind"] == "offline_mock" else ""
    notes = " ".join(
        f"{r['pair']} / {r['assignment']}: {r['debate_both']} stable debate preferences, {r['baseline_both']} stable baseline preferences, and {r['inconsistent']} order-inconsistent pairs out of {r['valid']} complete comparisons."
        for r in primary
    )
    html = f"""<section id="e"><h2>E · Final answers under the same output-length budget</h2>{mock}
<p>This follow-up reuses the current {len(records)}-question, turn-level-tone, strongly/leaning debates. No debate or initial position was regenerated.
Each exclusive dyad is evaluated separately: AB, CA and BC are not combined into a three-member committee.</p>
<p><strong>No debate:</strong> Terra synthesizes that pair's two original independent answers.
<strong>Debate:</strong> the same Terra chairman receives those exact answers plus the pair's four public replies.
The original and alternate continuations have separate syntheses but share one no-debate baseline for that question/pair.
Both branches use per-turn tone assignments; there are no friendly/neutral/hostile global arms here.</p>
<p>Every primary synthesis must contain <strong>190–210 whitespace-separated words</strong>, checked before acceptance.
The same prompt, model, reasoning setting and length bounds apply on both sides. Out-of-range or malformed answers get at most two
identical retries; no truncation or preference-based answer selection. Actual lengths and remaining differences are reported below.</p>
<p>One Gemini 3.8 Flash judge sees only the question/options and two answers, once in each order. Its rubric still says not to reward
length or style alone. An order-inconsistent pair has one decision favoring debate and one favoring baseline;
it is <strong>not an explicit tie</strong>. Two judge orders are not two independent questions.</p>
<p class="status">{report["completed_questions"]}/{len(manifest["plans"])} questions processed; {report["comparison_statuses"].get("success", 0)}/{manifest["planned_counts"]["primary_answer_pairs"]} complete primary comparisons;
{report["comparison_statuses"].get("failed", 0)} unavailable comparisons. {retries["additional_attempts"]} additional technical/length attempts,
{retries["exhausted_logical_requests"]} exhausted requests. {esc(billing)} A–D results above are unchanged.</p>
<h3>E1 · Debate versus no debate, by dyad and continuation</h3>
{table(["Pair", "Continuation", "Complete / planned pairs", "Debate in both orders", "Baseline in both orders", "Order-inconsistent", "Debate wins (%) [95% interval]"], rows)}
<p><strong>Debate wins (%)</strong> = judge decisions favoring the debate answer ÷ all judge decisions × 100.
Each complete pair contributes two decisions, one per answer order. An order-inconsistent pair contributes one win out of two.
This is a judge preference rate, not the percentage improvement in answer quality.</p>
<p>{esc(notes)}</p>
<p>Intervals are descriptive percentile bootstraps over whole questions (10,000 resamples), not multiplicity-adjusted claims.
Each row has at most {len(records)} independent questions; the two continuations reuse a baseline and prefix. A preference difference is evidence
about this judge and rubric, not a calibrated magnitude of quality improvement or proof of human benefit.</p>
<h3>E2 · Was the length control actually satisfied?</h3>
{table(["Pair", "Continuation", "Baseline mean words", "Debate mean words", "Mean debate − baseline words", "Largest absolute paired gap"], lengths)}
<p>This is a common narrow output budget, not identical word counts. It reduces the previous length gap; it does not prove that residual
length, style or information-density preferences are absent. The archived 60-question global-tone E differs in more than length and is
not a matched before/after control. No old/new causal contrast is reported.</p>
<h3>E3 · Small judge diagnostics: wording length and damaged answers</h3>
<p>{sum(bool(r["calibration"]) for r in records)} questions were selected by frozen batch plans before new E outputs, balanced across pair and continuation within each batch. For each selected synthesis,
Terra constructs: a 290–310-word expansion with the same substantive content intended; a 190–210-word version containing one contradictory
reason while retaining the conclusion; and a 190–210-word off-topic version. Gemini compares each to its source in both orders using the
same blind E rubric. A separate Gemini request checks whether each construction actually meets its specification.</p>
{cal_table("All presampled cases")}
<p>All presampled cases are shown, including failed construction checks. A high source preference on damaged pairs indicates sensitivity
to those particular defects. Expanded wording has no assumed correct winner: preference for it can reflect presentation sensitivity,
and content drift must be inspected. This small, model-constructed battery is not independently human validated and cannot certify the judge.</p>
<p>The single-contradiction condition is localized damage, not a calibrated magnitude of “moderate quality loss.” Detecting it or an off-topic answer
does not establish sensitivity to subtle quality differences between naturally generated syntheses.</p>
<details><summary>Diagnostics restricted to constructions passing the automatic check</summary>
<p>This subset is selected only by a separate property check, never by preference outcomes. The checker is the same model family as the
judge, so this is not independent validation. Failed/uncertain constructions remain visible above and below.</p>
{cal_table("Construction-check passed only")}</details>
<details><summary>Inspect all {sum(bool(r["calibration"]) for r in records)} diagnostic cases and both-order explanations</summary>
{"".join(calibration_case(r) for r in records)}</details>
<h3>Read all {len(records) * len(PAIRS)} matched synthesis cases</h3>
<p>Each case shows its shared baseline once, both branch syntheses, their exact turn-tone sequences, and both blind judge explanations.
Chairman and judge inputs do not include these researcher-visible tone annotations.</p>
{"".join(full_case(r, pair) for r in records for pair in PAIRS)}
<details><summary>Invalid-output attempts and completion details</summary>
{table(["Recorded reason", "Attempts"], [[r["reason"], r["attempts"]] for r in report["invalid_output_reasons"]])}
<p>No exhausted answer is silently replaced by a shorter-tree synthesis. Missing comparisons are unavailable, not scored as ties or losses.</p></details>
<p class="muted">E protocol: {esc(VERSION)}. This section adds a new equal-budget comparison; it does not change A–D or retroactively correct the archived E.</p>
</section>"""
    summary.update(
        protocol_version=VERSION,
        kind=manifest["kind"],
        report=report,
        planned_counts=manifest["planned_counts"],
        design=manifest["design"],
    )
    return html, summary


def include_quality(document, summary, source):
    manifest, report, records = load_run(source)
    if manifest["kind"] != "paid_turn_tone_quality":
        raise ValueError("Never append mock E to the real report")
    if (
        summary["source_directory"] != manifest["source"]["directory"]
        or summary["manifest_sha256"] != manifest["source"]["manifest_sha256"]
    ):
        raise ValueError("E belongs to a different A–D run")
    if '<section id="e">' in document or "length_controlled_E" in summary:
        raise ValueError("E is already included; rebuild from the unchanged A–D report first")
    fragment, extra = render(manifest, report, records)
    extra["source_directory"] = str(Path(source).resolve())
    extra["manifest_sha256"] = hashlib.sha256((Path(source) / "manifest.json").read_bytes()).hexdigest()
    document = document.replace('<section id="paired">', fragment + '<section id="paired">', 1)
    document = document.replace(
        '<a href="#paired">', '<a href="#e">E · Equal-budget final answers</a><a href="#paired">', 1
    )
    document = document.replace(
        "was run; no same-model/family roster, three-member discussion or final-answer synthesis was started.</p>",
        "was run; no same-model/family roster or three-member discussion was started. The E follow-up below adds final-answer synthesis from these same dyads.</p>",
    )
    return document, {**summary, "length_controlled_E": extra}
