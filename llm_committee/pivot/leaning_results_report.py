"""Offline HTML for the fresh leaning-rubric run, never relabeling legacy data.

python -m llm_committee.pivot.leaning_results_report RUN_DIRECTORY OUTPUT.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from .agreement import AGREEMENT, agreement_manifest
from .conditional_results_report import analyze_conditional, b_finding, d_choice_table, d_text_table, render_b
from .models import TEXT_SHIFT
from .results_report import CSS, MODEL_NAMES, analyze, c_cells, categories, changes, ci_cell, fraction, pct, table

PROMPT_VERSION = "pivot-main-2026-09-26-v5-leaning-agreement"
LABEL_NAMES = ("Fully agree", "Leaning agree", "Leaning disagree", "Fully disagree")

FEEDBACK_CSS = """
#forced-feedback { border-top:3px solid var(--accent); padding-top:28px; }
#forced-feedback .scroll { overflow-x:auto; margin:18px 0; }
#forced-feedback th, #forced-feedback td { text-align:left; }
#forced-feedback .notice { border-left:3px solid #ba8137; padding:8px 18px; background:#fffaf0; }
#forced-feedback h4 { font-size:16px; margin:22px 0 10px; }
@media print { #forced-feedback .scroll { overflow:visible; } }
"""


def include_feedback(document, summary, source):
    """Keep the supplementary cohort separate while making one self-contained page."""
    from .forced_feedback_report import load_run, render_embedded

    source = source.resolve()
    manifest, report, records = load_run(source)
    if manifest.get("kind") != "paid_forced_feedback_supplement" or report["status"] not in (
        "completed",
        "completed_with_failures",
    ):
        raise ValueError("Only a finished real feedback supplement belongs in the main report")
    if manifest["source"]["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("Feedback cases do not originate from this main experiment")
    if 'id="forced-feedback"' in document:
        raise ValueError("Feedback is already embedded; rebuild the base report before updating")
    if any(document.count(marker) != 1 for marker in ("</style>", "</nav>", "</header>", "<footer>")):
        raise ValueError("Unexpected main report structure")
    fragment, data = render_embedded(manifest, report, records)
    document = document.replace("</style>", FEEDBACK_CSS + "</style>", 1)
    document = document.replace("</nav>", '<a href="#forced-feedback">Supplement · Forced disagreement</a></nav>', 1)
    document = document.replace(
        "</header>",
        '<p class="status"><strong>Supplement also complete:</strong> '
        f"{report['completed_cases']} paired feedback cases. "
        '<a href="#forced-feedback">Jump to the strong-disagreement experiment, C/D tables and case texts.</a> '
        "Its results are reported separately, not pooled into the main A–E scores.</p></header>",
        1,
    )
    document = document.replace("<footer>", fragment + "<footer>", 1)
    summary = {
        **summary,
        "supplements": {
            **summary.get("supplements", {}),
            "forced_feedback": {
                **data,
                "source_directory": str(source),
                "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                "main_study_manifest_sha256": manifest["source"]["manifest_sha256"],
                "pooled_with_main_study": False,
            }
        },
    }
    return document, summary


def analyze_leaning(source):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("prompt_version") != PROMPT_VERSION or manifest.get("agreement_rubric") != agreement_manifest():
        raise ValueError("Expected the frozen leaning rubric; legacy relabeling is not allowed")
    if "continuation" in manifest:
        raise ValueError("This report expects the fresh rerun, not an imported checkpoint")
    result = analyze(source, agreement_labels=AGREEMENT, label_names=LABEL_NAMES, prompt_version=PROMPT_VERSION)
    summary, bank, questions, plans = result
    report = json.loads((source / "report.json").read_text())
    assert report["reused_checkpoint_records"] == 0
    summary.update(
        schema="mixed_leaning_results_summary_v1",
        prompt_version=PROMPT_VERSION,
        agreement_rubric=manifest["agreement_rubric"],
        label_names=list(LABEL_NAMES),
        call_status_counts=report["new_call_status_counts"],
        reused_checkpoint_records=0,
        old_data_included=False,
    )
    summary.pop("final_backfill_usd")  # This is a fresh run, not a backfill of the reference run.
    summary["low_mass_choice_reads"] = [
        {
            "question_id": q["question_id"],
            "reading_id": rid,
            "choice": q["C_readings"][rid]["choice"],
            "candidate_mass": dist["candidate_mass"],
        }
        for q in questions
        for rid, dist in q["D"]["choice_readings"].items()
        if dist["candidate_mass"] < 0.5
    ]
    summary["C_by_depth_and_self_label"] = position_turn_breakdown(questions)
    summary["conditional_analysis"] = analyze_conditional(questions)
    return summary, bank, questions, plans


def position_turn_breakdown(questions):
    """Keep the current reply's label and branch depth attached to its own-position comparison."""
    events = [
        {"question_id": q["question_id"], **event, "text_label": event["C_text_adjacent"]["label"]}
        for q in questions
        for event in q["sampled_events"]
    ]
    if any(e["T"] not in range(1, 6) or e["A"] not in (*AGREEMENT, None) for e in events):
        raise ValueError("Unexpected debate depth or self-report label")
    labels = list(zip(AGREEMENT, LABEL_NAMES, strict=True))
    if any(e["A"] is None for e in events):
        labels.append((None, "Unreported"))
    rows = []
    for depth in range(1, 6):
        for label, name in labels:
            selected = [e for e in events if e["T"] == depth and e["A"] == label]
            rows.append(
                {
                    "group": f"T = {depth} · {name}",
                    "T": depth,
                    "self_label": label,
                    "label_name": name,
                    "questions": len({e["question_id"] for e in selected}),
                    "choice": changes(selected, "C_choice_adjacent_changed"),
                    "text": categories(selected, "text_label", TEXT_SHIFT),
                }
            )
    return rows


def render_position_finding(s):
    return (
        "<li><strong>C · Does the member change its own position when it says it agrees or disagrees?</strong> "
        "Comparing its previous own position with its position after the reply, option changes within each "
        "current self-label category are "
        + "; ".join(
            f"{r['group']}: {fraction(r['choice']['changed'], r['choice']['valid'])}" for r in s["C_by_self_label"]
        )
        + '. <a href="#c-by-label">Text changes use the same groups</a>; '
        '<a href="#c-by-turn-label">T = 1…5 is then shown within each agreement level</a>. '
        "Small denominators, especially Fully disagree, must be read as case counts rather than stable rates.</li>"
    )


def agreement_cells(row):
    return [row["n"], *[pct(row["counts"][k], row["n"]) for k in AGREEMENT]]


def render_label_comparison(ab):
    rows = [
        [
            "Exactly the same label",
            ab["counts"].get("same", 0),
            pct(ab["counts"].get("same", 0), ab["n"]),
            "The member and Gemini select the same one of the four labels.",
        ],
        [
            "Gemini rates the reply closer to disagreement",
            ab["counts"].get("more_disagreeing", 0),
            pct(ab["counts"].get("more_disagreeing", 0), ab["n"]),
            "For example: the member says Fully agree, but Gemini reads its reply as Leaning agree.",
        ],
        [
            "Gemini rates the reply closer to agreement",
            ab["counts"].get("less_disagreeing", 0),
            pct(ab["counts"].get("less_disagreeing", 0), ab["n"]),
            "For example: the member says Leaning agree, but Gemini reads its reply as Fully agree.",
        ],
    ]
    return (
        '<div class="label-comparison">'
        + table(
            "Table B2. Does Gemini’s reading of the text match the member’s self-report?",
            ["Comparison", "Replies", f"Share of {ab['n']} replies", "Meaning"],
            rows,
            "Labels are ordered Fully agree → Leaning agree → Leaning disagree → Fully disagree. "
            "The examples explain direction; each count includes all label pairs in that direction, not just the example.",
        )
        + "</div>"
    )


def examples(bank, questions):
    pieces = []
    for title, predicate in (
        (
            "Same option, adjusted full text",
            lambda e: e["C_choice_adjacent_changed"] is False and e["C_text_adjacent"]["label"] == "adjusted",
        ),
        ("A changed option", lambda e: e["C_choice_adjacent_changed"] is True),
    ):
        for q in sorted(questions, key=lambda q: q["question_id"]):
            event = next((e for e in sorted(q["sampled_events"], key=lambda e: e["id"]) if predicate(e)), None)
            if event is None:
                continue
            question = bank[q["question_id"]]
            pieces.append(
                f'<div class="example"><h3>{title}</h3><p>{html.escape(question["text"])}</p>'
                f'<p class="muted">{MODEL_NAMES[event["member"]]} · {event["tone"]} · T={event["T"]} · '
                f"self-label about peer: {html.escape(str(event['A']))}</p>"
            )
            for name, key in (("Before", "previous_reading"), ("After", "current_reading")):
                reading = q["C_readings"][event[key]]
                option = reading["choice"]
                choice = f"{option} — {question['options'][ord(option) - 65]}" if option else "No selected option"
                pieces.append(
                    f"<p><strong>{name}: {html.escape(choice)}</strong></p>"
                    f"<blockquote>{html.escape(reading['position'])}</blockquote>"
                )
            pieces.append(
                f"<p><strong>Text judge:</strong> {html.escape(event['C_text_adjacent']['label'])}. "
                f"{html.escape(event['C_text_adjacent']['evidence'])}</p></div>"
            )
            break
    return (
        "<details><summary>Two actual before/after examples</summary><p>Illustrations only, not representative "
        "estimates: the first matching event by question ID and event ID for each category.</p>"
        + "".join(pieces)
        + "</details>"
    )


def render_leaning(s, bank, questions, plans):
    n = s["counts"]
    all_a, all_e = s["A"][3:], s["E"][-1]
    esc = html.escape
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Leaning-label rerun — 60-question committee results</title>"
        f"<style>{CSS}\n#labels td, .label-comparison th:last-child, "
        ".label-comparison td:last-child { text-align:left; }</style></head><body><main>"
        '<header><div class="eyebrow">GlobalOpinionQA · Fresh leaning-label run · 26 September 2026</div>'
        "<h1>Committee debate: the leaning-label rerun</h1>"
        '<p class="lede">A–E results for GPT-5.6 Terra, Qwen3.8-27B and Inkling, '
        "using fully agree, leaning agree, leaning disagree and fully disagree.</p>"
        f'<p class="status"><strong>Complete:</strong> {n["questions"]}/60 questions · {n["debates"]}/180 debates · '
        f"{s['trajectories']['success']:,}/1,080 paths · {n['formal_replies']:,} unique replies. "
        "No technical task remains missing.</p>"
        "<p>All initial answers, debates, measurements and judgments were generated anew. "
        "The previous partial-agreement run is reference only; none of its outputs is included here. "
        "Chairman: GPT-5.6 Terra. Single B/C/E judge: Gemini 3.8 Flash.</p></header>"
        '<nav aria-label="Sections"><a href="#summary">Summary</a><a href="#labels">Four labels</a>'
        '<a href="#a">A · Self-labels</a><a href="#b">B · Reply text</a><a href="#c">C · Position change</a>'
        '<a href="#d">D · Probabilities</a><a href="#e">E · Final answers</a><a href="#scope">Scope & accounting</a></nav>',
        '<section id="summary"><h2>Main observations</h2><div class="findings"><ul>',
        "<li><strong>A · Reported agreement varies by tone.</strong> Full agreement is "
        + ", ".join(f"{pct(r['counts']['fully_agree'], r['n'])} for {r['tone']}" for r in all_a)
        + ". This is agreement with the incoming peer message, not a changed position on the survey question.</li>",
        b_finding(s["conditional_analysis"]["B"]),
        render_position_finding(s),
        "<li><strong>D · Probability changes are paired with the same reply’s self-label.</strong> "
        "For each agreement level, Qwen and Inkling are shown separately: how the prior option’s probability "
        "changes, and how endorsement of the prior full text differs between peer feedback and filler. "
        "The latter uses a 1–7 rating scale, not a percentage of changed minds. "
        '<a href="#d">The conditional tables show the actual reference scores and denominators.</a></li>',
        f"<li><strong>E · Final-answer preference depends on tone.</strong> Across {all_e['n']} pairs, "
        f"{all_e['debate']} favor debate in both orders, {all_e['baseline']} favor no debate in both, and "
        f"{all_e['split']} change preference after reversal. The pooled debate preference score is "
        f"{ci_cell(all_e['estimate'], scale=100, digits=1, signed=False)}% (95% interval); "
        "50% is within that interval. The tone-specific results are shown separately below.</li></ul></div></section>",
        '<section id="labels"><h2>What the four labels mean</h2>'
        "<p>The following definitions were supplied verbatim to both the debater’s self-label prompt (A) "
        "and the external text judge (B). They concern the <strong>peer’s central position</strong>, not agreement "
        "with the original survey statement.</p>",
        table(
            "Shared A/B definitions",
            ["Label", "Definition"],
            [
                [name, s["agreement_rubric"]["definitions"][key]]
                for name, key in zip(LABEL_NAMES, AGREEMENT, strict=True)
            ],
        ),
        f"<p>{esc(s['agreement_rubric']['decision_rule'])}</p>"
        "<p>When a direction cannot be determined, A can return no label; B can return no position or unjudgeable. "
        "Those outcomes are not forced into either leaning category.</p></section>",
        '<section id="a"><h2>A · Members’ self-reported agreement</h2>'
        "<p>Every generated reply is counted once. A reply shared by several later paths is not counted repeatedly. "
        "The table includes all five debate depths; no separate one-round experiment was run.</p>",
        table(
            "Table A1. Four-level self-reported agreement (%)",
            ["Tone", "Replies", *LABEL_NAMES, "Unreported (n)"],
            [[r["tone"].title(), *agreement_cells(r), r["other"]] for r in all_a],
            "Each percentage divides by the number of replies in that tone. Shared branch prefixes are counted once.",
        ),
        f"<p>The friendly-minus-hostile full-agreement difference, first calculated within each question and then "
        f"averaged equally across questions, is {ci_cell(s['A_friendly_minus_hostile_pp'], digits=1)} percentage "
        "points (pp), with a question-level 95% interval. This is one mixed roster, not a test of whether tone "
        "dominates committee composition.</p>",
        "<details><summary>Model-by-tone breakdown</summary>"
        + table(
            "Table A2. All replies by responding member (%)",
            ["Member", "Tone", "Replies", *LABEL_NAMES, "Unreported (n)"],
            [[r["model"], r["tone"].title(), *agreement_cells(r), r["other"]] for r in s["A_by_model"]],
        )
        + "</details></section>",
        '<section id="b"><h2>B · Does the reply text support the member’s agreement label?</h2>'
        "<p>Each member supplies a written reply and one of the four agreement labels about the peer’s message. "
        "We then ask <strong>one external judge model, Gemini 3.8 Flash</strong>, to label that same reply using "
        "the same four definitions. Gemini sees the original question and options, the peer’s message, and the "
        "member’s reply. It does <strong>not</strong> see the member’s self-reported label or tone instruction. "
        "There is one Gemini rating per sampled reply, not a vote across several judges.</p>"
        "<p>The sample contains 480 replies: 8 selected in advance per question, spread across the three tones. "
        "An exact match means, for example, that a member reports Leaning disagree and Gemini independently "
        "assigns Leaning disagree from the reply text. The percentages below concern agreement with the peer, "
        "not whether the answer to the original question is correct or whether the member changed its own position.</p>",
    ]
    parts.extend(
        [
            render_b(s["conditional_analysis"]["B"]),
            '<p class="callout"><strong>Human check: planned, not yet completed on this new run.</strong> '
            "The agreed plan samples one event per question (60 events at the current scale). Two people "
            "independently judge the same sample, beginning with the author: each does 60 B ratings and 60 C "
            "text comparisons, without seeing the self-label, Gemini’s label or the other person’s label. "
            "C’s option comparison and D require no human labels. Until those annotations exist, this table "
            "shows consistency with one model judge, not human-validated accuracy.</p></section>",
            '<section id="c"><h2>C · Changes in the member’s own position</h2>'
            "<p>Here the question changes from <em>“Do you agree with the peer?”</em> to <em>“What is your own "
            "answer to the original survey question now?”</em> After a sampled debate reply, we ask the member to "
            "select one of the survey’s original options and write its full current position. These measurements "
            "are not fed back into the debate.</p>"
            "<p>We check the two parts separately. Code compares the selected option letters; this requires no "
            "judge. Gemini 3.8 Flash, the same single judge used for B, compares the member’s two written positions "
            "without seeing its selected letters or peer-agreement labels. It records: no substantive change; "
            "changed reasons, scope or qualifications with the same main conclusion; a changed main conclusion; "
            "or insufficient evidence to judge. Wording changes alone count as unchanged.</p>"
            "<p><strong>Previous → current:</strong> compare the current position with this same member’s position "
            "after its previous participation on this branch—not with the previous speaker’s position. If this is "
            "the member’s first turn, use its initial answer. <strong>Initial → sampled last:</strong> compare its "
            "initial answer with its last participation on a completed path, when that participation was sampled. "
            "A member’s last turn may occur before the path ends.</p>",
        ]
    )
    parts.extend(render_position_tables(s))
    parts.extend([examples(bank, questions), "</section>"])
    parts.extend(render_probabilities(s))
    parts.extend(render_quality_and_scope(s, bank, questions, plans))
    return "\n".join(parts) + "</main></body></html>"


def position_group_cells(row):
    if not row["text"]["n"]:
        return [row["group"], row["questions"], 0, "—", 0, "—", "—", "—", 0]
    cells = c_cells(row)
    return [cells[0], row["questions"], *cells[1:]]


def render_position_tables(s):
    headers = [
        "Current self-label / group",
        "Questions",
        "Sampled turns",
        "Option changed / comparable",
        "Option unavailable (n)",
        "Text unchanged",
        "Reasons / qualifications changed",
        "Main conclusion changed",
        "Text unjudgeable (n)",
    ]
    conditional = s["conditional_analysis"]
    parts = [
        '<h3 id="c-by-label">When the member reports each agreement level, does its own position change?</h3>'
        "<p>Group by the <strong>responding member’s self-reported agreement label on this exact debate "
        "reply</strong>. This is not the previous speaker’s label or Gemini’s label. Read a row as: among "
        "turns where the member says this about the peer, how often does its own position change from "
        "its previous participation? Table C1 combines T values only within the named agreement category; "
        "the following tables show T separately inside each category. These are observed associations, "
        "not effects caused by selecting a label.</p>",
        table(
            "Table C1. Previous own position → current, by the current reply’s self-label",
            headers,
            [position_group_cells(r) for r in s["C_by_self_label"]],
            "Reasons/qualifications changed means the main conclusion stays the same. Text categories are "
            "mutually exclusive; option changes are a separate measurement and must not be added to them. "
            "Fully disagree has only two sampled turns from two questions; its 50% option-change rate is "
            "one case out of two, not a reliable estimate for that category.",
        ),
        '<h3 id="c-by-turn-label">Within each agreement level, what happens at T = 1…5?</h3>'
        '<span id="c-by-turn"></span><p>T is the reply’s position on its branch, not the number of times '
        "its author has spoken. Each table below fixes the current reply’s agreement level. All comparisons "
        "are between the same member’s own positions. The second option column uses the initial position "
        "rather than the most recent one. T=1 is retained here as the first step of the trajectory, not as "
        "an independent one-round experiment.</p>",
    ]
    turn_rows = conditional["C_by_label_turn"]
    for index, name in enumerate(dict.fromkeys(r["label_name"] for r in turn_rows), 1):
        selected = [r for r in turn_rows if r["label_name"] == name]
        parts.append(
            table(
                f"Table C2.{index}. {name}: position changes by debate turn",
                [
                    "T",
                    "Questions",
                    "Sampled replies",
                    "Option: previous own → current",
                    "Option: initial own → current",
                    "Text unchanged",
                    "Reasons / qualifications changed",
                    "Main conclusion changed",
                    "Text unjudgeable (n)",
                ],
                [
                    [
                        f"T = {r['T']}",
                        r["questions"],
                        r["text"]["n"],
                        fraction(r["choice"]["changed"], r["choice"]["valid"]),
                        fraction(r["initial_choice"]["changed"], r["initial_choice"]["valid"]),
                        *[
                            fraction(r["text"]["counts"][key], r["text"]["n"])
                            for key in ("unchanged", "adjusted", "conclusion_changed")
                        ],
                        r["text"]["counts"]["unjudgeable"],
                    ]
                    for r in selected
                ],
                f"Only replies self-labeled {name}. Text compares previous own → current. No eligible observations "
                "means the cell has no comparable readings, not zero change. These sampled turns are not a complete "
                "longitudinal panel; differences across T do not alone isolate an extra round’s effect.",
            )
        )
    parts.extend(
        [
            '<h3 id="c-endpoints">Initial → sampled last position, grouped by the endpoint reply’s self-label</h3>',
            table(
                "Table C3. Initial → sampled last own position, by endpoint self-label",
                headers,
                [position_group_cells(r) for r in conditional["C_endpoint_by_label"]],
                "Only sampled last participations are included, not every member on every path. The group is the "
                "member’s self-label at that final sampled reply. Shared endpoints are counted once. This is a "
                "different reference from previous → current, and the counts must not be added together.",
            ),
            '<details id="c-by-participation"><summary>Within each agreement level: member’s participation count</summary>'
            "<p>If reply authors are A, B, A, the last reply is T=3 but A’s second participation. "
            "The comparison is still previous own position → current.</p>",
            table(
                "Table C4. Self-label × member’s participation count",
                headers,
                [
                    position_group_cells({**r, "group": f"{r['label_name']} · participation {r['participation']}"})
                    for r in conditional["C_by_label_participation"]
                ],
            ),
            "</details><details><summary>Within each agreement level: responding model</summary>",
            table(
                "Table C5. Self-label × responding model",
                headers,
                [
                    position_group_cells({**r, "group": f"{r['label_name']} · {r['model']}"})
                    for r in conditional["C_by_label_model"]
                ],
            ),
            "</details>",
        ]
    )
    return parts


def render_probability_interpretation(s):
    return (
        "<h3>How to read the magnitude</h3><p>Read filler and peer scores within the same self-label and "
        "model row. A one-point difference is one step on the coded scale, such as mostly agree (6) to "
        "completely agree (7). For example, 6.8 with filler and 6.3 with feedback gives −0.5 rating points: "
        "half a scale step, not a 50% drop in probability or 50% of members changing their minds. "
        "This is an illustration, not a pooled result. There is no validated practical-importance threshold.</p>"
        "<h3>How should B, C and D be compared?</h3>"
        "<p>Start with the same self-label category in each layer. <strong>B</strong> asks how the reply "
        "text is rated for that category. A reply can reject another member’s argument while preserving "
        'its own position. <strong>C</strong> checks own-position change by <a href="#c-by-label">self-label</a> '
        'and <a href="#c-by-turn-label">T within each self-label</a>. '
        "<strong>D</strong> measures probabilities, which can shift without the selected option changing. "
        "D covers only the two open models, so its support must be checked separately.</p>"
        "<p>C compares the member’s earlier answer with its answer after replying. D-text instead "
        "compares a peer message with filler before generating that reply. Its score difference cannot be "
        "converted into C’s percentage of changed answers, and reduced endorsement of one’s own text does "
        "not by itself establish adoption of the peer’s view.</p>"
    )


def render_probabilities(s):
    rows = s["conditional_analysis"]["D_by_label"]
    return [
        '<section id="d"><h2>D · Probability changes on the same debate data</h2>'
        "<p>D uses the same sampled debate turns for Qwen3.8-27B and Inkling, whose option probabilities are "
        "available. It does not cover Terra. There are two measurements: the probability of the member’s "
        "earlier selected survey option, and how strongly it endorses its earlier written position. "
        "Neither requires an external judge. Every result table retains the same current reply’s self-label; "
        "models are separated within each label rather than averaged across agreeing and disagreeing replies.</p>"
        "<h3>D-choice: does the earlier selected option become less likely?</h3>"
        "<p>C records which option the member selects. Here we also use the option probabilities from that "
        "same response. Keep the earlier selected option fixed and subtract its earlier probability from "
        "its current probability. This can reveal a change even if the member still chooses that option.</p>"
        '<div class="formula">Δchoice = 100 × [p<sub>current</sub>(reference option) '
        "− p<sub>reference</sub>(reference option)]</div>"
        "<p>“Reference option” means the member’s previously selected option, or its initial option for an "
        "initial-to-current comparison. For example, 80% before and 77% now would give −3 percentage points "
        "(pp); it would not mean that 3% of members switched answers. This example only illustrates the units. "
        "The survey options differ by question and are not mapped onto a shared numerical stance scale.</p>",
        d_choice_table(rows, "Table D1. Previous → current option probability, by self-label and model"),
        "<details><summary>By self-label: initial → current and initial → sampled last option probabilities</summary>",
        d_choice_table(rows, "Table D1a. Initial → current, by current self-label and model", "D_choice_initial_pp"),
        d_choice_table(
            rows, "Table D1b. Initial → sampled last, by endpoint self-label and model", "D_choice_final_pp"
        ),
        "</details>",
        "<h3>D-text: does the peer’s message reduce endorsement of the member’s earlier written position?</h3>"
        "<p>We quote the member’s exact earlier full position and ask: <em>“How much do you now agree with "
        "your previously recorded full position quoted above, considering it as a whole?”</em> We ask this "
        "in two separate copies of the same earlier conversation: one adds the actual incoming peer message; "
        "the other replaces that message with repetitions of "
        "<q>The moderator noted the time and reminded participants that further remarks would follow in due course.</q> "
        "This filler approximately matches the peer message’s token length without discussing the topic. "
        "It gives a comparison for adding text to the conversation without adding the peer’s argument. "
        "The model does not first write a debate reply in either copy, and these measurements never enter "
        "the actual debate history.</p>"
        "<p>The seven ratings are: 1 = completely disagree, 2 = mostly disagree, 3 = somewhat disagree, "
        "4 = neither agree nor disagree, 5 = somewhat agree, 6 = mostly agree, 7 = completely agree. "
        "These rate the member’s <strong>own earlier text</strong>, unlike A/B’s four labels about the peer. "
        "We read the probability of each rating and compute a probability-weighted average, rather than "
        "using only the most likely rating:</p>"
        '<div class="formula">score = Σ<sub>k=1…7</sub> k × p(k)<br>'
        "Δown-text = score<sub>with peer message</sub> − score<sub>with filler</sub></div>"
        "<p>Here p(k) is the probability assigned to rating k in that input, normalized over the seven "
        "rating letters. A negative difference means lower endorsement of the same earlier text with the "
        "peer message than with filler. The peer message may agree or disagree; it is not necessarily a "
        "counterargument. This average uses equally spaced numerical codes for the seven ordered ratings, "
        "not a calibrated percentage of belief.</p>",
        d_text_table(rows, "Table D2. Endorsement of own prior text (1–7), by self-label and model"),
        "<p>Within each row, average paired scores within question first, then give supported questions equal weight. "
        "The quoted text is the previous own position (initial on a first participation). No separate "
        "fixed-initial-text probe was run at every later event. No top-two margin or entropy matching is used.</p>",
        render_probability_interpretation(s),
        "<details><summary>Within each agreement level and model: debate turn T</summary>",
        d_choice_table(
            [
                {**r, "label_name": f"{r['label_name']} · T={r['T']}"}
                for r in s["conditional_analysis"]["D_by_label_turn"]
            ],
            "Table D3a. Self-label × T × model: previous → current option probability",
        ),
        d_text_table(
            [
                {**r, "label_name": f"{r['label_name']} · T={r['T']}"}
                for r in s["conditional_analysis"]["D_by_label_turn"]
            ],
            "Table D3b. Self-label × T × model: own-text endorsement",
        ),
        "</details>",
        "<details><summary>Probability-readout diagnostics</summary>",
        table(
            "Table D4. Probability-readout diagnostics",
            [
                "Member",
                "Read type",
                "Reads",
                "Zero-fill used",
                "Min. candidate mass",
                "Median mass",
                "Candidate mass < 0.5 (n)",
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
                    r["candidate_mass_below_half"],
                    fraction(r["max_probability_ge_99"], r["n"]),
                ]
                for r in s["probability_audit"]
            ],
            "Original top-20 scores at T=1; absent candidates are approximated as zero, then renormalized. "
            "Candidate mass is measured before normalization (a lower bound if candidates are absent). "
            "Low-mass reads are retained under the agreed rule; the 0.5 diagnostic is not an exclusion gate.",
        ),
        '</details><p class="callout">These are context-conditioned output probabilities, not calibrated confidence '
        "or hidden beliefs. They are conditional on the candidate options after normalization. "
        f"There are {len(s['low_mass_choice_reads'])} choice reads with returned candidate mass below 0.5. "
        "Concentration and top-20 truncation limit sensitivity. Formal debate uses reasoning; C/D direct reads do not.</p></section>",
    ]


def render_quality_and_scope(s, bank, questions, plans):
    n = s["counts"]
    counts = [
        ["Questions / debates", f"{n['questions']} / {n['debates']}", "One mixed roster; three tones"],
        [
            "Completed / failed paths",
            f"{s['trajectories']['success']} / {s['trajectories']['failed']}",
            "Six paths per question × tone; shared prefixes",
        ],
        [
            "Unique replies / sampled events",
            f"{n['formal_replies']} / {n['sampled_events']}",
            "Eight presampled events per question",
        ],
        [
            "C readings / unique judged text pairs",
            f"{n['C_readings']} / {n['C_judgments']}",
            f"{s['unselectable_readings']} readings do not select an option",
        ],
        [
            "D-choice readings / D-text pairs",
            f"{n['D_choice_readings']} / {n['D_text_pairs']}",
            "Open-weight members only",
        ],
        [
            "Syntheses / E judgments",
            f"{s['planned_counts']['syntheses']} / {n['E_judgments']}",
            "One no-debate synthesis per question; both orders for each pair",
        ],
        [
            "Valid scientific requests",
            s["planned_counts"]["logical_calls"],
            "All freshly generated; no imported responses",
        ],
        [
            "Additional retry attempts / recovered requests",
            f"{s['request_retries']['additional_attempts']} / {s['request_retries']['recovered_logical_requests']}",
            f"{s['request_retries']['exhausted_logical_requests']} exhausted; original failures remain archived",
        ],
    ]
    inventory = [
        [
            i,
            bank[q["question_id"]]["text"],
            "; ".join(f"{chr(65 + j)}: {v}" for j, v in enumerate(bank[q["question_id"]]["options"])),
            len(plans[q["question_id"]]["route"]) * 3,
        ]
        for i, q in enumerate(questions, 1)
    ]
    return [
        '<section id="e"><h2>E · Final answers with and without debate</h2>'
        "<p>The same Terra chairman synthesizes either the original three independent answers or those same answers "
        "plus the completed branch-isolated discussion. The no-debate synthesis is shared across tones within a question. "
        "Gemini compares each pair in both orders and must choose one answer; there is no explicit tie option.</p>",
        table(
            "Table E1. Blind final-answer preference",
            [
                "Tone",
                "Pairs",
                "Debate in both orders",
                "No debate in both orders",
                "Preference changes after reversal",
                "Debate score (%) [95% interval]",
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
            "Pair score: 1 = debate in both orders; 0 = no debate in both; 0.5 = inconsistent preferences, not an explicit tie. "
            "All pairs remain included. Pooled intervals average tones within each question before resampling questions.",
        ),
        "<p>The hostile arm has the strongest observed preference for debate in this run. This does not establish "
        "a general quality advantage: these are judgments from one model, rubric and roster, without a new human "
        "validation or damaged-answer sensitivity check.</p>"
        "<details><summary>Final-answer lengths</summary>",
        table(
            "Table E2. Mean synthesis length (whitespace-separated words)",
            ["Tone", "No debate", "Debate"],
            [[r["tone"], f"{r['baseline_words']:.1f}", f"{r['debate_words']:.1f}"] for r in s["E"][:3]],
            "The judge is instructed not to reward length or style alone; that does not rule out such bias.",
        ),
        '</details></section><section id="scope"><h2>Scope, uncertainty and completion</h2>'
        "<p>Routing, question selection, models, tones, sampling and probability rules match the earlier run. "
        "Only the A/B agreement rubric and its output labels changed. All responses were regenerated, including initial "
        "answers: changes from the reference run combine a protocol change with generation variability, not an isolated "
        "causal effect of the word “leaning.” No old and new results are pooled.</p>"
        "<p>B/C/D sampling takes one event at each depth 1–5, then three more with depth weights 1:1:1:1:1.5, "
        "pooling tones. Tables describe this sample without inverse-probability weighting. A/B/C percentages use pooled "
        "counts and explicit denominators. D averages within each supported question before giving questions equal weight.</p>"
        "<p>The A contrast and E scores also weight questions equally. Intervals use 10,000 question-level percentile "
        "bootstrap resamples (fixed seed 20260926); they are descriptive, pointwise and not multiplicity-adjusted. "
        "Repeated replies and overlapping paths are not independent questions.</p>",
        table("Table S1. Collection and analysis units", ["Item", "Count", "Interpretation"], counts),
        f"<p>The run completed at {html.escape(s['completed_at_utc'])}. Its token-based cost estimate, including "
        f"retries, is <strong>US${s['charged_or_reserved_usd']:.2f}</strong>; this is not a provider invoice. "
        f"There are {s['unselectable_readings']} C responses without a selected original option. Those are semantic "
        "outcomes, not unresolved API failures; relevant numeric comparisons show their available denominators.</p>"
        '<p class="callout">The study does not establish that every C change was caused by peer content: there is no '
        "matched no-peer C resampling arm. It also does not establish lasting beliefs, human-validated judge accuracy "
        "or effects across other committee rosters. The current run includes no extra instruction ablation or reference-answer benchmark.</p>",
        '<details class="inventory"><summary>The 60 questions and original options</summary>'
        + table("Question inventory", ["#", "Question", "Original options", "Unique replies, all tones"], inventory)
        + "</details>",
        "<details><summary>Version and reproducibility</summary>"
        f"<p>Prompt version: <code>{html.escape(s['prompt_version'])}</code><br>"
        f"Rubric: <code>{html.escape(s['agreement_rubric']['version'])}</code><br>"
        f"Source: <code>{html.escape(s['source_directory'])}</code><br>"
        f"Manifest SHA-256: <code>{s['manifest_sha256']}</code></p>"
        "<p>This report was generated offline, without model calls or changes to raw results. "
        "The companion JSON contains exact aggregates, denominators, the frozen rubric and hashes of every source question file. "
        "Generator: <code>python -m llm_committee.pivot.leaning_results_report RUN_DIRECTORY OUTPUT.html</code>.</p>"
        '<p>Table style follows <a href="https://arxiv.org/pdf/2609.08016">arXiv:2609.08016</a>, '
        "but the old metrics and data are not included. "
        '<a href="main-mixed-results-2026-09-26.html">Earlier partial-agreement run: reference only</a>.</p>'
        "</details></section><footer>Leaning-label rerun · fresh data only · single mixed committee · "
        "all content and styles embedded for offline reading.</footer>",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--feedback-run", type=Path, help="Embed this completed supplement in the same HTML")
    parser.add_argument("--dyadic-run", type=Path, help="Embed the separate two-member supplement in the same HTML")
    parser.add_argument("--replace", action="store_true", help="Update an existing report for the same main run")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.suffix != ".html" or source in output.parents:
        parser.error("Use an .html output outside the immutable source directory")
    existing = None
    if output.exists() or output.with_suffix(".json").exists():
        if not args.replace:
            parser.error("Refusing to overwrite an existing report; choose a new path or use --replace")
        if not (output.is_file() and output.with_suffix(".json").is_file()):
            parser.error("Replacement requires both an existing HTML and its JSON companion")
        existing = json.loads(output.with_suffix(".json").read_text())
        if existing.get("source_directory") != str(source):
            parser.error("Cannot replace a report for a different main study")
        if "forced_feedback" in existing.get("supplements", {}) and args.feedback_run is None:
            parser.error("Specify --feedback-run to preserve the existing supplement")
        if "dyadic" in existing.get("supplements", {}) and args.dyadic_run is None:
            parser.error("Specify --dyadic-run to preserve the existing two-member supplement")
    summary, bank, questions, plans = analyze_leaning(source)
    if existing and any(v != summary.get(k) for k, v in existing.items() if k != "supplements"):
        parser.error("Existing main-study aggregates differ; refusing an unrelated replacement")
    document = render_leaning(summary, bank, questions, plans)
    if args.feedback_run is not None:
        document, summary = include_feedback(document, summary, args.feedback_run)
    if args.dyadic_run is not None:
        from .dyadic_report import include_dyadic

        document, summary = include_dyadic(document, summary, args.dyadic_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
