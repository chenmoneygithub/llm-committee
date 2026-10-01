"""Strong-label results only; archive the fully-label report without converting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from . import prompts
from .dyadic import PAIRS, make_route
from .dyadic_report import MODELS, REQUIRED, conditional_groups, flatten, label
from .forced_feedback_report import esc, question_mean, rate, table
from .models import TONES
from .results_report import CSS
from .strong_agreement import AGREEMENT, AGREEMENT_DEFINITIONS
from .strong_study import ARMS, VERSION
from .turn_tone import STREAM
from .turn_tone_fork_report import CSS as FORK_CSS
from .turn_tone_fork_report import case_detail
from .turn_tone_report import EXTRA_CSS


def load_run(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] != VERSION or report["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Need a completed fresh strong-label run")
    records = []
    route = make_route()
    for plan in manifest["plans"]:
        record = json.loads((source / "questions" / f"{plan['question_id']}.json").read_text())
        assert record["question_id"] == plan["question_id"]
        for arm in ARMS:
            branch, p = record["branches"][arm], plan["arms"][arm]
            assert branch["tone_schedule"] == p["tone_schedule"]
            assert branch["initial_answers"] == record["initial_answers"]
            assert branch["probe_tone"] == "neutral"
            assert [e["id"] for e in branch["events"]] == [e["id"] for e in p["events"]]
            history = branch["formal_replies"][STREAM]
            assert all(v["agreement"] in (*AGREEMENT, None) for v in history.values())
            for event in branch["events"]:
                node = route.get(event["node_id"])
                assert event["previous_peer_self_label"] == (history.get(node.parent) or {}).get("agreement")
                assert event["current_self_label"] == (history.get(node.id) or {}).get("agreement")
                assert event["current_tone"] == p["tone_schedule"][node.id]
                assert event["previous_tone"] == p["tone_schedule"][node.parent]
                assert (event["B"] or {}).get("label") in (*AGREEMENT, "no_position", "unjudgeable", None)
        a, b = (record["branches"][arm] for arm in ARMS)
        for node in route.nodes:
            if node.depth <= 2:
                assert a["formal_replies"][STREAM].get(node.id) == b["formal_replies"][STREAM].get(node.id)
        for first, second in zip(a["events"], b["events"], strict=True):
            if first["T"] <= 3:
                assert first["D_text"] == second["D_text"]
            if first["T"] == 2:
                assert first == second
        records.append(record)
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        ledger = db.execute("SELECT status, COUNT(*), SUM(charge) FROM calls GROUP BY status").fetchall()
    report["cost_accounting"] = {
        "received_response_estimate_usd": sum(v for s, _, v in ledger if s in ("completed", "received", "invalid")),
        "unresolved_reservations_usd": sum(v for s, _, v in ledger if s not in ("completed", "received", "invalid")),
        "unresolved_attempts": sum(n for s, n, _ in ledger if s not in ("completed", "received", "invalid")),
    }
    return manifest, report, records


def collect(records):
    formal, rows = [], []
    for record in records:
        for arm in ARMS:
            branch = record["branches"][arm]
            for nid, reply in branch["formal_replies"][STREAM].items():
                t = int(nid.rsplit("-", 1)[1])
                if arm == "alternate" and t <= 2:
                    continue
                formal.append(
                    {
                        "question_id": record["question_id"],
                        "node_id": nid,
                        "T": t,
                        "assignment": "Shared T1/T2" if t <= 2 else arm.title(),
                        "current_tone": branch["tone_schedule"][nid],
                        "self_label": reply["agreement"],
                    }
                )
            for row in flatten([branch]):
                if arm == "alternate" and row["T"] == 2:
                    continue
                rows.append({**row, "assignment": "Shared T1/T2" if row["T"] == 2 else arm.title()})
    assert len({(r["question_id"], r["assignment"], r["node_id"]) for r in formal}) == len(formal)
    assert len({(r["question_id"], r["assignment"], r["node_id"]) for r in rows}) == len(rows)
    return formal, rows


def b_rows(rows):
    result = []
    for current in (*AGREEMENT, None):
        group = [r for r in rows if r["current_self_label"] == current]
        if current is None and not group:
            continue
        counts = Counter((r["B"] or {}).get("label") for r in group)
        completed = len(group) - counts[None]
        result.append(
            [
                label(current),
                len(group),
                *[rate(counts[v], completed) for v in AGREEMENT],
                counts["no_position"],
                counts["unjudgeable"],
                counts[None],
            ]
        )
    return result


def c_summary(rows, *, endpoint=False):
    """Pool distinct sampled replies within the two-label group, never percentages."""
    if endpoint:
        rows = [r for r in rows if r["final_pair"]]
    choice_field = "C_initial_choice_changed" if endpoint else "C_choice_changed"
    text_field = "endpoint_text_label" if endpoint else "text_label"
    result = []
    for key, group in conditional_groups(rows, REQUIRED, agreement_order=AGREEMENT):
        choices = [r[choice_field] for r in group if r[choice_field] is not None]
        text = Counter(r[text_field] for r in group)
        text_compared = sum(text[v] for v in ("unchanged", "adjusted", "conclusion_changed"))
        result.append(
            {
                **dict(zip(REQUIRED, key, strict=True)),
                "questions": len({r["question_id"] for r in group}),
                "replies": len(group),
                "option_compared": len(choices),
                "option_changed": sum(choices),
                "text_compared": text_compared,
                "text_unchanged": text["unchanged"],
                "reasons_adjusted": text["adjusted"],
                "conclusion_changed": text["conclusion_changed"],
                "option_excluded": len(group) - len(choices),
                "text_excluded": len(group) - text_compared,
            }
        )
    return result


def compact_c_table(groups):
    def cell(n, d):
        return rate(n, d) if d else "—"

    data = [
        [
            label(g["previous_peer_self_label"]),
            label(g["current_self_label"]),
            g["replies"],
            cell(g["option_changed"], g["option_compared"]),
            cell(g["reasons_adjusted"], g["text_compared"]),
            cell(g["conclusion_changed"], g["text_compared"]),
        ]
        for g in groups
    ]
    result = table(
        [
            "Peer's preceding label",
            "Receiver's current label",
            "Replies",
            "Option changed",
            "Reasons adjusted",
            "Conclusion changed",
        ],
        data,
    )
    option_excluded = sum(g["option_excluded"] for g in groups)
    text_excluded = sum(g["text_excluded"] for g in groups)
    if option_excluded or text_excluded:
        result += (
            f'<p class="muted">Not included in the respective rates: {option_excluded} option comparison(s) '
            f"without two selected options; {text_excluded} text comparison(s) without a classifiable rating. "
            "Each cell shows its own denominator; omitted comparisons are not counted as unchanged.</p>"
        )
    return result


def d_summary(rows, *, kind):
    """Reaggregate matched readings within question, keeping labels and model fixed."""
    fields = {
        "choice": ("choice_p_before", "choice_p_after", "D_choice_adjacent"),
        "text": ("D_control", "D_argument", "D_text_delta"),
        "initial": ("initial_p_before", "initial_p_after", "D_choice_initial_to_current"),
    }[kind]
    result = []
    for key, group in conditional_groups(
        [r for r in rows if r["member"] in (1, 2)], (*REQUIRED, "member"), agreement_order=AGREEMENT
    ):
        compared = [r for r in group if all(r.get(f) is not None for f in fields)]
        values = [question_mean(compared, f) for f in fields]
        choices = [r["C_choice_changed"] for r in compared if r["C_choice_changed"] is not None]
        result.append(
            {
                **dict(zip((*REQUIRED, "member"), key, strict=True)),
                "kind": kind,
                "sampled": len(group),
                "compared": len(compared),
                "questions": values[0][2],
                "excluded": len(group) - len(compared),
                "reference_mean": values[0][0],
                "updated_mean": values[1][0],
                "change_mean": values[2][0],
                "adjacent_choice_compared": len(choices),
                "adjacent_choice_changed": sum(choices),
            }
        )
    return result


def d_number(value, *, signed=False):
    if value is None:
        return "—"
    if signed and value and round(value, 2) == 0:
        return "≈0.00"
    return format(value, "+.2f" if signed else ".2f").replace("-", "−")


def compact_d_tables(groups, *, kind):
    metric, change = {
        "choice": ("Old option probability: before → after", "Change (pp)"),
        "text": ("Own-text rating: filler → peer", "Difference (rating points)"),
        "initial": ("Initial option probability: initial → current", "Change (pp)"),
    }[kind]
    sections = []
    for member in (1, 2):
        subset = [g for g in groups if g["member"] == member]
        data = []
        for g in subset:
            suffix = "%" if kind != "text" else ""
            values = (
                f"{d_number(g['reference_mean'])}{suffix} → {d_number(g['updated_mean'])}{suffix}"
                if g["compared"]
                else "—"
            )
            data.append(
                [
                    label(g["previous_peer_self_label"]),
                    label(g["current_self_label"]),
                    f"{g['compared']} comparison{'s' if g['compared'] != 1 else ''} "
                    f"({g['questions']} question{'s' if g['questions'] != 1 else ''})",
                    values,
                    d_number(g["change_mean"], signed=True),
                ]
            )
        sections.append(
            f"<h4>{esc(MODELS[member])}</h4>"
            + table(["Peer's preceding label", "Receiver's current label", "Sample", metric, change], data)
        )
        excluded = sum(g["excluded"] for g in subset)
        if excluded:
            sections.append(
                f'<p class="muted">{excluded} sampled comparison(s) lack a usable paired reading and are '
                "excluded from the means, not counted as zero change. Sample counts show the included comparisons.</p>"
            )
    return "".join(sections)


def d_choice_examples(groups):
    """Explain two identifiable rows; counts refer to exactly the D1 subset."""
    examples = []
    for previous, current in (("strongly_agree", "strongly_agree"), ("leaning_disagree", "leaning_agree")):
        g = next(
            (
                g
                for g in groups
                if g["member"] == 1
                and g["previous_peer_self_label"] == previous
                and g["current_self_label"] == current
                and g["compared"]
            ),
            None,
        )
        if g:
            examples.append(
                f"<p>Qwen, peer <em>{esc(label(previous).lower())}</em> / receiver "
                f"<em>{esc(label(current).lower())}</em>: the previous option's question-weighted mean probability "
                f"is {d_number(g['reference_mean'])}% before and {d_number(g['updated_mean'])}% after, "
                f"a change of {d_number(g['change_mean'], signed=True)} pp. "
                f"C records {g['adjacent_choice_changed']}/{g['adjacent_choice_compared']} actual option changes "
                f"among these same replies, drawn from {g['questions']} questions. "
                "The probability difference and the fraction of replies that switch options are different measurements.</p>"
            )
    return "".join(examples)


def render(manifest, report, records, *, legacy_href, supplementary_href):
    formal, rows = collect(records)
    # T3 own-text inputs are identical, before either different-tone reply. Count once,
    # under the original reply's outcome label, never as an independent alternate reading.
    dtext = [r for r in rows if not (r["assignment"] == "Alternate" and r["T"] == 3)]
    a_rows, a_detail = [], []
    for tone in TONES:
        subset = [r for r in formal if r["current_tone"] == tone]
        counts = Counter(r["self_label"] for r in subset)
        a_rows.append([tone, len(subset), *[rate(counts[v], len(subset)) for v in AGREEMENT], counts[None]])
        for assignment in ("Shared T1/T2", "Original", "Alternate"):
            for t in range(1, 5):
                group = [r for r in subset if r["assignment"] == assignment and r["T"] == t]
                if group:
                    c = Counter(r["self_label"] for r in group)
                    a_detail.append(
                        [assignment, t, tone, len(group), *[rate(c[v], len(group)) for v in AGREEMENT], c[None]]
                    )
    b_data = b_rows(rows)
    a_notes = []
    for tone in TONES:
        group = [r for r in formal if r["current_tone"] == tone]
        counts = Counter(r["self_label"] for r in group)
        a_notes.append(
            f"{tone.title()}: strongly agree {rate(counts['strongly_agree'], len(group))}; "
            f"strongly disagree {rate(counts['strongly_disagree'], len(group))}."
        )
    b_notes = []
    for current in AGREEMENT:
        group = [r for r in rows if r["current_self_label"] == current]
        valid = [r for r in group if r["B"]]
        if valid:
            matched = sum(r["B"]["label"] == current for r in valid)
            b_notes.append(
                f"For self-reported {label(current).lower()}, Gemini assigns the same label in {rate(matched, len(valid))}."
            )
        else:
            all_replies = [r for r in formal if r["self_label"] == current]
            if all_replies and all(r["T"] == 1 for r in all_replies):
                b_notes.append(
                    f"All {len(all_replies)} self-reported {label(current).lower()} replies occur at T1, outside the B/C/D sample; no current-receiver results are available for this label."
                )
            else:
                b_notes.append(f"No completed B judgments for self-reported {label(current).lower()} in this sample.")
    b_header = [
        "Receiver self-label",
        "Sampled replies",
        *[f"Gemini text: {label(v)}" for v in AGREEMENT],
        "No position",
        "Unjudgeable",
        "Missing judge",
    ]
    coverage = []
    for previous in (*AGREEMENT, None):
        subset = [r for r in rows if r["previous_peer_self_label"] == previous]
        if previous is None and not subset:
            continue
        coverage.append(
            [
                label(previous),
                *[sum(r["current_self_label"] == v for r in subset) for v in AGREEMENT],
                sum(r["current_self_label"] is None for r in subset),
            ]
        )
    c_adjacent = c_summary(rows)
    c_endpoint = c_summary(rows, endpoint=True)
    d_choice = d_summary(rows, kind="choice")
    d_own_text = d_summary(dtext, kind="text")
    d_initial = d_summary(rows, kind="initial")
    transitions, tone_pairs = [], []
    for t in (3, 4):
        counts, tones = Counter(), Counter()
        for record in records:
            a, b = (record["branches"][arm] for arm in ARMS)
            for pair in PAIRS:
                nid = f"{pair}-{t}"
                counts[
                    (a["formal_replies"][STREAM].get(nid) or {}).get("agreement"),
                    (b["formal_replies"][STREAM].get(nid) or {}).get("agreement"),
                ] += 1
                tones[a["tone_schedule"][nid], b["tone_schedule"][nid]] += 1
        for previous in (*AGREEMENT, None):
            if previous is None and not any(a is None for a, b in counts):
                continue
            transitions.append([t, label(previous), *[counts[previous, v] for v in AGREEMENT], counts[previous, None]])
        tone_pairs.extend([t, a, b, n] for (a, b), n in sorted(tones.items()))
    cost = report["cost_accounting"]
    billing = (
        f"Recorded responses: token-estimated US${cost['received_response_estimate_usd']:.2f}, not a provider invoice."
    )
    if cost["unresolved_attempts"]:
        billing += f" {cost['unresolved_attempts']} unresolved-billing attempt(s): US${cost['unresolved_reservations_usd']:.2f} reserved, not confirmed spending."
    mock = (
        '<p class="callout">OFFLINE MOCK — synthetic fixtures, not scientific results.</p>'
        if manifest["kind"] == "offline_mock"
        else ""
    )
    title = "Strongly / leaning labels · turn-level-tone dyadic pilot"
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{CSS}{EXTRA_CSS}{FORK_CSS}</style></head><body><main>
<header><p class="eyebrow">Fresh rerun · 20 questions · mixed models · paired continuations</p><h1>{title}</h1>{mock}
<p class="lede">Tone is assigned to each debate reply, not globally to a conversation. New initial answers, debates and judgments
use the strongly/leaning rubric. This is not a relabeling of the old fully-label results.</p>
<p class="status">{report["completed_questions"]}/20 questions complete; {len(formal)}/360 unique replies;
{len(rows)}/267 sampled B/C events. {report["trajectory_statuses"].get("success", 0)} successful branches,
{report["trajectory_statuses"].get("failed", 0)} failed branches (shared prefixes counted once). {esc(billing)}</p>
<p><a href="{esc(legacy_href)}">Previous fully-label pilot (reference only)</a> ·
<a href="{esc(supplementary_href)}">Earlier fixed-tone supplementary experiments</a>. Neither is pooled here.</p></header>
<nav><a href="#design">Design</a><a href="#a">A · Self-labels</a><a href="#b">B · Text check</a>
<a href="#c">C · Position changes</a><a href="#d">D · Probabilities</a><a href="#paired">Paired paths</a><a href="#cases">All cases</a></nav>
<section id="design"><h2>What was run?</h2>
<p>A = GPT-5.6 Terra; B = Qwen3.8-27B; C = Inkling. Each question has three exclusive two-member paths:
AB, CA, BC (initial sender → first receiver). For AB, A supplies its initial position and replies are B → A → B → A.
Each member's new independent initial answer is shared across its pairings, but later histories are separate.</p>
<p>Each path has two continuations: an original tone assignment and an alternate assignment. They share T1/T2 exactly;
at both T3 and T4 the alternate uses a different tone. Thus 20 questions × 3 pairs × (2 shared + 2 original + 2 alternate)
= 360 unique replies, 120 under each tone. T is the reply's index on a path, not the member's participation count.</p>
<p>The 20 questions, original options, routing, tone schedules and sampled nodes are frozen from the previous local-tone pilot.
Only the four-label rubric changes. Both continuations and the 60 initial answers were generated afresh. Only the mixed-model roster
was run; no same-model/family roster, three-member discussion or final-answer synthesis was started.</p>
<p>Only the current reply's private tone instruction is supplied. Earlier tone instructions and self-label metadata are hidden;
earlier discussion text remains visible. Reasoning is enabled for debate and disabled for direct, neutral C/D probes.
Measurement outputs never enter the debate history.</p>
<p>Eight of nine eligible pair × T positions at T2/3/4 are presampled per question, with the same selection for B/C/D.
This gives 160 original events plus 107 alternate T3/T4 events. The 53 shared T2 events count once. T1 stays in A;
it has no preceding reply label, so it is outside the two-label B/C/D sample. D is available only for Qwen and Inkling.</p>
<p>There are 20 independent questions, not 267 independent questions. Tables are descriptive pilot results; many label/turn/model
cells are small. No significance claim or direct old/new causal comparison is made.</p></section>
<section id="a"><h2>A · How does the reply label its incoming peer message?</h2>
<p>All 360 unique replies, counted once. Rows use the current reply's assigned tone; columns are its self-reported agreement
with the peer, not its choice on the survey question. Counts and percentages are shown for all four labels.</p>
{table(["Current assigned tone", "Replies", *[label(v) for v in AGREEMENT], "Not reported"], a_rows)}
<p>{esc(" ".join(a_notes))}</p>
<details><summary>Four-label definitions — used by both debaters and the B judge</summary>
<p>“Strongly” describes substantive support or opposition, not aggression, certainty or agreement with every detail.
“Leaning” describes qualified, on-balance support or opposition.</p>
{table(["Label", "Definition"], [[label(v), AGREEMENT_DEFINITIONS[v]] for v in AGREEMENT])}</details>
<details><summary>A · Keep assignment, turn and current tone separate</summary>
{table(["Assignment", "T", "Current tone", "Replies", *[label(v) for v in AGREEMENT], "Not reported"], a_detail)}</details></section>
<section id="b"><h2>B · Does the reply text support its self-label?</h2>
<p>One external judge, Gemini 3.8 Flash, reads the incoming peer message and reply, without self-labels or private tone instructions.
Rows are the receiver's self-report; columns are Gemini's reading of the text using the same four definitions. Each percentage is
within that row's completed judgments. For example, a strongly-agree row and leaning-disagree column means the member reported
strong agreement but Gemini read its reply as qualified opposition. This is a judge comparison, not human-validated accuracy.</p>
{table(b_header, b_data)}
<p>{esc(" ".join(b_notes))}</p>
<details><summary>B · Separate the shared prefix and two continuations</summary>
{"".join("<h3>" + assignment + "</h3>" + table(b_header, b_rows([r for r in rows if r["assignment"] == assignment])) for assignment in ("Shared T1/T2", "Original", "Alternate"))}</details></section>
<section id="c"><h2>C · Does the member's own position change?</h2>
<p>For each sampled reply, compare this member's current position with its position after its own previous participation on the
same branch (or its initial position if this is its first reply). Code checks whether the selected survey option changes.
Gemini 3.8 Flash separately compares the full position texts: “Reasons adjusted” means revised reasons or qualifications without
a different main conclusion; “Conclusion changed” means a different main conclusion. All other classifiable text comparisons
are unchanged. Option changes and text judgments are separate measurements, not counts to add together.</p>
<p>Each row groups replies by two labels: how the incoming peer labeled its response to the receiver's prior contribution,
and how the receiver labels its current response to that peer. Turns T2–T4, all three model pairs and both tone assignments are
combined within each label pair. The shared T2 reply counts once; each distinct continuation reply counts once.</p>
{compact_c_table(c_adjacent)}
<p>Cells show changed / compared replies and the percentage. Only observed label pairs appear; absent combinations are not zero-change
results. Rates describe sampled replies, not independent questions or causal effects of the labels.</p>
<details><summary>C · Initial position → sampled final own participation</summary>
<p>T3/T4 are each member's last own reply on its branch. This compares against its independent initial position, not just its previous reply.</p>
{compact_c_table(c_endpoint)}</details>
<p><a href="#cases">Inspect the full turn-by-turn cases</a> to explore reactions to disagreement at different points in a path.
Turn indices, tone assignments, both labels and position texts remain available there and in the underlying data.
Such examples illustrate the sequence; they do not by themselves establish an effect of turn index.</p></section>
<section id="d"><h2>D · Probability readouts on the same sampled contexts</h2>
<p>C checks whether the chosen survey option or expressed position changes. D adds the model's output probabilities:
D1 follows the previously chosen option; D2 asks how much the model still endorses its own previous full position text.
These readings come from Qwen and Inkling, on the same sampled debate contexts as C; Terra does not provide these readouts.</p>
<p>Each model has a separate table, grouped only by the preceding peer's label and the receiver's current label, as in C.
Turns and tone assignments are combined within each group. For each number, we first average paired readings within a question,
then give each represented question equal weight. The sample column shows how many comparisons and distinct questions contribute.
These are descriptive associations with observed reply labels, not causal effects of agreeing or disagreeing.</p>
<h3>D1 · Does the model still favor the option it previously chose?</h3>
<p>At each C measurement, the model is asked to select an option from the original question and then explain its position.
We record the probabilities of the option letters <em>before the explanation is generated</em>. For one comparison:</p>
<ol><li>Take the option selected at this member's previous own measurement on the same branch (or its initial measurement).
Call it the <strong>old option</strong>, k. It can differ across questions and comparisons.</li>
<li>Read the probability of k at that earlier measurement, and again after the next peer message and the member's new debate reply.</li>
<li>Subtract the earlier probability from the later probability. Keep tracking k even if the model now selects a different option.</li></ol>
<div class="formula">Change (percentage points) = 100 × [p<sub>after</sub>(k) − p<sub>before</sub>(k)].</div>
<p><strong>How to read the last two columns:</strong> “80% → 70%” means the same old option had 80% of the normalized option probability
at the previous measurement and 70% at the current one. “−10 pp” means 10 percentage points less probability on that option;
it does <em>not</em> mean 10% of members changed their answer. A positive value means more probability on the old option;
a negative value means less. The selected option can remain unchanged in either case.</p>
{compact_d_tables(d_choice, kind="choice")}
<h4>Reading the numbers alongside C</h4>
{d_choice_examples(d_choice)}
<p>A fraction-of-a-percentage-point change near 100% is a small absolute probability shift; a drop of tens of percentage points
is much larger on this output-probability scale. Neither is calibrated as a change in human-like confidence or lasting belief.
Many readings are already near 100%, leaving little room for increases. Group means can also hide offsetting movements.
D1 does not by itself show movement toward the peer, or isolate the peer message's effect from the member's own generated reply.</p>
<h3>D2 · Does the peer message reduce endorsement of the member's previous full text?</h3>
<p>This is a different comparison, made <em>before</em> the new debate reply. Fix the member's previously recorded position paragraph
verbatim. In two copies of the same preceding context, add either the incoming peer message or a topic-free filler message, then ask:
“How much do you now agree with your previously recorded full position quoted above, considering it as a whole?”
The filler repeats “{esc(prompts.FILLER_SENTENCE)}” to approximately match the peer message's token length.
It provides a comparison for extending the context without making an argument about the question.</p>
<p>The model assigns probabilities to seven ratings, from 1 = completely disagree, through 4 = neither agree nor disagree,
to 7 = completely agree. We use all seven probabilities, not just the most likely rating:</p>
<div class="formula">Mean endorsement rating r = ∑<sub>k=1</sub><sup>7</sup> k × p(k).<br>
Difference = r<sub>peer</sub> − r<sub>filler</sub>.</div>
<p>For example, equal probability on ratings 6 and 7 gives r = 6.5. In the tables, “6.8 → 6.6” means a mean rating of 6.8 with filler
and 6.6 with the peer message: the difference is −0.2 points. This is <strong>filler versus peer, not before versus after a reply</strong>.
A negative difference means less endorsement of the same own-position text when the peer message is present.
A positive difference means more endorsement. A drop of 0.2 is one fifth of one rating step, not 20% of members switching positions;
the 1–7 coding treats rating steps as equally spaced, not as a validated scale of psychological belief.</p>
{compact_d_tables(d_own_text, kind="text")}
<p>D2 can change while C's selected option stays the same: a full position also contains reasons and qualifications.
It does not establish which part of the text lost endorsement, agreement with the peer, or a later change of answer.
Unlike C and D1, it is measured before the member replies, so its sign or size need not match their post-reply measurements.</p>
<details><summary>D · Initial option probability → current</summary>
<p>Use the option selected before any debate as the fixed reference, rather than the option selected at the previous own measurement.
The calculation and question weighting are otherwise the same as D1. This view includes every sampled current measurement,
not only the final reply.</p>
{compact_d_tables(d_initial, kind="initial")}</details>
<details><summary>Probability extraction and shared readings</summary>
<p>For both D1 and D2, normalize the recorded candidate-letter probabilities at read temperature 1.
Letters absent from the endpoint's top 20 tokens are approximated as zero; they are not known to be impossible.
Displayed values are rounded to two decimals, so 100.00% is not proof of absolute certainty and ≈0.00 is not an exact zero.
These are probabilities of output tokens under a particular prompt, not a truth score or direct access to internal beliefs.
The direct C/D readings disable reasoning; normal debate keeps reasoning on.</p>
<p>The shared T2 event counts once. D2 also shares the pre-reply reading at T3, where the histories have not yet diverged;
that reading stays counted once under the original continuation's reply label, not again under the alternate reply label.
Both D1 continuations are retained because their post-reply contexts differ. This preserves the existing measurement assignment;
the shared D2 reading is not a comparison of tone effects. Turn-specific data remain in the <a href="#cases">full cases</a>.</p>
</details></section>
<section id="paired"><h2>Two continuations from the same prefix</h2>
<p>At T3 the histories and incoming message are identical; only the current private tone instruction changes.
At T4 both the incoming T3 message and tone instruction differ, so this is a subsequent-path comparison, not an isolated T4 tone effect.
There is one generation per assignment: individual differences also include generation variability.</p>
<p>This table counts self-label pairs, not position changes. Labels are observed outcomes, so subtracting C/D means from differently
populated label groups is not a matched treatment-effect estimate.</p>
{table(["T", "Original self-label", *["Alternate: " + label(v) for v in AGREEMENT], "Alternate: missing/unreported"], transitions)}
<details><summary>Matched tone assignments</summary>{table(["T", "Original tone", "Alternate tone", "Paths"], tone_pairs)}</details></section>
<section id="cases"><h2>All 60 paired paths · full text and measurements</h2>
<p>Open a case to see its question/options, two initial positions and shared T1/T2, followed by original versus alternate T3/T4.
Researcher-visible tone and self-label annotations are not metadata supplied in later debate prompts.</p>
{"".join(case_detail(r["branches"]["original"], r["branches"]["alternate"], pair, show_prefix_measurements=True) for r in records for pair in PAIRS)}
</section><footer>Protocol {esc(VERSION)}. Rubric {esc(manifest["agreement_rubric"]["version"])}.
Frozen design SHA-256: {esc(manifest["source"]["design_sha256"])}. Old raw runs remain unchanged.</footer></main></body></html>'''
    summary = {
        "schema": "strong_turn_tone_dyadic_report_v1",
        "protocol_version": VERSION,
        "kind": manifest["kind"],
        "planned_counts": manifest["planned_counts"],
        "report": report,
        "design": manifest["design"],
        "agreement_rubric": manifest["agreement_rubric"],
        "rows": rows,
        "D_text_counted_rows": dtext,
        "A_by_current_tone": a_rows,
        "A_by_assignment_T_tone": a_detail,
        "B_by_self_label": b_data,
        "C_by_label_pair": c_adjacent,
        "C_initial_to_endpoint_by_label_pair": c_endpoint,
        "D_choice_by_label_pair_model": d_choice,
        "D_text_by_label_pair_model": d_own_text,
        "D_initial_by_label_pair_model": d_initial,
        "label_pair_support": coverage,
        "paired_label_transitions": transitions,
        "tone_pairs": tone_pairs,
        "unique_formal_replies": formal,
        "shared_prefix_counted_once": True,
        "D_text_T3_counted_once_under_original_label": True,
        "pooled_with_archived_studies": False,
    }
    return html, summary


def publish(source, output, legacy, supplementary, *, quality_source=None):
    source, output, legacy, supplementary = [Path(p).resolve() for p in (source, output, legacy, supplementary)]
    if output.suffix != ".html" or source in output.parents or len({output, legacy, supplementary}) != 3:
        raise ValueError("Separate the run, new report and two archives")
    if legacy.parent != output.parent or supplementary.parent != output.parent or not supplementary.exists():
        raise ValueError("Archives must be beside the main report for portable links")
    manifest, report, records = load_run(source)
    if manifest["kind"] != "paid_strong_dyadic_pilot":
        raise ValueError("Never publish mock outputs as results")
    document, summary = render(
        manifest, report, records, legacy_href=legacy.name, supplementary_href=supplementary.name
    )
    if output.exists():
        old = json.loads(output.with_suffix(".json").read_text())
        if old.get("schema") == "strong_turn_tone_dyadic_report_v1":
            if old.get("source_directory") != str(source):
                raise ValueError("Refuse to overwrite another strong run")
            if old.get("length_controlled_E") and quality_source is None:
                raise ValueError("Retain --quality-source; do not silently remove the completed E follow-up")
            if quality_source is not None and not old.get("length_controlled_E"):
                quality_root = Path(quality_source).resolve()
                for prior in (output, output.with_suffix(".json")):
                    snapshot = quality_root / ("report-before-E" + prior.suffix)
                    if snapshot.exists() and snapshot.read_bytes() != prior.read_bytes():
                        raise ValueError("Existing pre-E report snapshot differs")
                    if not snapshot.exists():
                        shutil.copyfile(prior, snapshot)
        elif old.get("schema") == "turn_tone_dyadic_pilot_report_v1":
            for src, dst in ((output, legacy), (output.with_suffix(".json"), legacy.with_suffix(".json"))):
                if dst.exists():
                    if dst.read_bytes() != src.read_bytes():
                        raise ValueError("Existing legacy archive differs")
                else:
                    shutil.copyfile(src, dst)
        else:
            raise ValueError("Unknown report at target; preserve it")
    if not legacy.exists() or not legacy.with_suffix(".json").exists():
        raise ValueError("The fully-label reference must be preserved before publication")
    summary.update(
        source_directory=str(source),
        manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        legacy_archive_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (legacy, legacy.with_suffix(".json"))
        },
    )
    if quality_source is not None:
        from .quality_report import include_quality

        document, summary = include_quality(document, summary, quality_source)
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, required=True)
    parser.add_argument(
        "--quality-source", type=Path, help="Append the length-controlled E follow-up for this same run"
    )
    args = parser.parse_args()
    print(publish(args.source, args.output, args.legacy, args.supplementary, quality_source=args.quality_source))
