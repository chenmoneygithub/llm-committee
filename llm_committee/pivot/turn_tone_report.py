"""A fresh pilot HTML; fixed-tone experiments live in a separate supplementary archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from . import prompts
from .agreement import AGREEMENT, AGREEMENT_DEFINITIONS
from .dyadic import PAIRS, make_route
from .dyadic_report import MODELS, REQUIRED, c_table, d_table, flatten, label
from .forced_feedback_report import esc, number, rate, table
from .models import TONES
from .results_report import CSS
from .turn_tone import STREAM, VERSION

EXTRA_CSS = """
.scroll { width:100%; overflow-x:auto; margin:18px 0; }
th, td { text-align:left; }
.tone { font-size:12px; padding:2px 8px; border-radius:12px; background:#eaf1f5; }
.history p { white-space:pre-wrap; }
.history h4 { margin:20px 0 4px; }
@media print { .scroll { overflow:visible; } }
"""


def load_run(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] != VERSION:
        raise ValueError("Not the turn-level tone pilot")
    if report["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Do not label an unfinished pilot as completed")
    # A failed transport attempt can retain a conservative reservation after
    # its replacement succeeds. Never present that reservation as money spent.
    with sqlite3.connect(f"file:{source.resolve()}/requests.sqlite3?mode=ro", uri=True) as db:
        ledger = db.execute("SELECT status, COUNT(*), SUM(charge) FROM calls GROUP BY status").fetchall()
    report = {
        **report,
        "report_cost_accounting": {
            "received_response_estimate_usd": sum(
                cost for status, _, cost in ledger if status in ("completed", "received", "invalid")
            ),
            "unresolved_reservations_usd": sum(
                cost for status, _, cost in ledger if status not in ("completed", "received", "invalid")
            ),
            "unresolved_attempts": sum(
                n for status, n, _ in ledger if status not in ("completed", "received", "invalid")
            ),
        },
    }
    records = []
    route = make_route()
    for plan in manifest["plans"]:
        record = json.loads((source / "questions" / f"{plan['question_id']}.json").read_text())
        assert record["question_id"] == plan["question_id"]
        assert record["tone_schedule"] == plan["tone_schedule"]
        assert record["probe_tone"] == "neutral"
        assert [e["id"] for e in record["events"]] == [e["id"] for e in plan["events"]]
        history = record["formal_replies"][STREAM]
        for event in record["events"]:
            node = route.get(event["node_id"])
            peer, current = history.get(node.parent), history.get(node.id)
            assert event["previous_peer_self_label"] == (peer["agreement"] if peer else None)
            assert event["current_self_label"] == (current["agreement"] if current else None)
            assert event["previous_tone"] == plan["tone_schedule"][node.parent]
            assert event["current_tone"] == plan["tone_schedule"][node.id]
        records.append(record)
    return manifest, report, records


def toned_table(value):
    return value.replace(">previous_tone<", ">Peer's assigned tone<").replace(
        ">current_tone<", ">Receiver's assigned tone<"
    )


def question_details(record):
    q = record["question"]
    route = make_route()
    history = record["formal_replies"][STREAM]
    events = {e["node_id"]: e for e in record["events"]}
    initial = record["initial_answers"]
    options = "".join(f"<li>{esc(chr(65 + i))}: {esc(option)}</li>" for i, option in enumerate(q["options"]))
    branches = []
    for pair, members in PAIRS.items():
        sequence = " → ".join(record["tone_schedule"][f"{pair}-{t}"] for t in range(1, 5))
        initials = "".join(
            f"<h4>{esc(MODELS[m])}: independent initial position</h4><p>{esc(initial[str(m)])}</p>" for m in members
        )
        turns = []
        for t in range(1, 5):
            nid = f"{pair}-{t}"
            node = route.get(nid)
            value, event = history.get(nid), events.get(nid)
            body = f"<p>{esc(value['reply'] if value else 'Missing reply')}</p>"
            if event:
                c = event["C_text"]
                b = event["B"]
                body += f"<p><strong>Previous peer label:</strong> {esc(label(event['previous_peer_self_label']))}. "
                body += f"<strong>Current receiver label:</strong> {esc(label(event['current_self_label']))}.</p>"
                body += f"<p><strong>B — Gemini text rating:</strong> {esc(b if b else 'Missing')}</p>"
                for title, position in (
                    ("Previous own position", event["position_before"]),
                    ("Current own position", event["position_after"]),
                ):
                    body += f"<h4>{title}</h4><p>{esc(position['choice'] if position else 'Missing')} — "
                    body += f"{esc(position['position'] if position else 'Missing')}</p>"
                body += f"<p><strong>C — Gemini text comparison:</strong> {esc(c if c else 'Missing')}</p>"
                if event["D_text"]:
                    d = event["D_text"]
                    scores = {
                        arm: sum(i * d[arm]["probabilities"][k] for i, k in enumerate("ABCDEFG", 1))
                        for arm in ("control", "argument")
                    }
                    body += "<p><strong>D — endorsement of the same full prior position:</strong> "
                    body += f"filler {number(scores['control'])}, peer {number(scores['argument'])}; "
                    body += f"peer − filler {number(d['mean_own_agreement_argument_minus_control'], signed=True)} points.</p>"
            else:
                body += '<p class="muted">Not a sampled B/C/D event. Its position may still be read as a reference for a later event.</p>'
            turns.append(
                f"<details><summary>T={t} · {esc(MODELS[node.receiver])} · "
                f"assigned {esc(record['tone_schedule'][nid])} · self-label "
                f"{esc(label(value['agreement'] if value else None))}</summary>{body}</details>"
            )
        branches.append(f"<details><summary>{pair}: {esc(sequence)}</summary>{initials}{''.join(turns)}</details>")
    return f'<details class="history"><summary>{esc(q["id"])} — {esc(q["text"])}</summary><ul>{options}</ul>{"".join(branches)}</details>'


def render(manifest, report, records, *, supplementary_href):
    rows = flatten(records)
    formal = []
    for q in records:
        for nid, reply in q["formal_replies"][STREAM].items():
            formal.append(
                {
                    "question_id": q["question_id"],
                    "pair": nid.split("-")[0],
                    "current_tone": q["tone_schedule"][nid],
                    "self_label": reply["agreement"],
                }
            )
    a_rows = []
    for tone in TONES:
        subset = [r for r in formal if r["current_tone"] == tone]
        counts = Counter(r["self_label"] for r in subset)
        a_rows.append([tone, len(subset), *[rate(counts[v], len(subset)) for v in AGREEMENT], counts[None]])
    b_rows = []
    for current in (*AGREEMENT, None):
        group = [r for r in rows if r["current_self_label"] == current]
        if not group and current is None:
            continue
        counts = Counter((r["B"] or {}).get("label") for r in group)
        valid = len(group) - counts[None]
        b_rows.append(
            [
                label(current),
                len(group),
                *[rate(counts[v], valid) for v in AGREEMENT],
                counts["no_position"],
                counts["unjudgeable"],
                counts[None],
            ]
        )
    support_rows = []
    for previous in (*AGREEMENT, None):
        subset = [r for r in rows if r["previous_peer_self_label"] == previous]
        if not subset and previous is None:
            continue
        support_rows.append(
            [
                label(previous),
                *[sum(r["current_self_label"] == v for r in subset) for v in AGREEMENT],
                sum(r["current_self_label"] is None for r in subset),
            ]
        )
    tone_support = []
    for previous in TONES:
        for current in TONES:
            subset = [r for r in rows if r["previous_tone"] == previous and r["current_tone"] == current]
            tone_support.append([previous, current, *[sum(r["T"] == t for r in subset) for t in (2, 3, 4)]])
    c_sections = []
    for previous in (*AGREEMENT, None):
        subset = [r for r in rows if r["previous_peer_self_label"] == previous]
        if previous is None and not subset:
            continue
        c_sections.append(
            f"<h3>Preceding peer self-label: {esc(label(previous))}</h3>"
            + (c_table(subset) if subset else "<p>No sampled cases in this group.</p>")
        )
    mixed_count = sum(len(set(t["tone_sequence"])) > 1 for q in records for t in q["trajectories"])
    title = "Turn-level tone · 20-question two-member pilot"
    accounting = report["report_cost_accounting"]
    cost_text = f"Recorded responses: token-estimated US${accounting['received_response_estimate_usd']:.2f}."
    if accounting["unresolved_attempts"]:
        cost_text += (
            f" {accounting['unresolved_attempts']} provider attempt(s) have unknown billing; "
            f"US${accounting['unresolved_reservations_usd']:.2f} is retained as a conservative reservation, not confirmed spending."
        )
    cost_text += " These are local accounting estimates, not a provider invoice."
    mock = (
        '<p class="callout">OFFLINE MOCK — synthetic engineering fixture, not scientific results.</p>'
        if manifest["kind"] == "offline_mock"
        else ""
    )
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}{EXTRA_CSS}</style></head>
<body><main><header><p class="eyebrow">New protocol · turn-level tone · neutral measurement probes</p><h1>{title}</h1>
{mock}<p class="lede">Tone is assigned independently of the generated content at each debate turn, not once for the whole conversation.
This page contains only the new dyadic pilot. Earlier fixed-tone experiments are in the supplementary report.</p>
<p class="status">{report["completed_questions"]}/20 questions complete. Trajectory outcomes: {esc(report["trajectory_statuses"])}.
{len(formal)}/240 formal replies; {len(rows)}/160 sampled B/C/D events. {cost_text}</p>
<p><a href="{esc(supplementary_href)}">Open supplementary fixed-tone experiments</a> — archived separately, not pooled here.</p></header>
<nav><a href="#design">Design</a><a href="#a">A · Self-labels</a><a href="#b">B · Text check</a>
<a href="#support">Sample coverage</a><a href="#c">C · Position</a><a href="#d">D · Probabilities</a><a href="#cases">Read all questions and trajectories</a></nav>
<section id="design"><h2>What was run?</h2>
<p>A = GPT-5.6 Terra, B = Qwen3.8-27B, C = Inkling. Each of 20 presampled questions has three independent directed
pairs: AB, CA, BC. AB starts with A's archived independent position and receives replies from B, A, B, A. Each pair
has four new replies. There is no three-member run, no extra global-tone arm and no final-answer synthesis in this pilot.</p>
<p>The 20 questions were drawn by seed {manifest["question_selection"]["seed"]} from the approved 60-question bank before new generation,
not selected for their prior effects. Assigned tones are balanced within pair × T across questions: 80 friendly,
80 neutral, 80 hostile turns overall. {mixed_count}/60 scheduled trajectories contain more than one tone;
random scheduling permits an occasional constant-tone sequence without creating a separate global-tone arm.</p>
<p>Only the current turn's private tone instruction is supplied. Historical text and speaker/reply relationships remain,
but historical tone instructions, explicit tone tags and self-reported labels are not passed to the member.
The public text can still express a tone. All C/D probes use the same neutral instruction; their outputs never enter debate.</p>
<p>Each question has nine eligible pair × T positions at T=2/3/4. Eight are presampled, and B/C/D share them.
T=1 remains in A but has no preceding reply label. These are 20 independent questions, not 160 independent questions.
This pilot is for inspecting the design and outputs; small label/turn/model strata are not conclusive effect estimates.</p>
<p class="callout">The archived fixed-tone experiments used tone-bearing probes. This pilot changes both tone assignment
and probe instructions. Do not attribute an old/new difference solely to turn-level tone, and do not pool the cohorts.</p></section>
<section id="a"><h2>A · What does each reply say about its incoming peer message?</h2>
<p>Rows use the receiver's current assigned tone. Columns are its four self-reported agreement labels, not its option
on the original survey question. “Friendly” and “hostile” describe the assigned debate policy, not a separately verified style rating.</p>
{table(["Current assigned tone", "Replies", *[label(v) for v in AGREEMENT], "Not reported"], a_rows)}
<details><summary>Definitions of the four labels (shared by debaters and Gemini)</summary>
{table(["Label", "Definition"], [[label(v), AGREEMENT_DEFINITIONS[v]] for v in AGREEMENT])}</details></section>
<section id="b"><h2>B · Does the reply text match its self-label?</h2>
<p>One judge, Gemini 3.8 Flash, reads the incoming peer contribution and the current reply without either private tone
instruction or the self-report. Rows are the receiver's self-label; columns are Gemini's text rating. Percentages are
within each row among completed judgments, not pooled across labels. No human validation has been completed for this pilot.</p>
{table(["Current self-label", "Sampled replies", *[f"Gemini: {label(v)}" for v in AGREEMENT], "No position", "Unjudgeable", "Missing"], b_rows)}</section>
<section id="support"><h2>Which feedback/response combinations were sampled?</h2>
<p>The preceding peer self-label concerns its response to this receiver's latest public contribution; the current self-label
concerns the receiver's reply to that peer. These are observed response labels, not randomized treatments.
A zero is no sampled cases, not a zero effect. C/D show observed groups only.</p>
{table(["Preceding peer self-label", *[f"Current: {label(v)}" for v in AGREEMENT], "Current: not reported"], support_rows)}
<details><summary>Coverage by preceding and current assigned tones, at each T</summary>
{table(["Peer assigned tone", "Receiver assigned tone", "T=2", "T=3", "T=4"], tone_support)}</details></section>
<section id="c"><h2>C · How does the receiver's own position change?</h2>
<p>Compare the same member's previous own participation with its current one on this pair's trajectory, using the independent
initial position if it has not replied before. Option change is a deterministic comparison of selected original options.
Gemini separately labels full-text change: unchanged, adjusted reasons/qualifications, or changed main conclusion.
These two change counts are not added. Primary tables retain both self-labels and T; they describe the randomized-tone mixture.</p>
{"".join(c_sections)}
<details><summary>C · Retain both assigned tones as well as both self-labels and T</summary>
<p>Fine-grained descriptive cells for inspection, often with only one or two cases; not an adequately powered factorial comparison.</p>
{toned_table(c_table(rows, (*REQUIRED, "previous_tone", "current_tone", "T")))}</details>
<details><summary>C · Initial → sampled last own position (different reference)</summary>
<p>Sampled T=3/4 nodes are each member's last own participation. The endpoint labels do not explain when earlier cumulative changes occurred.</p>
{c_table(rows, endpoint=True)}</details></section>
<section id="d"><h2>D · Additional quantitative readings for Qwen and Inkling</h2>
<p>Debate uses reasoning; direct C/D readings disable it. Probabilities are normalized over original candidate letters at read
temperature 1. Letters absent from the returned top-20 are approximated as zero, not known to be truly impossible.</p>
<h3>D1 · Probability of the same previously selected survey option</h3>
<p>Compare the probability of the prior option before and after the receiver's reply. For example, 80% to 70% is −10
percentage points, even if the selected option remains unchanged. This is not automatically movement toward the peer.</p>
{d_table(rows, kind="choice")}
<h3>D2 · Endorsement of the complete prior position: peer versus filler</h3>
<p>Before the new formal reply, freeze the full prior position text. Read its endorsement in two parallel inputs:
one adds the peer message; the other adds approximately length-matched repetitions of
“{esc(prompts.FILLER_SENTENCE)}”. Both probes are neutral. The expected score is on a 1–7 scale,
from completely disagree to completely agree. A −0.1 difference means one tenth of a scale point less endorsement,
not 10% of members changing their minds. Reference scores and differences use the same valid pairs, averaged within
question first and then equally across represented questions in each displayed group.</p>
{d_table(rows, kind="text")}
<details><summary>D · Both assigned tones, both self-labels, receiver model and T</summary>
<h3>Prior option probability</h3>{toned_table(d_table(rows, (*REQUIRED, "previous_tone", "current_tone", "member", "T"), kind="choice"))}
<h3>Full prior-position endorsement</h3>{toned_table(d_table(rows, (*REQUIRED, "previous_tone", "current_tone", "member", "T"), kind="text"))}</details>
<details><summary>D · Probability of the initial option, initial → current</summary>
{d_table(rows, kind="initial")}</details></section>
<section id="cases"><h2>All 20 questions and 60 trajectories</h2>
<p>The tone annotations below are shown to the researcher only, not inserted into the model's historical context.
Open a question, pair, and turn to inspect the original options, discussion and sampled measurements.</p>
{"".join(question_details(q) for q in records)}</section>
<footer>Protocol {esc(VERSION)}. Source-run manifest SHA-256: {esc(manifest["source"]["manifest_sha256"])}.
No global C/D effect headline, no human-validated accuracy claim, no unperformed significance test.</footer></main></body></html>'''
    summary = {
        "schema": "turn_tone_dyadic_pilot_report_v1",
        "protocol_version": VERSION,
        "kind": manifest["kind"],
        "planned_counts": manifest["planned_counts"],
        "report": report,
        "question_selection": manifest["question_selection"],
        "sampling": manifest["sampling"],
        "design": manifest["design"],
        "rows": rows,
        "A_by_current_tone": a_rows,
        "B_by_self_label": b_rows,
        "label_pair_support": support_rows,
        "tone_pair_support_by_T": tone_support,
        "pooled_with_archived_studies": False,
    }
    return html, summary


def archive_fixed_tone(source, output, *, replace=False):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Keep the old canonical report intact")
    if source.suffix != ".html" or output.suffix != ".html":
        raise ValueError("Archive source and output must be HTML files")
    if output.exists() and not replace:
        raise ValueError("Supplementary archive already exists; use --replace to regenerate")
    if output.exists() or output.with_suffix(".json").exists():
        old = json.loads(output.with_suffix(".json").read_text())
        if old.get("report_role") != "supplementary_fixed_tone_archive" or old.get("archived_from_html") != str(source):
            raise ValueError("Do not replace an unrelated supplementary report")
    text = source.read_text()
    summary = json.loads(source.with_suffix(".json").read_text())
    title = "Supplementary results · archived fixed-tone experiments"
    text, titles = re.subn(r"<title>.*?</title>", f"<title>{title}</title>", text, count=1, flags=re.S)
    text, headings = re.subn(r"<h1>.*?</h1>", f"<h1>{title}</h1>", text, count=1, flags=re.S)
    if titles != 1 or headings != 1:
        raise ValueError("Unexpected source report wrapper")
    note = (
        '<p class="callout"><strong>Supplementary archive — fixed-tone protocols.</strong> '
        "These are the earlier three-member study, forced-feedback supplement and two-member fixed-tone study. "
        "They are retained for reference, not pooled into the new turn-level-tone pilot. "
        "Their C/D probes carry the assigned tone, unlike the new neutral probes. "
        "The tables below reproduce the archived results; original data and the old report are unchanged.</p>"
    )
    text = text.replace("</h1>", "</h1>" + note, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                **summary,
                "report_role": "supplementary_fixed_tone_archive",
                "archived_from_html": str(source),
                "archived_from_html_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supplementary-source", type=Path, required=True)
    parser.add_argument("--supplementary-output", type=Path, required=True)
    parser.add_argument("--fork-source", type=Path, help="Add the matched T3/T4 continuation to the same pilot report")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.suffix != ".html" or source in output.parents:
        parser.error("Use a separate HTML outside the raw run")
    if output in (args.supplementary_source.resolve(), args.supplementary_output.resolve()):
        parser.error("New pilot and supplementary HTML must have distinct paths")
    if output.exists() and not args.replace:
        parser.error("New pilot output exists; use --replace to regenerate the same run")
    if output.exists():
        old = json.loads(output.with_suffix(".json").read_text())
        if old.get("source_directory") != str(source):
            parser.error("Do not replace a report for another source")
        if old.get("paired_continuation") and args.fork_source is None:
            parser.error("Do not silently remove the paired continuation; retain --fork-source")
    manifest, report, records = load_run(source)
    if manifest["kind"] != "paid_turn_tone_dyadic_pilot":
        parser.error("Production report must not contain offline mock outputs")
    if args.supplementary_output.resolve().parent != output.parent:
        parser.error("Place pilot and supplementary HTML beside each other for portable local links")
    document, summary = render(manifest, report, records, supplementary_href=args.supplementary_output.name)
    summary.update(
        source_directory=str(source),
        manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
    )
    if args.fork_source:
        from .turn_tone_fork_report import include_fork

        document, summary = include_fork(document, summary, args.fork_source)
    archive_fixed_tone(args.supplementary_source, args.supplementary_output, replace=args.replace)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(output)
    print(args.supplementary_output.resolve())


if __name__ == "__main__":
    main()
