"""Offline report for integrated-position debate; never mixes archived protocol outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot.dyadic import PAIRS, make_route
from llm_committee.pivot.dyadic_report import MODELS, REQUIRED, conditional_groups, flatten, label
from llm_committee.pivot.forced_feedback_report import esc, question_mean, table
from llm_committee.pivot.models import TONES, PilotConfig
from llm_committee.pivot.quality_report import include_quality
from llm_committee.pivot.results_report import CSS
from llm_committee.pivot.stateful_study import PROTOCOL_PROMPTS, PUBLIC_HISTORY_VERSION
from llm_committee.pivot.strong_agreement import AGREEMENT, AGREEMENT_DEFINITIONS
from llm_committee.pivot.strong_report import c_summary, compact_c_table, compact_d_tables, d_number, d_summary
from llm_committee.pivot.study import atomic_json

ARMS = ("original", "alternate")


def rate(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "—"


def load_run(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] not in PROTOCOL_PROMPTS or report["status"] not in (
        "completed",
        "completed_with_failures",
    ):
        raise ValueError("Need a completed integrated-position run")
    records = [json.loads((source / "questions" / f"{p['question_id']}.json").read_text()) for p in manifest["plans"]]
    route = make_route()
    for record, plan in zip(records, manifest["plans"], strict=True):
        assert (
            record["question_id"] == plan["question_id"] and record["protocol_version"] == manifest["protocol_version"]
        )
        for arm in ARMS:
            branch = record["branches"][arm]
            assert branch["tone_schedule"] == plan["arms"][arm]["tone_schedule"]
            assert [e["id"] for e in branch["events"]] == [e["id"] for e in plan["arms"][arm]["events"]]
            history = branch["formal_replies"]["turn_level"]
            for event in branch["events"]:
                node = route.get(event["node_id"])
                assert event["previous_peer_self_label"] == (history.get(node.parent) or {}).get("agreement")
                assert event["current_self_label"] == (history.get(node.id) or {}).get("agreement")
                if event["position_after"]:
                    assert event["position_after"]["position"] == history[node.id]["position"]
                    assert event["position_after"]["choice"] == history[node.id]["choice"]
                if event["D_text"]:
                    assert event["D_text"]["fixed_full_position"] == event["position_before"]["position"]
        for node in route.nodes:
            if node.depth <= 2:
                assert record["branches"]["original"]["formal_replies"]["turn_level"].get(node.id) == record[
                    "branches"
                ]["alternate"]["formal_replies"]["turn_level"].get(node.id)
    return manifest, report, records


def collect(records):
    formal, rows = [], []
    route = make_route()
    for record in records:
        for arm in ARMS:
            branch = record["branches"][arm]
            for nid, reply in branch["formal_replies"]["turn_level"].items():
                node = route.get(nid)
                if arm == "alternate" and node.depth <= 2:
                    continue
                formal.append(
                    {
                        "question_id": record["question_id"],
                        "node_id": nid,
                        "T": node.depth,
                        "assignment": arm,
                        "tone": branch["tone_schedule"][nid],
                        "member": node.receiver,
                        "self_label": reply["agreement"],
                    }
                )
            for row in flatten([branch]):
                if arm == "alternate" and row["T"] == 2:
                    continue
                rows.append({**row, "assignment": arm})
    return formal, rows


def d2_groups(rows):
    result = []
    for key, group in conditional_groups(
        [r for r in rows if r["member"] in (1, 2)], (*REQUIRED, "member"), agreement_order=AGREEMENT
    ):
        complete = [r for r in group if r["D_text"]]
        flat = [{**r, **r["D_text"]} for r in complete]
        metrics = {
            name: question_mean(flat, name)[0]
            for name in (
                "modal_rating_change",
                "reference_probability_change_pp",
                "mean_own_agreement_argument_minus_control",
                "mean_rating_control",
                "mean_rating_argument",
            )
        }
        result.append(
            {
                **dict(zip((*REQUIRED, "member"), key, strict=True)),
                **metrics,
                "pairs": len(complete),
                "questions": len({r["question_id"] for r in complete}),
                "modal_changed": sum(r["D_text"]["modal_rating_change"] not in (None, 0) for r in complete),
                "modal_compared": sum(r["D_text"]["modal_rating_change"] is not None for r in complete),
                "reference_compared": sum(r["D_text"]["reference_probability_change_pp"] is not None for r in complete),
            }
        )
    return result


def d2_tables(groups):
    parts = []
    for member in (1, 2):
        subset = [g for g in groups if g["member"] == member]

        def prefix(g):
            return [label(g[k]) for k in REQUIRED] + [f"{g['pairs']} / {g['questions']}"]

        parts.append(f"<h3>{MODELS[member]} · D2</h3>")
        parts.append(
            table(
                [
                    "Peer label",
                    "Receiver label",
                    "Pairs / questions",
                    "Top category changed",
                    "Category difference (steps)",
                ],
                [
                    prefix(g)
                    + [rate(g["modal_changed"], g["modal_compared"]), d_number(g["modal_rating_change"], signed=True)]
                    for g in subset
                ],
            )
        )
        parts.append(
            table(
                [
                    "Peer label",
                    "Receiver label",
                    "Pairs / questions",
                    "Filler-top category Δ probability (pp)",
                    "Mean rating: filler → argument",
                    "Mean rating difference",
                ],
                [
                    prefix(g)
                    + [
                        d_number(g["reference_probability_change_pp"], signed=True)
                        + (f" ({g['reference_compared']} pairs)" if g["reference_compared"] != g["pairs"] else ""),
                        f"{d_number(g['mean_rating_control'])} → {d_number(g['mean_rating_argument'])}",
                        d_number(g["mean_own_agreement_argument_minus_control"], signed=True),
                    ]
                    for g in subset
                ],
            )
        )
    return "".join(parts)


def cases(records, model_names=MODELS):
    parts = []
    route = make_route()
    for record in records:
        q = record["question"]
        options = " · ".join(f"{chr(65 + i)}: {text}" for i, text in enumerate(q["options"]))
        for pair, members in PAIRS.items():
            parts.append(
                f'<details class="case"><summary>{esc(q["id"])} · {pair} · {esc(q["text"])}</summary><p>{esc(options)}</p>'
            )
            for m in members:
                value = record["initial_positions"].get(str(m))
                if value:
                    parts.append(
                        f"<h4>{esc(model_names[m])} · initial choice {esc(value['choice'])}</h4><p>{esc(value['position'])}</p>"
                    )
            for arm in ARMS:
                branch = record["branches"][arm]
                parts.append(f"<h3>{arm.title()} tone assignment</h3>")
                for node in route.nodes:
                    if not node.id.startswith(pair + "-"):
                        continue
                    value = branch["formal_replies"]["turn_level"].get(node.id)
                    if not value:
                        parts.append(f"<p>{node.id}: missing reply</p>")
                        continue
                    shared = " · shared prefix" if node.depth <= 2 else ""
                    parts.append(
                        f"<h4>T{node.depth} · {esc(model_names[node.receiver])} · {branch['tone_schedule'][node.id]}{shared}</h4>"
                        f"<p><b>Self-label:</b> {esc(label(value['agreement']))} · <b>Choice:</b> {esc(value['choice'])}</p>"
                        f"<p><b>Reply to previous member:</b> {esc(value['reply'])}</p>"
                        f"<p><b>Updated own position:</b> {esc(value['position'])}</p>"
                    )
                    event = next((e for e in branch["events"] if e["node_id"] == node.id), None)
                    if event:
                        ratings = {k: event[k] for k in ("B", "C_text", "C_initial_to_endpoint")}
                        parts.append(
                            f"<details><summary>Judge labels and supporting evidence</summary><pre>{esc(json.dumps(ratings, ensure_ascii=False, indent=2))}</pre></details>"
                        )
            parts.append("</details>")
    return "".join(parts)


def probe_examples(source, rows):
    """First five unique sampled D2 events per model: deterministic, not effect-selected."""
    selected = []
    for member in (1, 2):
        subset = [r for r in rows if r["member"] == member and r["D_text"]]
        selected.extend(sorted(subset, key=lambda r: (r["question_id"], r["node_id"], r["assignment"]))[:5])
    with sqlite3.connect(f"file:{Path(source).resolve()}/requests.sqlite3?mode=ro", uri=True) as db:
        entries = {}
        for rq, rs, parsed in db.execute(
            "SELECT request,response,parsed FROM calls WHERE status='completed' AND json_extract(request,'$.purpose') IN ('d_text','d_choice')"
        ):
            request = json.loads(rq)
            entries[request["key"]] = {"request": request, "response": json.loads(rs), "parsed": json.loads(parsed)}
    parts = []
    for row in selected:
        prefix = row["position_after"]["source_request"].split("/turn_level/debate/")[0]
        parts.append(
            f'<details class="probe-case"><summary>{MODELS[row["member"]]} · {esc(row["question_id"])} · {row["node_id"]} · {row["assignment"]}</summary>'
        )
        previous_key = row["position_before"]["source_request"].replace("/turn_level/debate/", "/Dchoice/turn_level/")
        if row["previous_reading"].startswith("initial/"):
            previous_key = previous_key.replace("/initial/", "/Dchoice/initial/")
        reads = [
            ("D1 previous input", previous_key, row["D_choice_before"]),
            ("D1 current input", row["D_choice_request"], row["D_choice_after"]),
            *[(f"D2 {t}", f"{prefix}/Dtext/{row['id']}/{t}", row["D_text"][t]) for t in ("argument", "control")],
        ]
        for name, key, dist in reads:
            if key not in entries or not dist:
                parts.append(f"<p>{esc(name)}: unavailable.</p>")
                continue
            entry = entries[key]
            request, response = entry["request"], entry["response"]
            readout = entry["parsed"]["_readout"]
            parts.append(
                f"<h4>{esc(name)} · output: {esc(response['text'])} · reasoning tokens: {response['reasoning_tokens']} · read token index: {readout['sampled_output_index']}</h4>"
            )
            parts.append(
                table(
                    ["Rating token", "Recorded log probability", "Normalized probability"],
                    [
                        [k, readout["candidate_logprobs"].get(k, "Not in top 20; zero-filled"), f"{100 * p:.6f}%"]
                        for k, p in dist["probabilities"].items()
                    ],
                )
            )
            for i, message in enumerate(request["messages"], 1):
                parts.append(f"<h5>Message {i} · {esc(message['role'])}</h5><pre>{esc(message['text'])}</pre>")
        parts.append("</details>")
    return "".join(parts)


def member_endpoints(records, rows):
    """End results are not attributed to the labels on the final adjacent exchange."""
    route = make_route()
    endpoints = []
    for record in records:
        for arm, branch in record["branches"].items():
            history = branch["formal_replies"]["turn_level"]
            for leaf in route.leaves:
                path = route.path(leaf.id)
                for member in PAIRS[leaf.id.split("-")[0]]:
                    final = next(n for n in reversed(path) if n.receiver == member)
                    initial, after = record["initial_positions"].get(str(member)), history.get(final.id)
                    changed = (
                        initial["choice"] != after["choice"]
                        if initial and after and initial["choice"] is not None and after["choice"] is not None
                        else None
                    )
                    endpoints.append(
                        {
                            "question_id": record["question_id"],
                            "assignment": arm,
                            "node_id": final.id,
                            "member": member,
                            "option_changed": changed,
                        }
                    )
    groups = []
    for member in range(3):
        group = [r for r in endpoints if r["member"] == member]
        choices = [r["option_changed"] for r in group if r["option_changed"] is not None]
        texts = [r for r in rows if r["member"] == member and r["final_pair"]]
        counts = Counter(r["endpoint_text_label"] for r in texts)
        text_n = sum(counts[k] for k in ("unchanged", "adjusted", "conclusion_changed"))
        groups.append(
            {
                "member": member,
                "paths": len(group),
                "option_compared": len(choices),
                "option_changed": sum(choices),
                "sampled_text_pairs": len(texts),
                "text_compared": text_n,
                "reasons_adjusted": counts["adjusted"],
                "conclusion_changed": counts["conclusion_changed"],
            }
        )
    return groups


def render(manifest, report, records, *, source):
    public_history = manifest["protocol_version"] == PUBLIC_HISTORY_VERSION
    config = PilotConfig(**manifest["config"])
    d_enabled = manifest["design"]["D_enabled"]
    model_names = MODELS if config.roster == "mixed_family" else dict(enumerate(config.members))
    if config.roster == "same_model":
        model_names = {m: f"{chr(65 + m)} (member {m}) · {model}" for m, model in enumerate(config.members)}
    formal, rows = collect(records)
    # Neutral T3 input is shared; assign it once to the original continuation's label.
    drows = [r for r in rows if not (r["assignment"] == "alternate" and r["T"] == 3)] if d_enabled else []
    c, endpoint = c_summary(rows), c_summary(rows, endpoint=True)
    d1, d1_initial, d2 = d_summary(drows, kind="choice"), d_summary(drows, kind="initial"), d2_groups(drows)
    a = []
    for tone in TONES:
        group = [r for r in formal if r["tone"] == tone]
        counts = Counter(r["self_label"] for r in group)
        a.append([tone, len(group), *[rate(counts[k], len(group)) for k in AGREEMENT]])
    b, b_excluded = [], 0
    for current in AGREEMENT:
        group = [r for r in rows if r["current_self_label"] == current]
        counts = Counter((r["B"] or {}).get("label") for r in group)
        n = sum(counts[k] for k in AGREEMENT)
        b_excluded += len(group) - n
        b.append([label(current), len(group), *[rate(counts[k], n) for k in AGREEMENT]])
    summary = {
        "protocol_version": manifest["protocol_version"],
        "roster": config.roster,
        "models": list(config.members),
        "D_enabled": d_enabled,
        "source_directory": str(Path(source).resolve()),
        "manifest_sha256": hashlib.sha256((Path(source) / "manifest.json").read_bytes()).hexdigest(),
        "questions": len(records),
        "formal_replies": len(formal),
        "sampled_events": len(rows),
        "A": a,
        "B": b,
        "C_adjacent": c,
        "C_endpoint": endpoint,
        "D1_adjacent": d1,
        "D1_initial": d1_initial,
        "D2": d2,
        "rows": rows,
        "D_shared_T3_grouping": "Count once under original continuation labels; descriptive, not an effect of its later tone",
        "run_report": report,
    }
    state_description = (
        "The next turn receives the original answers and public reply history only. Updated choice/position fields are saved for analysis, not reinjected into debate."
        if public_history
        else "The next turn receives public reply history and its own last recorded position on this branch."
    )
    if public_history and d_enabled:
        state_description += " D1 likewise receives no updated private fields. D2 alone receives the fixed prior full text as its measurement target, without the old option field."
    distributions, end_result = "", ""
    if public_history and d_enabled:
        from scripts.stateful_probability_diagnostics import distribution_plot, grouped

        summary["D1_distributions"] = grouped(drows)
        distributions = (
            "<h3>D1 · Individual changes, not only signed averages</h3><p>Each dot is one sampled comparison; the diamond is the question-equal mean. Orange starts at the initial question-only read; blue starts at an earlier participation. The ±1pp display band is not a significance or equivalence threshold. Hover for exact probabilities and event IDs.</p>"
            + "".join(
                f'<h4>{MODELS[m]}</h4><div class="scroll">{distribution_plot(summary["D1_distributions"], m)}</div>'
                for m in (1, 2)
            )
        )
    if public_history:
        summary["E_member_endpoints"] = member_endpoints(records, rows)
        end_result = (
            '<section id="member-endpoints"><h2>E1 · Members’ initial → last-participation positions</h2>'
            f"<p>End-result analysis, not a local C/D comparison. Keep each member and branch distinct; do not group by the final reply’s tone or agreement labels. Last participation is T3 or T4, and does not imply the member read every later message. These are repeated endpoints across {len(records)} questions, not independent new questions.</p>"
            "<p>Option changes use all available branch/member endpoints. Text changes use only the presampled endpoint pairs already judged by Gemini; each cell reports its own denominator. No extra judgments were requested.</p>"
            + table(
                ["Member", "Branch endpoints", "Option changed", "Reasons adjusted", "Conclusion changed"],
                [
                    [
                        model_names[g["member"]],
                        g["paths"],
                        rate(g["option_changed"], g["option_compared"]),
                        rate(g["reasons_adjusted"], g["text_compared"]),
                        rate(g["conclusion_changed"], g["text_compared"]),
                    ]
                    for g in summary["E_member_endpoints"]
                ],
            )
            + "</section>"
        )
    mock = (
        '<p class="warning">OFFLINE MOCK — synthetic fixtures, not scientific results.</p>'
        if manifest["kind"] == "offline_mock"
        else ""
    )
    html = [
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Integrated-position debate · September 27</title><style>{CSS}\n.scroll{{overflow-x:auto}} pre{{white-space:pre-wrap;overflow-wrap:anywhere}} details{{margin:14px 0;padding:12px;border:1px solid #ccc}} summary{{cursor:pointer}} td,th{{vertical-align:top}}</style></head><body><main>',
        "<h1>Turn-level tone · "
        + ("public-history-only debate" if public_history else "positions updated inside debate")
        + "</h1>",
        mock,
        '<nav><a href="#a">A · Self-labels</a> · <a href="#b">B · Text check</a> · <a href="#c">C · Position changes</a> · '
        + ('<a href="#d">D · Probability readings</a> · ' if d_enabled else "")
        + '<a href="#paired">Full debates</a>'
        + (' · <a href="#probes">D2 prompts</a>' if d_enabled else "")
        + "</nav>",
        f"<p>{len(records)} questions · {esc(' / '.join(model_names.values()))} · AB, CA, BC · four replies per path · paired turn-tone assignments sharing T1/T2. {len(formal)} unique replies; {len(rows)} sampled B/C events. No global-tone arms.</p>",
        f"<p>Each reasoning-enabled turn produces <b>reply, agreement, choice, position</b> together. {state_description} C uses these formal positions; it does not ask the member to generate another position. Only the reply is broadcast to peers.</p>",
        (
            '<p>Reference only: <a href="turn-tone-dyadic-stateful-2026-09-27.html">earlier explicit-position-feedback pilot</a>. All formal generations and measurements in this cohort are fresh. Earlier results are not pooled here.</p>'
            if public_history and d_enabled
            else ""
        ),
        (
            '<p>Reference only: <a href="turn-tone-dyadic-independent-position-reference-2026-09-27.html">previous independent-position report</a>. Its outputs are not pooled with this rerun.</p>'
            if d_enabled
            else "<p>Layer D is outside this roster’s scope; no probability probes were requested.</p>"
        ),
        "<details><summary>Four self-label definitions</summary>",
        table(["Label", "Definition"], [[label(k), AGREEMENT_DEFINITIONS[k]] for k in AGREEMENT]),
        "</details>",
        '<section id="a"><h2>A · What label does the receiver give its reply?</h2><p>The label describes agreement with the immediate peer message, not the original survey question. Percentages include all formal turns, including T1; shared replies count once.</p>',
        table(["Current turn tone", "Replies", *[label(k) for k in AGREEMENT]], a),
        "</section>",
        '<section id="b"><h2>B · Does the reply text support its self-label?</h2><p>One external judge, Gemini 3.8 Flash, sees the question, incoming peer message and reply, without the self-label or tone instruction. Read each row as: among replies self-labeled X, what label does Gemini assign from the text?</p>',
        table(["Receiver self-label", "Sampled replies", *[f"Gemini: {label(k)}" for k in AGREEMENT]], b),
        f"<p>{b_excluded} judgments outside the four labels or unavailable are excluded from the cell denominators. This compares self-reports with one model judge; it is not human-validated accuracy.</p></section>",
        '<section id="c"><h2>C · Did the member change its own position?</h2><p>Rows keep both labels: the incoming peer’s preceding self-label, then the receiver’s current self-label. In these two-member paths, both refer to the other member’s immediately preceding contribution. They are observed outcomes, not randomized agreement conditions.</p>',
        "<p><b>Option changed</b> is a direct comparison of the formal option letters. Gemini separately compares the two full position texts: <b>reasons adjusted</b> means changed reasoning or qualifications without a changed main conclusion; <b>conclusion changed</b> means the substantive answer changed. Text categories are disjoint; option and text counts must not be added.</p>",
        "<h3>Since this member’s previous participation</h3><p>Compare the receiver’s own previous choice and full position with its current choice and full position, on the same branch. This does not compare its answer with the peer’s answer. At its first participation, the reference is its initial position. Sampled measurements cover T2–T4; T1 remains visible in full debates.</p>",
        compact_c_table(c),
        (
            ""
            if public_history
            else "<h3>Initial position → last participation on the path</h3>" + compact_c_table(endpoint)
        ),
        "</section>",
        *(
            [
                '<section id="d"><h2>D · Supplementary probability readings</h2><p>Only Qwen and Inkling. Formal debate keeps reasoning on; these separate neutral probes turn reasoning off. Probabilities come from the first visible option/rating token, using the returned top 20 tokens: missing candidate letters are assigned zero, then the candidates are normalized. These are context-sensitive output probabilities, not internal beliefs.</p>',
                "<p>Every table keeps the peer/receiver labels and separates models. Mean differences first average events within a question, then weight questions equally. Shared T3 neutral inputs are counted once, under the original continuation’s observed label, not as independent readings for both later replies. Those labels do not imply that the subsequent tone caused this pre-generation reading.</p>",
                "<h3>D1 · Probability of the previously chosen survey option</h3><p>A no-reasoning replay sees the input available before the corresponding formal generation, but not its newly emitted choice, reply or position. Compare the previous participation’s replay with the current replay. A −10 pp difference means the same old option receives, for example, 70% → 60%; it need not change the most probable option or the formal choice.</p>",
                '<p class="callout"><b>A positive D1 difference is increased probability support for the fixed old survey option; a negative difference is decreased support.</b> This need not match a change in the reasoning-enabled formal choice or its wording. It does not isolate the peer argument’s causal effect: the histories differ, and the initial replay contains no own answer. '
                + (
                    "No updated private choice/position is separately injected into these new D1 inputs."
                    if public_history
                    else "This protocol also changes the explicitly supplied own-position state."
                )
                + " Signed means can hide individual increases and drops.</p>",
                compact_d_tables(d1, kind="choice"),
                (
                    ""
                    if public_history
                    else "<details><summary>D1 · Initial replay → current replay</summary>"
                    + compact_d_tables(d1_initial, kind="initial")
                    + "</details>"
                ),
                distributions,
                "<h3>D2 · Endorsement of the fixed pre-turn position: argument vs filler</h3><p>Both inputs contain the same branch history and the receiver’s same complete pre-turn position. One includes the peer’s incoming argument; the other replaces only that message with token-length-matched procedural filler. The model rates its own fixed text from A=1 (completely disagree) to G=7 (completely agree).</p>",
                "<p><b>Top category:</b> compare the most probable A–G category (not the sampled letter). Negative category difference means lower endorsement, in 1–7 steps. <b>Filler-top probability:</b> fix whichever category was most probable under filler; report its probability under argument minus filler, in pp. A decline alone is not a direction of stance. <b>Mean rating:</b> sum rating × probability over all seven options; argument minus filler gives rating points. For scale, −0.15 is 15% of one rating step, not 15% of replies changing. A small signed mean can hide opposite individual changes, so category-change counts and full distributions are also shown.</p>",
                "<p>No entropy adjustment. Tied modes are not arbitrarily broken: ambiguous category differences/reference probabilities are omitted from their respective metrics. None of these numbers alone establishes movement toward the peer or lasting belief change.</p>",
                d2_tables(d2),
                "</section>",
            ]
            if d_enabled
            else []
        ),
        end_result,
        '<section id="paired"><h2>Qualitative review · full paired debates</h2><p>Shared T1/T2 are displayed in each continuation for readability but counted once in the results. Each turn shows its local tone, public reply, formal option and full updated own position.</p>',
        cases(records, model_names),
        "</section>",
        *(
            [
                '<section id="probes"><h2>D1/D2 · Full prompts and token probabilities</h2><p>The first five unique sampled events per model, sorted by question/node/assignment, without selecting for effect size. Each block includes D1 previous/current replay inputs and D2 argument/filler inputs, every message, returned letter, probability-read index and all candidate probabilities.</p>',
                probe_examples(source, drows),
                "</section>",
            ]
            if d_enabled
            else []
        ),
        "<details><summary>Execution, failures and measured cache usage</summary><pre>",
        esc(json.dumps(report, ensure_ascii=False, indent=2)),
        "</pre></details>",
        "</main></body></html>",
    ]
    document = "".join(html)
    if public_history:
        document = document.replace("<title>Integrated-position debate", "<title>Public-history-only debate")
        document = document.replace(
            '<a href="#paired">',
            '<a href="#member-endpoints">E1 · Member outcomes</a> · <a href="#paired">',
            1,
        )
    document = document.replace('href="#probes">D2 prompts', 'href="#probes">D1/D2 prompts')
    if config.roster != "mixed_family":
        roster_title = "GPT same-family" if config.roster == "same_family" else "Same-model"
        document = document.replace("<h1>", f"<h1>{roster_title} · ", 1)
        document = document.replace("<title>", f"<title>{roster_title} · ", 1)
    return document, summary


def publish(source, output, *, quality_source=None, diagnostic_source=None):
    manifest, report, records = load_run(source)
    document, summary = render(manifest, report, records, source=source)
    if quality_source:
        document, summary = include_quality(document, summary, quality_source)
        if manifest["protocol_version"] == PUBLIC_HISTORY_VERSION:
            document = document.replace("E · Final answers under", "E2 · Final answers under").replace(
                "E · Equal-budget final answers", "E2 · Final-answer quality"
            )
            for old in ("E1 · Debate versus", "E2 · Was the length", "E3 · Small judge"):
                document = document.replace(f"<h3>{old}", f"<h3>{old.split(' · ', 1)[1]}")
            document = document.replace(
                "Each 20-pair row has 40 decisions across the two presentation orders.",
                "Each complete pair contributes two decisions, one per answer order; a row with 20 complete pairs has 40 decisions.",
            )
        if not manifest["design"]["D_enabled"]:
            document = document.replace(
                "A–D results above are unchanged.", "The debate and B/C results above are unchanged."
            )
    if diagnostic_source:
        from scripts.stateful_probability_diagnostics import include_diagnostics

        document, summary = include_diagnostics(document, summary, diagnostic_source)
    output = Path(output).resolve()
    output.write_text(document)
    atomic_json(output.with_suffix(".json"), summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quality-source", type=Path)
    parser.add_argument("--diagnostic-source", type=Path)
    args = parser.parse_args()
    print(
        publish(args.source, args.output, quality_source=args.quality_source, diagnostic_source=args.diagnostic_source)
    )
