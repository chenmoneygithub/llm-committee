"""Offline D1 distributions and evidence-linked close reading; no provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from statistics import median

from llm_committee.pivot.dyadic_report import MODELS, REQUIRED, conditional_groups, label
from llm_committee.pivot.forced_feedback_report import esc, question_mean, table
from llm_committee.pivot.models import canonical
from llm_committee.pivot.probabilities import distribution
from llm_committee.pivot.results_report import CSS
from llm_committee.pivot.stateful_study import StatefulGraph
from llm_committee.pivot.strong_agreement import AGREEMENT
from llm_committee.pivot.study import atomic_json

DISPLAY_BAND_PP = 1.0


def identity(row):
    return f"{row['question_id']}:{row['node_id']}:{row['assignment']}"


def number(value, digits=2, signed=False):
    return "—" if value is None else format(value, f"{'+' if signed else ''}.{digits}f")


def grouped(rows):
    result = []
    for key, members in conditional_groups(rows, ("member", *REQUIRED), agreement_order=AGREEMENT):
        usable = [r for r in members if r["D_choice_adjacent"] is not None]
        if not usable:
            continue
        ds = [r["D_choice_adjacent"] for r in usable]
        result.append(
            {
                **dict(zip(("member", *REQUIRED), key, strict=True)),
                "events": len(usable),
                "questions": len({r["question_id"] for r in usable}),
                "excluded": len(members) - len(usable),
                "question_equal_mean_pp": question_mean(usable, "D_choice_adjacent")[0],
                "event_median_pp": median(ds),
                "minimum_pp": min(ds),
                "maximum_pp": max(ds),
                "any_negative": sum(d < 0 for d in ds),
                "any_positive": sum(d > 0 for d in ds),
                "drop_over_1pp": sum(d < -DISPLAY_BAND_PP for d in ds),
                "within_1pp": sum(abs(d) <= DISPLAY_BAND_PP for d in ds),
                "rise_over_1pp": sum(d > DISPLAY_BAND_PP for d in ds),
                "points": [
                    {
                        "event_id": identity(r),
                        "question_id": r["question_id"],
                        "T": r["T"],
                        "delta_pp": r["D_choice_adjacent"],
                        "before_percent": r["choice_p_before"],
                        "after_percent": r["choice_p_after"],
                        "bare_initial_reference": r["previous_reading"].startswith("initial/"),
                        "formal_option_changed": r["C_choice_changed"],
                    }
                    for r in usable
                ],
            }
        )
    return result


def own_state(request):
    for message in request["messages"]:
        try:
            value = json.loads(message["text"])
        except ValueError:
            continue
        if isinstance(value, dict) and "your_current_position" in value:
            return value["your_current_position"]
    return None


def slim_entry(entry):
    request, response, parsed = entry
    return {
        "request": request,
        "output_text": response["text"],
        "reasoning_tokens": response["reasoning_tokens"],
        "readout": {k: v for k, v in parsed["_readout"].items() if k != "prefix_token_ids"},
        "own_state_supplied": own_state(request),
    }


def build(source, notes_file):
    # Lazy import prevents the optional main-report embedding from forming a cycle.
    from scripts.report_stateful_dyadic import collect, load_run

    source = Path(source).resolve()
    manifest, _, records = load_run(source)
    if manifest["kind"] != "paid_stateful_dyadic_pilot":
        raise ValueError("Qualitative notes refer only to the paid stateful cohort")
    notes = json.loads(Path(notes_file).read_text())
    _, original_rows = collect(records)
    rows = [r for r in original_rows if r["member"] in (1, 2) and not (r["assignment"] == "alternate" and r["T"] == 3)]
    contexts = json.loads((source / "source-contexts.json").read_text())
    plans = {p["question_id"]: p for p in manifest["plans"]}
    record_map = {r["question_id"]: r for r in records}
    entries = {}
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        for raw_request, raw_response, raw_parsed in db.execute(
            "SELECT request,response,parsed FROM calls WHERE status='completed' "
            "AND json_extract(request,'$.purpose') IN ('d_choice','d_text')"
        ):
            request = json.loads(raw_request)
            assert request["key"] not in entries
            entries[request["key"]] = (request, json.loads(raw_response), json.loads(raw_parsed))
    seen = set()
    for r in rows:
        if r["D_choice_adjacent"] is None:
            continue
        qid, arm = r["question_id"], r["assignment"]
        graph = StatefulGraph(contexts[qid], plans[qid], manifest)
        keys = [graph.key(arm, f"Dchoice/{r[k]}") for k in ("previous_reading", "current_reading")]
        reference = r["position_before"]["choice"]
        signature = (*keys, reference)
        assert signature not in seen, "Duplicate before/after/reference comparison"
        seen.add(signature)
        for key, stored in zip(keys, (r["D_choice_before"], r["D_choice_after"]), strict=True):
            assert entries[key][0]["effort"] == "none" and entries[key][1]["reasoning_tokens"] == 0
            assert distribution(entries[key][2]["_readout"], graph.question.labels, missing_as_zero=True) == stored
        expected = 100 * (
            r["D_choice_after"]["probabilities"][reference] - r["D_choice_before"]["probabilities"][reference]
        )
        assert abs(expected - r["D_choice_adjacent"]) < 1e-10
    row_map = {identity(r): r for r in rows}
    cases = []
    for note in notes["cases"]:
        r = row_map[identity(note)]
        assert r["previous_peer_self_label"] == "leaning_disagree"
        qid, arm = r["question_id"], r["assignment"]
        graph = StatefulGraph(contexts[qid], plans[qid], manifest)
        record = record_map[qid]
        branch = record["branches"][arm]
        history = branch["formal_replies"]["turn_level"]
        node = graph.route.get(r["node_id"])
        peer, receiver = history[node.parent], history[node.id]
        quote_sources = {
            "peer_reply": peer["reply"],
            "receiver_reply": receiver["reply"],
            "position_before": r["position_before"]["position"],
            "position_after": r["position_after"]["position"],
        }
        for quote in note["quotes"]:
            assert quote["text"] in quote_sources[quote["field"]], (identity(note), quote)
        readings = {}
        for name, rid in (("D1_before", r["previous_reading"]), ("D1_after", r["current_reading"])):
            readings[name] = slim_entry(entries[graph.key(arm, f"Dchoice/{rid}")])
        for treatment in ("argument", "control"):
            readings[f"D2_{treatment}"] = slim_entry(entries[graph.key(arm, f"Dtext/{r['id']}/{treatment}")])
        assert readings["D1_after"]["request"]["messages"][:-1] == readings["D2_argument"]["request"]["messages"][:-1]
        aa, bb = [readings[f"D2_{k}"]["request"]["messages"] for k in ("argument", "control")]
        assert [i for i in range(len(aa)) if aa[i] != bb[i]] == [len(aa) - 2]
        timeline = []
        references = [(0, f"initial/{r['member']}", record["initial_positions"][str(r["member"])])]
        references += [
            (n.depth, f"turn_level/{n.id}", history[n.id])
            for n in graph.route.path(node.id)
            if n.receiver == r["member"]
        ]
        for t, reading, formal in references:
            key = graph.key(arm, f"Dchoice/{reading}")
            value = entries.get(key)
            probs = None
            if value:
                probs = distribution(value[2]["_readout"], graph.question.labels, missing_as_zero=True)["probabilities"]
            timeline.append(
                {"T": t, "formal_choice": formal["choice"], "D1_probabilities": probs, "reading_id": reading}
            )
        cases.append(
            {
                **note,
                "event_id": identity(r),
                "row": r,
                "question": record["question"],
                "peer": peer,
                "receiver": receiver,
                "quote_sources": quote_sources,
                "readings": readings,
                "timeline": timeline,
                "public_history": [
                    {"node": n.id, "tone": branch["tone_schedule"][n.id], "member": n.receiver, **history[n.id]}
                    for n in graph.route.path(node.id)
                ],
            }
        )
    return {
        "schema_version": 1,
        "kind": "offline_stateful_D1_diagnostic",
        "source_directory": str(source),
        "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "notes_sha256": hashlib.sha256(Path(notes_file).read_bytes()).hexdigest(),
        "source_question_sha256": {
            qid: hashlib.sha256((source / "questions" / f"{qid}.json").read_bytes()).hexdigest() for qid in plans
        },
        "scope": notes["scope"],
        "selection": notes["selection"],
        "new_model_calls": 0,
        "display_band_pp": DISPLAY_BAND_PP,
        "rows": rows,
        "groups": grouped(rows),
        "cases": cases,
        "checks": {
            "unique_comparisons_recomputed": len(seen),
            "verbatim_quotes_checked": sum(len(n["quotes"]) for n in notes["cases"]),
        },
    }


def distribution_plot(groups, member):
    groups = [g for g in groups if g["member"] == member]
    width, left, right, top, step = 1130, 285, 885, 68, 60
    height = top + step * len(groups) + 56

    def x(d):
        return left + (d + 100) / 200 * (right - left)

    parts = [
        f'<svg class="distribution" width="100%" style="min-width:850px" viewBox="0 0 {width} {height}" role="img" aria-label="{MODELS[member]} D1 changes by both agreement labels">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="8" y="24" font-size="15">Incoming peer / receiver label</text>',
        '<text x="910" y="24" font-size="13">Drop / within ±1 / rise</text>',
        f'<rect x="{x(-1)}" y="40" width="{x(1) - x(-1)}" height="{step * len(groups) + 12}" fill="#eef0f2"/>',
    ]
    for tick in (-100, -50, 0, 50, 100):
        parts += [
            f'<line x1="{x(tick)}" x2="{x(tick)}" y1="40" y2="{height - 48}" stroke="{"#515b66" if tick == 0 else "#d9e0e6"}"/>',
            f'<text x="{x(tick)}" y="{height - 25}" text-anchor="middle" font-size="13">{tick:+d}</text>',
        ]
    for i, g in enumerate(groups):
        y = top + i * step
        parts.append(f'<line x1="0" x2="{width}" y1="{y + 30}" y2="{y + 30}" stroke="#edf0f3"/>')
        parts.append(
            f'<text x="8" y="{y - 4}" font-size="13">{esc(label(g[REQUIRED[0]]))} / {esc(label(g[REQUIRED[1]]))}</text>'
        )
        parts.append(
            f'<text x="8" y="{y + 15}" font-size="12" fill="#596875">{g["events"]} events · {g["questions"]} questions</text>'
        )
        for k, p in enumerate(sorted(g["points"], key=lambda p: (p["delta_pp"], p["event_id"]))):
            px, py = x(p["delta_pp"]), y + ((k % 5) - 2) * 5
            color = "#b36a08" if p["bare_initial_reference"] else "#2368a2"
            title = f"{p['event_id']}: {p['before_percent']:.6f}% → {p['after_percent']:.6f}%; Δ {p['delta_pp']:+.6f} pp; formal option changed: {p['formal_option_changed']}"
            parts.append(
                f'<circle class="event-dot" cx="{px}" cy="{py}" r="4" fill="{color}" opacity="0.8"><title>{esc(title)}</title></circle>'
            )
        px = x(g["question_equal_mean_pp"])
        parts.append(
            f'<path class="mean-diamond" d="M {px} {y - 8} l 6 8 l -6 8 l -6 -8 Z" fill="none" stroke="#151a20" stroke-width="2"><title>Question-equal mean: {g["question_equal_mean_pp"]:+.4f} pp</title></path>'
        )
        parts.append(
            f'<text x="925" y="{y - 3}" font-size="14">{g["drop_over_1pp"]} / {g["within_1pp"]} / {g["rise_over_1pp"]}</text>'
        )
        parts.append(f'<text x="925" y="{y + 16}" font-size="12">mean {g["question_equal_mean_pp"]:+.2f} pp</text>')
    parts.append(
        f'<text x="{(left + right) / 2}" y="{height - 3}" text-anchor="middle" font-size="13">Change in old survey-option probability (percentage points)</text></svg>'
    )
    return "".join(parts)


def overview(data, detail_href):
    focus = {g["member"]: g for g in data["groups"] if g[REQUIRED[0]] == g[REQUIRED[1]] == "leaning_disagree"}
    descriptions = []
    for member, g in sorted(focus.items()):
        initial_rises = sum(p["bare_initial_reference"] and p["delta_pp"] > DISPLAY_BAND_PP for p in g["points"])
        descriptions.append(
            f"In the {MODELS[member]} leaning-disagree / leaning-disagree row: {g['drop_over_1pp']} events fall by more than 1pp, "
            f"{g['within_1pp']} stay within ±1pp, and {g['rise_over_1pp']} rise by more than 1pp; "
            f"the question-equal mean is {g['question_equal_mean_pp']:+.2f}pp. "
            f"{initial_rises} of the {g['rise_over_1pp']} rises begin with a bare-question initial read."
        )
    return (
        '<section id="d1-diagnostics"><h3>D1 · Distribution and counterintuitive cases</h3>'
        "<p><b>A positive row mean does not mean every event rises, nor that the formal position became stronger.</b> "
        "Dots show each unique sampled comparison; the outline diamond is the same question-equal mean as the table. "
        "Orange dots start from a bare-question initial read, which has no recorded own position; blue dots start from an earlier discussion read. "
        "The ±1pp band is a display convention, not a significance or equivalence threshold. Hover over dots for exact values. "
        "Counts describe sampled events, not independent questions.</p>"
        + "".join(
            f'<h4>{MODELS[m]}</h4><div class="scroll">{distribution_plot(data["groups"], m)}</div>' for m in (1, 2)
        )
        + "<p>"
        + esc(" ".join(descriptions))
        + "</p>"
        "<p><b>What the close reading found:</b> large initial jumps can compare a neutral read that did not favor the formal initial answer "
        "with a later read explicitly supplied with that answer. Later-turn rises also occur alongside concessions without an option change. "
        "In one case the formal option changes B → C even though neutral D1 for old B rises. Another incoming leaning-disagree reply "
        "explicitly shares the receiver’s survey choice and challenges its reasons instead.</p>"
        "<p><b>Interpretation boundary:</b> before/after D1 changes history and the supplied own-position text; it does not isolate the incoming "
        "argument while holding those fixed. D1 uses a different, neutral no-reasoning task from the formal generation. "
        "D2 is instead a matched argument/filler read of endorsement of fixed prior text. The checks of token alignment remain valid; "
        "they do not establish causal reinforcement or validate D1 as a direct measure of the formal choice’s confidence.</p>"
        f'<p><a href="{esc(detail_href)}">Open the ten evidence-linked case studies, full prompts, both probability distributions and the saved paired continuations.</a> '
        "Offline analysis only: no new model calls. Cases were deliberately selected for diagnostic contrasts, not to estimate prevalence.</p></section>"
    )


def probability_table(case):
    r = case["row"]
    reference = r["position_before"]["choice"]
    return table(
        ["Original survey option", "Earlier D1 (%)", "Current D1 (%)"],
        [
            [
                f"{chr(65 + i)} · {text}" + (" ← fixed old option" if chr(65 + i) == reference else ""),
                number(100 * r["D_choice_before"]["probabilities"][chr(65 + i)], 6),
                number(100 * r["D_choice_after"]["probabilities"][chr(65 + i)], 6),
            ]
            for i, text in enumerate(case["question"]["options"])
        ],
    )


def case_html(case, index):
    r, d = case["row"], case["row"]["D_text"]
    header = f'<article id="case-{index}" class="diagnostic-case"><h2>{index}. {esc(case["title"])}</h2>'
    header += f'<p>{esc(case["question"]["text"])}</p><p class="muted">{esc(case["event_id"])} · incoming {label(r[REQUIRED[0]])} / receiver {label(r[REQUIRED[1]])}</p>'
    summary = table(
        ["What was measured", "Earlier / filler", "Current / argument", "Difference"],
        [
            [
                f"D1 · probability of old survey option {r['position_before']['choice']}",
                f"{r['choice_p_before']:.6f}%",
                f"{r['choice_p_after']:.6f}%",
                number(r["D_choice_adjacent"], 4, True) + " pp",
            ],
            [
                "Formal reasoning-enabled survey choice",
                r["position_before"]["choice"],
                r["position_after"]["choice"],
                "Changed" if r["C_choice_changed"] else "Unchanged",
            ],
            [
                f"D2 · probability of fixed filler-modal rating {d['reference_category']}",
                f"{100 * d['reference_probability_control']:.4f}%",
                f"{100 * d['reference_probability_argument']:.4f}%",
                number(d["reference_probability_change_pp"], 4, True) + " pp",
            ],
            [
                "D2 · expected endorsement rating (1–7)",
                number(d["mean_rating_control"], 4),
                number(d["mean_rating_argument"], 4),
                number(d["mean_own_agreement_argument_minus_control"], 4, True) + " points",
            ],
        ],
    )
    parts = [
        header,
        summary,
        f"<p><b>Observed:</b> {esc(case['finding'])}</p><p><b>Interpretation:</b> {esc(case['interpretation'])}</p>",
        f'<p class="muted">Selection: {esc(case["selection_reason"])}</p>',
    ]
    for quote in case["quotes"]:
        parts.append(f"<blockquote><b>{esc(quote['field'].replace('_', ' '))}:</b> “{esc(quote['text'])}”</blockquote>")
    reference = r["position_before"]["choice"]
    parts.append("<h3>Earlier readings on this member’s path</h3>")
    parts.append(
        table(
            ["Participation", "Formal option", f"D1 probability of fixed option {reference} (%)"],
            [
                [
                    "Initial" if a["T"] == 0 else f"T{a['T']}",
                    a["formal_choice"],
                    number(100 * a["D1_probabilities"][reference], 6) if a["D1_probabilities"] else "Not sampled",
                ]
                for a in case["timeline"]
            ],
        )
    )
    parts.append(
        "<p>D1 is taken on the input <em>before</em> each corresponding formal generation, not after observing its new output. The formal option and D1 are different readouts, not two fields of one generation.</p>"
    )
    parts.append("<details><summary>Full text: prior position → incoming argument → reply → updated position</summary>")
    for name in ("position_before", "peer_reply", "receiver_reply", "position_after"):
        parts.append(f"<h4>{esc(name.replace('_', ' ').title())}</h4><p>{esc(case['quote_sources'][name])}</p>")
    parts.append(f"<p><b>Existing Gemini C judgment:</b> {esc(canonical(r['C_text']))}</p></details>")
    parts.append(
        "<details><summary>All original survey-option probabilities (D1)</summary>"
        + probability_table(case)
        + "</details>"
    )
    parts.append("<details><summary>All seven endorsement probabilities (D2)</summary>")
    meanings = (
        "completely disagree",
        "mostly disagree",
        "somewhat disagree",
        "neither",
        "somewhat agree",
        "mostly agree",
        "completely agree",
    )
    parts.append(
        table(
            ["Rating of fixed prior text", "Filler (%)", "Argument (%)"],
            [
                [
                    f"{k} · {mean}",
                    number(100 * d["control"]["probabilities"][k], 6),
                    number(100 * d["argument"]["probabilities"][k], 6),
                ]
                for k, mean in zip("ABCDEFG", meanings, strict=True)
            ],
        )
        + "</details>"
    )
    for name, reading in case["readings"].items():
        meta, req = reading["readout"], reading["request"]
        parts.append(f'<details class="full-prompt"><summary>{name} · full prompt and returned token</summary>')
        parts.append(
            f"<p>Output: <b>{esc(reading['output_text'])}</b> · reasoning tokens: {reading['reasoning_tokens']} · sampled token index: {meta['sampled_output_index']} · effort: {req['effort']}</p>"
        )
        parts.append("<p>Own position supplied: " + esc(canonical(reading["own_state_supplied"])) + "</p>")
        for i, msg in enumerate(req["messages"], 1):
            parts.append(f"<h4>Message {i} · {esc(msg['role'])}</h4><pre>{esc(msg['text'])}</pre>")
        parts.append(
            "<details><summary>Probability provenance and original candidate log probabilities</summary><pre>"
            + esc(json.dumps(meta, ensure_ascii=False, indent=2))
            + "</pre></details></details>"
        )
    parts.append("<details><summary>Full public branch through this event (tone metadata is researcher-only)</summary>")
    for turn in case["public_history"]:
        parts.append(
            f"<h4>{turn['node']} · {MODELS[turn['member']]} · {turn['tone']} · {label(turn['agreement'])}</h4><p>{esc(turn['reply'])}</p>"
        )
    parts.append("</details></article>")
    return "".join(parts)


def render(data):
    styles = (
        CSS
        + "\npre{white-space:pre-wrap;overflow-wrap:anywhere} .scroll{overflow-x:auto} .distribution{width:100%;min-width:850px} .diagnostic-case{margin:48px 0;border-top:2px solid #126459;padding-top:28px} blockquote{border-left:3px solid #126459;margin:16px 0;padding:8px 18px;background:#f4f8f7} td:first-child{text-align:left}"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>D1 distributions and counterintuitive cases</title><style>'
        + styles
        + "</style></head><body><main>"
        "<h1>Why can an old-option probability rise after disagreement?</h1>"
        "<p>This is an offline diagnostic of the saved integrated-position experiment, not a new counterfactual intervention. "
        "No debate, probability probe or judge was rerun. The existing 20-question cohort is unchanged.</p>"
        '<p><a href="turn-tone-dyadic-stateful-2026-09-27.html#d">Back to the main A–E report</a></p>'
        + overview(data, "#cases")
        + '<section id="cases"><h2>Close reading of ten selected cases</h2><p>'
        + esc(data["selection"])
        + "</p>"
        "<p>Interpretations below are assistant-authored post-hoc analysis, not additional Gemini ratings or independent human validation. "
        "Quotations are checked verbatim against saved outputs. Political and historical assertions inside these quotations are model-generated claims, not fact-checked evidence.</p>"
        + "<ol>"
        + "".join(f'<li><a href="#case-{i}">{esc(c["title"])}</a></li>' for i, c in enumerate(data["cases"], 1))
        + "</ol>"
        + "".join(case_html(c, i) for i, c in enumerate(data["cases"], 1))
        + "</section>"
        "<section><h2>What the saved data can—and cannot—settle</h2><p>These examples rule out equating every positive D1 difference with stronger formal commitment. "
        "They do not prove a single cause of D1 concentration. To isolate the immediate argument’s contribution to the option distribution, "
        "a future matched D1 check would hold the exact pre-turn position and history fixed and replace only the incoming message with filler, "
        "keeping the neutral option task identical. That control was not run in this diagnostic. D2 already has an argument/filler comparison, "
        "but for a different target: endorsement of prior full text.</p></section>"
        "<details><summary>Provenance and offline checks</summary><pre>"
        + esc(
            json.dumps(
                {k: v for k, v in data.items() if k not in ("rows", "groups", "cases")}, ensure_ascii=False, indent=2
            )
        )
        + "</pre></details>"
        "</main></body></html>"
    )


def include_diagnostics(document, summary, path):
    path = Path(path).resolve()
    data = json.loads(path.with_suffix(".json").read_text())
    if (
        data["kind"] != "offline_stateful_D1_diagnostic"
        or data["manifest_sha256"] != summary["manifest_sha256"]
        or data["source_directory"] != summary["source_directory"]
    ):
        raise ValueError("D1 diagnostic belongs to a different cohort")
    if "d1-diagnostics" in document or "D1_diagnostic" in summary:
        raise ValueError("Diagnostic already included")
    marker = "<h3>D2 · Endorsement of the fixed pre-turn position: argument vs filler</h3>"
    if document.count(marker) != 1:
        raise ValueError("Cannot locate D section")
    document = document.replace(marker, overview(data, path.name) + marker, 1)
    summary = {
        **summary,
        "D1_diagnostic": {"file": str(path), "groups": data["groups"], "checks": data["checks"], "new_model_calls": 0},
    }
    return document, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("notes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = build(args.source, args.notes)
    args.output.write_text(render(data))
    atomic_json(args.output.with_suffix(".json"), data)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
