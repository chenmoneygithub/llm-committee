"""Paired continuations, embedded without changing the original-assignment tables."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .agreement import AGREEMENT
from .dyadic import PAIRS, make_route
from .dyadic_report import MODELS, REQUIRED, c_table, d_table, flatten, label
from .forced_feedback_report import esc, number, rate, table
from .models import TONES
from .turn_tone import STREAM
from .turn_tone_fork import VERSION

CSS = """
#paired { border-top:3px solid var(--accent); margin-top:40px; padding-top:24px; }
.fork-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.fork-grid article { border:1px solid #d9e1e6; border-radius:8px; padding:14px; overflow-wrap:anywhere; }
.fork-grid p, .shared-text { white-space:pre-wrap; }
.fork-case { margin:14px 0; }
@media(max-width:760px) { .fork-grid { grid-template-columns:1fr; } }
"""


def load_run(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] != VERSION or report["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Need a completed paired-continuation run")
    original_root = Path(manifest["source"]["directory"])
    if (
        hashlib.sha256((original_root / "manifest.json").read_bytes()).hexdigest()
        != manifest["source"]["manifest_sha256"]
    ):
        raise ValueError("Original manifest changed")
    original, alternate = [], []
    route = make_route()
    for plan in manifest["plans"]:
        qid = plan["question_id"]
        raw = (original_root / "questions" / f"{qid}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest["source"]["question_sha256"][qid]
        before = json.loads(raw)
        after = json.loads((source / "questions" / f"{qid}.json").read_text())
        assert before["question"] == after["question"] and after["question_id"] == qid
        assert before["initial_answers"] == after["initial_answers"]
        assert after["tone_schedule"] == plan["tone_schedule"]
        assert before["tone_schedule"] == plan["original_tone_schedule"]
        assert after["probe_tone"] == before["probe_tone"] == "neutral"
        history = after["formal_replies"][STREAM]
        for node in route.nodes:
            if node.depth <= 2:
                assert history[node.id] == before["formal_replies"][STREAM][node.id]
                assert after["tone_schedule"][node.id] == before["tone_schedule"][node.id]
            else:
                assert after["tone_schedule"][node.id] != before["tone_schedule"][node.id]
        assert (
            [e["id"] for e in before["events"]]
            == [e["id"] for e in after["events"]]
            == [e["id"] for e in plan["events"]]
        )
        for a, b in zip(before["events"], after["events"], strict=True):
            node = route.get(b["node_id"])
            assert b["previous_peer_self_label"] == (history.get(node.parent) or {}).get("agreement")
            assert b["current_self_label"] == (history.get(node.id) or {}).get("agreement")
            assert b["current_tone"] == plan["tone_schedule"][node.id]
            assert b["previous_tone"] == plan["tone_schedule"][node.parent]
            if b["T"] <= 3:
                assert a["D_text"] == b["D_text"]
            if b["T"] == 2:
                assert all(b[key] == value for key, value in a.items())
        original.append(before)
        alternate.append(after)
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        ledger = db.execute("SELECT status,COUNT(*),SUM(charge) FROM calls GROUP BY status").fetchall()
    report = {
        **report,
        "cost_accounting": {
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
    return manifest, report, original, alternate


def judgment(value):
    if value is None:
        return "Missing"
    names = {
        "unchanged": "Unchanged",
        "adjusted": "Reasons or qualifications adjusted",
        "conclusion_changed": "Main conclusion changed",
        "unjudgeable": "Unjudgeable",
    }
    return f"{names.get(value['label'], label(value['label']))}: {value.get('evidence', '')}"


def turn_panel(record, pair, t, assignment):
    nid = f"{pair}-{t}"
    node = make_route().get(nid)
    reply = record["formal_replies"][STREAM].get(nid)
    event = next((e for e in record["events"] if e["node_id"] == nid), None)
    text = f"<h4>{assignment} · {esc(record['tone_schedule'][nid])}</h4><p><strong>{esc(MODELS[node.receiver])}</strong> · self-label: {esc(label((reply or {}).get('agreement')))}</p>"
    text += f"<p>{esc(reply['reply'] if reply else 'Missing reply')}</p>"
    if event:
        text += f"<p><strong>Incoming peer's self-label:</strong> {esc(label(event['previous_peer_self_label']))}</p>"
        text += f"<p><strong>B — Gemini text rating:</strong> {esc(judgment(event['B']))}</p>"
        for title, value in (
            ("Previous own position (shared reference)", event["position_before"]),
            ("Current own position", event["position_after"]),
        ):
            text += f"<details><summary>{title} · option {esc(value['choice'] if value else 'missing')}</summary><p>{esc(value['position'] if value else 'Missing')}</p></details>"
        text += f"<p><strong>C — previous own → current:</strong> {esc(judgment(event['C_text']))}</p>"
        text += f"<p><strong>C — initial → current:</strong> {esc(judgment(event['C_initial_to_endpoint']))}</p>"
        if event["D_choice_adjacent"] is not None:
            text += f"<p><strong>D — probability of the same previous option:</strong> {number(event['D_choice_adjacent'], signed=True)} pp.</p>"
        if event["D_text"]:
            d = event["D_text"]
            score = {
                arm: sum(i * d[arm]["probabilities"][k] for i, k in enumerate("ABCDEFG", 1))
                for arm in ("argument", "control")
            }
            text += f"<p><strong>D — endorsement of full prior position:</strong> filler {number(score['control'])}, peer {number(score['argument'])}; peer − filler {number(d['mean_own_agreement_argument_minus_control'], signed=True)} points."
            text += " Shared pre-reply reading, NOT a new tone contrast.</p>" if t == 3 else "</p>"
    else:
        text += '<p class="muted">This node was not in the original B/C/D sample; neither assignment is newly sampled here.</p>'
    return f"<article>{text}</article>"


def case_detail(before, after, pair, *, show_prefix_measurements=False):
    q = before["question"]
    initial = "".join(
        f"<h4>{esc(MODELS[m])}: initial position</h4><p class='shared-text'>{esc(before['initial_answers'][str(m)])}</p>"
        for m in PAIRS[pair]
    )
    options = "".join(f"<li>{chr(65 + i)}: {esc(option)}</li>" for i, option in enumerate(q["options"]))
    shared = ""
    for t in (1, 2):
        nid = f"{pair}-{t}"
        if show_prefix_measurements:
            shared += f"<h4>T={t} · shared prefix</h4>" + turn_panel(before, pair, t, "Shared prefix")
            continue
        r = before["formal_replies"][STREAM][nid]
        shared += f"<h4>T={t} · {esc(MODELS[make_route().get(nid).receiver])} · {esc(before['tone_schedule'][nid])} · {esc(label(r['agreement']))}</h4><p class='shared-text'>{esc(r['reply'])}</p>"
    panels = "".join(
        f"<h4>T={t} — {'identical input history' if t == 3 else 'divergent histories after T3'}</h4><div class='fork-grid'>{turn_panel(before, pair, t, 'Original')}{turn_panel(after, pair, t, 'Alternate')}</div>"
        for t in (3, 4)
    )
    return f"<details class='fork-case'><summary>{esc(q['id'])} · {pair} — {esc(q['text'])}</summary><ul>{options}</ul><details><summary>Shared initial positions and exact T1/T2 prefix (shown once)</summary>{initial}{shared}</details>{panels}</details>"


def render(manifest, report, original, alternate):
    rows = [
        {**r, "assignment": arm}
        for arm, records in (("Original", original), ("Alternate", alternate))
        for r in flatten(records)
        if r["T"] >= 3
    ]
    original_rows = {(r["question_id"], r["node_id"]): r for r in rows if r["assignment"] == "Original"}
    paired_rows = [
        {"original": original_rows[r["question_id"], r["node_id"]], "alternate": r}
        for r in rows
        if r["assignment"] == "Alternate"
    ]
    a_rows = []
    for t in (3, 4):
        for tone in TONES:
            for arm, records in (("Original", original), ("Alternate", alternate)):
                replies = [
                    reply
                    for record in records
                    for nid, reply in record["formal_replies"][STREAM].items()
                    if nid.endswith(f"-{t}") and record["tone_schedule"][nid] == tone
                ]
                counts = Counter(r["agreement"] for r in replies)
                a_rows.append(
                    [t, tone, arm, len(replies), *[rate(counts[v], len(replies)) for v in AGREEMENT], counts[None]]
                )
    b_rows = []
    for current in (*AGREEMENT, None):
        for arm in ("Original", "Alternate"):
            group = [r for r in rows if r["assignment"] == arm and r["current_self_label"] == current]
            if current is None and not group:
                continue
            counts = Counter((r["B"] or {}).get("label") for r in group)
            valid = len(group) - counts[None]
            b_rows.append(
                [
                    arm,
                    label(current),
                    len(group),
                    *[rate(counts[v], valid) for v in AGREEMENT],
                    counts["no_position"],
                    counts["unjudgeable"],
                    counts[None],
                ]
            )
    transitions = []
    tone_pairs = []
    for t in (3, 4):
        counts, tones = Counter(), Counter()
        for before, after in zip(original, alternate, strict=True):
            for pair in PAIRS:
                nid = f"{pair}-{t}"
                a = before["formal_replies"][STREAM].get(nid)
                b = after["formal_replies"][STREAM].get(nid)
                counts[(a or {}).get("agreement"), (b or {}).get("agreement")] += 1
                tones[before["tone_schedule"][nid], after["tone_schedule"][nid]] += 1
        for old in (*AGREEMENT, None):
            if old is None and not any(a is None for a, b in counts):
                continue
            transitions.append([t, label(old), *[counts[old, new] for new in AGREEMENT], counts[old, None]])
        tone_pairs += [[t, a, b, n] for (a, b), n in sorted(tones.items())]
    cost = report["cost_accounting"]
    billing = f"Additional recorded responses: token-estimated US${cost['received_response_estimate_usd']:.2f}."
    if cost["unresolved_attempts"]:
        billing += f" {cost['unresolved_attempts']} unresolved-billing attempt(s) retain US${cost['unresolved_reservations_usd']:.2f} in conservative reservations, not confirmed spending."
    new_count = sum(int(nid.rsplit("-", 1)[1]) >= 3 for r in alternate for nid in r["formal_replies"][STREAM])
    mock = '<p class="callout">OFFLINE MOCK — not scientific results.</p>' if manifest["kind"] == "offline_mock" else ""
    html = f"""<section id="paired"><style>{CSS}</style><h2>Paired continuations · same T1/T2, different tones at T3/T4</h2>{mock}
<p>{len(original)} questions, {len(original) * 3} matched paths. {new_count}/120 new replies; trajectory outcomes: {esc(report["trajectory_statuses"])}.
The combined pilot has {240 + new_count} unique replies: 120 shared prefix replies, 120 original suffix replies, and {new_count} alternate suffix replies.
Shared prefixes are counted once; there are still 20 independent questions. {esc(billing)} These are not provider invoices.</p>
<p>Only T3/T4 were regenerated. Each alternate turn has a different assigned tone from its original counterpart;
assignments were frozen before new outputs, balanced at 40 turns per tone. Initial positions, routes, prompt wording,
neutral measurements, models and judge are unchanged. Historical tone instructions and self-label metadata are not shown to debaters.</p>
<p><strong>T3:</strong> the two replies see exactly the same history and incoming message, with different current private tone instructions.
<strong>T4:</strong> both the incoming T3 message and current tone differ. This compares subsequent paths, not an isolated T4 instruction effect.
One continuation per assignment also includes generation variability; case differences are not certain causal effects.</p>
<p>The original 107 sampled T3/T4 events have matching alternate events. The 53 shared T2 events are not duplicated here.
The original-assignment tables elsewhere on this page are unchanged.</p>
<details><summary>Assigned tone pairs at each changed turn</summary>{table(["T", "Original tone", "Alternate tone", "Matched paths"], tone_pairs)}</details>
<h3>A · How do the two replies label their incoming peer messages?</h3>
<p>Counts of matched label transitions, not stance changes on the survey question. At T3, both replies rate the same incoming message;
at T4, the incoming messages have diverged. Missing replies are not assigned an agreement label.</p>
{table(["T", "Original self-label", *[f"Alternate: {label(v)}" for v in AGREEMENT], "Alternate: missing/unreported"], transitions)}
<details><summary>A · Four-label distributions by turn, current assigned tone and assignment</summary>
{table(["T", "Assigned tone", "Assignment", "Replies", *[label(v) for v in AGREEMENT], "Not reported"], a_rows)}</details>
<h3>B · Self-label versus Gemini's reading of the text</h3>
<p>The same single Gemini 3.8 Flash judge reads the peer message and reply without self-labels or private tones.
Rows condition on the current self-label, separately for each assignment. No human annotation was added.</p>
{table(["Assignment", "Current self-label", "Sampled replies", *[f"Gemini: {label(v)}" for v in AGREEMENT], "No position", "Unjudgeable", "Missing"], b_rows)}
<h3>C · Position changes after the reply</h3>
<p>Option changes compare the member's selected survey option with its own previous participation; Gemini separately compares the full
position texts. Rows retain both reply labels, assignment and T. Labels are observed outcomes: the members of a label group can differ
between assignments, so subtracting these group averages would not be a matched treatment-effect estimate.</p>
<details><summary>C · Previous own position → current, by both labels and T</summary>{c_table(rows, (*REQUIRED, "assignment", "T"))}</details>
<details><summary>C · Initial position → current, by both labels and T</summary>{c_table(rows, (*REQUIRED, "assignment", "T"), endpoint=True)}</details>
<h3>D · Additional probability readings for Qwen and Inkling</h3>
<p>The option reading follows the new C position response. For example, 80% → 70% for the same prior option is −10 pp;
it is not automatically movement toward the peer. Readings are contextual output probabilities, not direct access to beliefs.</p>
<details><summary>D-choice · Previous own option probability → current, both labels / model / T</summary>{d_table(rows, (*REQUIRED, "assignment", "member", "T"), kind="choice")}</details>
<details><summary>D-choice · Initial option probability → current, both labels / model / T</summary>{d_table(rows, (*REQUIRED, "assignment", "member", "T"), kind="initial")}</details>
<p><strong>D-text at T3 is shared, not a new observation.</strong> It asks about endorsement of the prior full position before the T3 reply,
with a neutral probe. Since the incoming T2 message and all earlier context are identical, we reuse the original reading.
Its equality across assignments is built into the design, not evidence of no tone effect.</p>
<p>At T4, the incoming T3 message changes. We therefore rebuild both peer and length-matched filler inputs. The expected endorsement
score runs from 1 (completely disagree) to 7 (completely agree). A −0.1 peer-minus-filler difference is one tenth of a scale point,
not 10% of members switching positions. Reference scores and differences use matched valid readings, averaged within question first.</p>
<details><summary>D-text · T4 only: peer versus filler, both labels and receiver model</summary>{d_table([r for r in rows if r["T"] == 4], (*REQUIRED, "assignment", "member", "T"), kind="text")}</details>
<h3>Read all 60 matched paths side by side</h3>
<p>Each case shows its shared initial positions and T1/T2 once, then the original and alternate T3/T4 replies beside their measurements.
Tone and self-label annotations are for the researcher; they are not historical prompt metadata.</p>
{"".join(case_detail(a, b, pair) for a, b in zip(original, alternate, strict=True) for pair in PAIRS)}
</section>"""
    summary = {
        "protocol_version": VERSION,
        "planned_counts": manifest["planned_counts"],
        "report": report,
        "rows": rows,
        "paired_events": paired_rows,
        "A_label_transitions": transitions,
        "A_by_tone_T_assignment": a_rows,
        "B_by_self_label_assignment": b_rows,
        "tone_pair_counts": tone_pairs,
        "original_tables_unchanged": True,
        "shared_prefix_counted_once": True,
        "D_text_T3_reused_not_an_effect": True,
    }
    return html, summary


def include_fork(document, summary, source):
    source = Path(source).resolve()
    manifest, report, original, alternate = load_run(source)
    if manifest["kind"] != "paid_turn_tone_fork":
        raise ValueError("Do not publish mock continuations as experimental results")
    if (
        manifest["source"]["manifest_sha256"] != summary["manifest_sha256"]
        or str(Path(manifest["source"]["directory"]).resolve()) != summary["source_directory"]
    ):
        raise ValueError("Continuation belongs to a different pilot")
    if 'id="paired"' in document or "paired_continuation" in summary:
        raise ValueError("Paired continuation already included")
    embedded, addition = render(manifest, report, original, alternate)
    addition["source_directory"] = str(source)
    addition["manifest_sha256"] = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    banner = '<p class="callout"><strong>Expanded pilot: shared T1/T2, alternate T3/T4.</strong> The original assignment is retained below. <a href="#paired">Jump to the paired continuation and side-by-side cases</a>. Shared replies and probes are not new observations.</p>'
    document = document.replace("</h1>", "</h1>" + banner, 1)
    document = document.replace("</nav>", '<a href="#paired">Paired tone continuations</a></nav>', 1)
    document = document.replace("<h2>What was run?</h2>", "<h2>Original assignment: what was run?</h2>", 1)
    document = document.replace("<footer>", embedded + "<footer>", 1)
    return document, {**summary, "paired_continuation": addition}
