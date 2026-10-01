"""Two-member results, always separate from main data and conditioned on both reply labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import prompts
from .agreement import AGREEMENT
from .dyadic import PAIRS, VERSION
from .forced_feedback_report import esc, number, question_mean, rate, table
from .models import TONES

MODELS = {0: "GPT-5.6 Terra", 1: "Qwen3.8-27B", 2: "Inkling"}
REQUIRED = ("previous_peer_self_label", "current_self_label")
TEXT = ("unchanged", "adjusted", "conclusion_changed", "unjudgeable")
CSS = """
#dyadic { border-top:3px solid var(--accent); padding-top:28px; }
#dyadic .scroll { width:100%; overflow-x:auto; margin:18px 0; }
#dyadic th, #dyadic td { text-align:left; }
#dyadic h4 { font-size:16px; margin:24px 0 10px; }
@media print { #dyadic .scroll { overflow:visible; } }
"""


def label(value):
    return value.replace("_", " ").capitalize() if value is not None else "Not reported"


def flatten(records):
    rows = []
    for record in records:
        for e in record["events"]:
            d = e["D_text"]
            scores = (
                {
                    arm: sum(i * d[arm]["probabilities"][k] for i, k in enumerate("ABCDEFG", 1))
                    for arm in ("argument", "control")
                }
                if d
                else {}
            )
            before, after = e["D_choice_before"], e["D_choice_after"]
            choice = (e["position_before"] or {}).get("choice")
            initial, initial_choice = e["D_choice_initial"], (e["position_initial"] or {}).get("choice")
            rows.append(
                {
                    **e,
                    "question_id": record["question_id"],
                    "text_label": (e["C_text"] or {}).get("label"),
                    "endpoint_text_label": (e["C_initial_to_endpoint"] or {}).get("label"),
                    "D_control": scores.get("control"),
                    "D_argument": scores.get("argument"),
                    "D_text_delta": d["mean_own_agreement_argument_minus_control"] if d else None,
                    "choice_p_before": 100 * before["probabilities"][choice] if before and choice else None,
                    "choice_p_after": 100 * after["probabilities"][choice] if after and choice else None,
                    "initial_p_before": 100 * initial["probabilities"][initial_choice]
                    if initial and initial_choice
                    else None,
                    "initial_p_after": 100 * after["probabilities"][initial_choice]
                    if after and initial_choice
                    else None,
                }
            )
    return rows


def conditional_groups(rows, dimensions, *, agreement_order=AGREEMENT):
    if not set(REQUIRED) <= set(dimensions):
        raise ValueError("Every dyadic C/D table must retain BOTH previous and current self-labels")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[d] for d in dimensions)].append(row)

    def order(key):
        return tuple(
            (0, agreement_order.index(v)) if d in REQUIRED and v in agreement_order else (1, str(v))
            for d, v in zip(dimensions, key, strict=True)
        )

    return [(key, groups[key]) for key in sorted(groups, key=order)]


def names(dimensions, key):
    return [
        label(v) if d in REQUIRED else MODELS[v] if d == "member" else str(v)
        for d, v in zip(dimensions, key, strict=True)
    ]


def headers(dimensions):
    return [
        {
            "previous_peer_self_label": "Peer's preceding self-label",
            "current_self_label": "Receiver's current self-label",
            "member": "Receiver model",
            "T": "Turn T",
            "pair": "Directed pair",
        }.get(d, d)
        for d in dimensions
    ]


def c_table(rows, dimensions=(*REQUIRED, "T"), *, endpoint=False, agreement_order=AGREEMENT):
    data = []
    if endpoint:
        rows = [r for r in rows if r["final_pair"]]
    for key, group in conditional_groups(rows, dimensions, agreement_order=agreement_order):
        field = "C_initial_choice_changed" if endpoint else "C_choice_changed"
        choices = [r[field] for r in group if r[field] is not None]
        text_field = "endpoint_text_label" if endpoint else "text_label"
        text = Counter(r[text_field] for r in group)
        valid = sum(text[t] for t in TEXT[:3])
        data.append(
            [
                *names(dimensions, key),
                len({r["question_id"] for r in group}),
                len(group),
                rate(sum(choices), len(choices)),
                *[rate(text[t], valid) for t in TEXT[:3]],
                text["unjudgeable"],
                text[None],
            ]
        )
    return table(
        [
            *headers(dimensions),
            "Questions",
            "Sampled events",
            "Option changed / valid",
            "Text unchanged / valid",
            "Reasons or qualifications adjusted / valid",
            "Main conclusion changed / valid",
            "Text unjudgeable",
            "Text missing",
        ],
        data,
    )


def d_table(rows, dimensions=(*REQUIRED, "member", "T"), *, kind="text", agreement_order=AGREEMENT):
    fields = (
        ("D_control", "D_argument", "D_text_delta")
        if kind == "text"
        else (
            ("choice_p_before", "choice_p_after", "D_choice_adjacent")
            if kind == "choice"
            else ("initial_p_before", "initial_p_after", "D_choice_initial_to_current")
        )
    )
    data = []
    for key, group in conditional_groups(
        [r for r in rows if r["member"] in (1, 2)], dimensions, agreement_order=agreement_order
    ):
        valid = [r for r in group if all(r.get(f) is not None for f in fields)]
        values = [question_mean(valid, f) for f in fields]
        data.append(
            [
                *names(dimensions, key),
                f"{len(valid)}/{len(group)}",
                values[0][2],
                number(values[0][0]),
                number(values[1][0]),
                number(values[2][0], signed=True),
            ]
        )
    metric_headers = (
        ["Own-text score: filler", "Own-text score: peer", "Peer − filler (1–7 points)"]
        if kind == "text"
        else ["Reference option before (%)", "Same option after (%)", "After − before (pp)"]
    )
    return table([*headers(dimensions), "Valid pairs / sampled events", "Questions", *metric_headers], data)


def load_run(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    if manifest["protocol_version"] != VERSION:
        raise ValueError("Unknown dyadic protocol")
    records = [json.loads((source / "questions" / f"{p['question_id']}.json").read_text()) for p in manifest["plans"]]
    for plan, record in zip(manifest["plans"], records, strict=True):
        if record["question_id"] != plan["question_id"]:
            raise ValueError("Question mismatch")
        if [e["id"] for e in record["events"]] != [e["id"] for e in plan["events"]]:
            raise ValueError("Sampled events changed")
        for e in record["events"]:
            replies = record["formal_replies"][e["tone"]]
            peer = replies.get(f"{e['pair']}-{e['T'] - 1}")
            reply = replies.get(e["node_id"])
            if e["previous_peer_self_label"] != (peer["agreement"] if peer else None):
                raise ValueError("Previous label is not the incoming peer's self-report")
            if e["current_self_label"] != (reply["agreement"] if reply else None):
                raise ValueError("Current label is not the receiver's self-report")
    return manifest, report, records


def render_embedded(manifest, report, records):
    rows = flatten(records)
    a_rows = []
    for tone in TONES:
        for pair in PAIRS:
            replies = [
                reply for r in records for nid, reply in r["formal_replies"][tone].items() if nid.startswith(pair + "-")
            ]
            counts = Counter(r["agreement"] for r in replies)
            a_rows.append([pair, tone, len(replies), *[rate(counts[v], len(replies)) for v in AGREEMENT], counts[None]])
    b_rows = []
    for current in (*AGREEMENT, None):
        group = [r for r in rows if r["current_self_label"] == current]
        if not group and current is None:
            continue
        counts = Counter((r["B"] or {}).get("label") for r in group)
        judged = len(group) - counts[None]
        b_rows.append(
            [
                label(current),
                len(group),
                *[rate(counts[v], judged) for v in AGREEMENT],
                counts["no_position"],
                counts["unjudgeable"],
                counts[None],
            ]
        )
    support = []
    for previous in (*AGREEMENT, None):
        group = [r for r in rows if r["previous_peer_self_label"] == previous]
        if not group and previous is None:
            continue
        support.append(
            [
                label(previous),
                *[sum(r["current_self_label"] == current for r in group) for current in AGREEMENT],
                sum(r["current_self_label"] is None for r in group),
            ]
        )
    details = []
    for record in records:
        examples = []
        for e in record["events"]:
            history = record["formal_replies"][e["tone"]]
            incoming = history.get(f"{e['pair']}-{e['T'] - 1}")
            reply = history.get(e["node_id"])
            texts = [
                ("Own position before", (e["position_before"] or {}).get("position")),
                ("Incoming peer reply", (incoming or {}).get("reply")),
                ("Receiver's reply", (reply or {}).get("reply")),
                ("Own position after", (e["position_after"] or {}).get("position")),
                ("Gemini C judgment", e["C_text"]),
            ]
            examples.append(
                f"<details><summary>{esc(e['pair'])} · {esc(e['tone'])} · T={e['T']} · "
                f"peer: {esc(label(e['previous_peer_self_label']))} → receiver: "
                f"{esc(label(e['current_self_label']))}</summary>"
                + "".join(
                    f"<h5>{esc(name)}</h5><p>{esc(value if value is not None else 'Missing')}</p>"
                    for name, value in texts
                )
                + "</details>"
            )
        details.append(
            f"<details><summary>{esc(record['question_id'])} · {esc(record['question']['text'])}</summary>"
            + "".join(examples)
            + "</details>"
        )
    sections = []
    for previous in (*AGREEMENT, None):
        group = [r for r in rows if r["previous_peer_self_label"] == previous]
        if not group:
            if previous is not None:
                sections.append(f"<h4>Preceding peer self-label: {esc(label(previous))}</h4><p>No sampled cases.</p>")
            continue
        sections.append(f"<h4>Preceding peer self-label: {esc(label(previous))}</h4>" + c_table(group))
    fragment = f"""
<section id="dyadic"><h2>Supplement · Two-member exclusive debate</h2>
<p>This is a separate experiment, not pooled with three-member results or the forced-feedback supplement.
A = GPT-5.6 Terra; B = Qwen3.8-27B; C = Inkling. AB, CA and BC specify the initial sender → first receiver.
Each pair alternates for four new replies; both members speak twice. Only the pair's own history is supplied.</p>
<p>{report["completed_questions"]}/60 questions complete; trajectory outcomes: {esc(report["trajectory_statuses"])}.
Planned: 540 trajectories and 2,160 formal replies. Initial answers are reused from the original independent round.
Token-based charge estimate: ${report["charged_or_reserved_usd"]:.2f}, not a provider invoice.</p>
<p>B/C/D share {len(rows)} presampled events across all 60 questions: one T=2/3/4 from eight of nine pair × tone
trajectories per question. T weights are 1/1/1.5. T=1 has no preceding reply label and is excluded from this
two-label measurement sample, not from debate. Sampled-event percentages are not full-population estimates.</p>
<h3>A · Self-reported labels, by directed pair and tone</h3>
{table(["Directed pair", "Tone", "Replies", *[label(v) for v in AGREEMENT], "Not reported"], a_rows)}
<h3>B · Does the text match the receiver's self-label?</h3>
<p>One external judge, Gemini 3.8 Flash, rates each sampled reply against its incoming peer message, without seeing
the self-label or tone instruction. Each row is the receiver's self-label; columns are Gemini's text ratings.
Percentages use completed judgments in that row. Agreement with this judge is not human validation.</p>
{
        table(
            [
                "Receiver self-label",
                "Sampled replies",
                *[f"Gemini: {label(v)}" for v in AGREEMENT],
                "No position",
                "Unjudgeable",
                "Missing judge",
            ],
            b_rows,
        )
    }
<h3>C/D · Which label combinations were observed?</h3>
<p>The preceding peer label describes its response to this receiver's latest public contribution.
The current label describes the receiver's response to that peer. These are descriptive groups, not randomized
treatments. The labels refer to public messages, not a direct reading of the other member's entire position.
A zero here means no sampled cases in that combination, not zero position change. The C/D tables below show
observed combinations only; they do not fill unsupported groups with zero effects.</p>
{table(["Preceding peer self-label", *[f"Receiver: {label(v)}" for v in AGREEMENT], "Receiver: not reported"], support)}
<h3>C · Own-position change, by both labels and debate turn</h3>
<p>Compare this receiver's previous own participation with its current one on the same trajectory (use its independent
initial position if it has not replied before). Option change is a deterministic comparison of the selected original
options. The same single Gemini judge compares the full position texts: unchanged, adjusted reasons/qualifications,
or changed main conclusion. Option and text changes are separate measures and must not be added.</p>
{"".join(sections)}
<details><summary>C · Cumulative initial → sampled last own position (separate reference)</summary>
<p>Only sampled T=3/4 endpoints are included. These comparisons cover all prior participation in the dyad.
The final two reply labels identify the endpoint context; they do not show when the cumulative change occurred.</p>
{c_table(rows, endpoint=True)}</details>
<h3>D · Quantitative readings at the same sampled events</h3>
<p>Qwen and Inkling only. Debate reasoning is enabled; direct probability reads disable reasoning.
Probabilities are normalized over returned candidates at temperature 1; candidates absent from the endpoint's
top-20 are approximated as zero, not known to be truly impossible. No margin or entropy adjustment is used.</p>
<h4>D1 · Probability of the receiver's previously selected original option</h4>
<p>For example, 80% before and 70% after means −10 percentage points of probability on the same prior option,
even if it remains the selected answer. This is not automatically movement toward the peer. Within each displayed
label × model × T group, average paired changes within question first, then give represented questions equal weight.</p>
{d_table(rows, kind="choice")}
<h4>D2 · Endorsement of the receiver's full prior position: peer versus filler</h4>
<p>Freeze the complete prior position text. Before generating the receiver's new reply, compare its endorsement under
two otherwise identical contexts: incoming peer text versus approximately length-matched repetitions of
“{esc(prompts.FILLER_SENTENCE)}”.
The expected rating ranges from 1 (completely disagree) to 7 (completely agree). Peer − filler = −0.1 means one tenth
of a scale point less endorsement with the peer message, not 10% of members switching views. Negative values alone
do not establish movement toward the peer. Reference scores and paired differences use the same events.</p>
{d_table(rows, kind="text")}
<details><summary>D · Probability of the initial option, initial → current</summary>
<p>This cumulative reference differs from D1's previous-own-turn reference. Both labels still describe the current endpoint.</p>
{d_table(rows, kind="initial")}</details>
<details><summary>Read all {len(rows)} sampled event texts and C judgments</summary>{"".join(details)}</details>
<p>No new human annotation, final-answer synthesis or significance test is claimed here. Sparse or empty label
combinations do not establish absence of an effect. Two- and three-member final endpoints have different depths.</p>
</section>"""
    return fragment, {
        "rows": rows,
        "planned_counts": manifest["planned_counts"],
        "trajectory_statuses": report["trajectory_statuses"],
        "sampling": manifest["sampling"],
        "charged_or_reserved_usd": report["charged_or_reserved_usd"],
    }


def include_dyadic(document, summary, source):
    source = Path(source).resolve()
    manifest, report, records = load_run(source)
    if manifest["kind"] != "paid_dyadic_supplement" or report["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Only a finished real dyadic supplement belongs in the combined report")
    if manifest["source"]["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("Dyadic initial answers are not from this main study")
    if 'id="dyadic"' in document or document.count("<footer>") != 1:
        raise ValueError("Unexpected or already-embedded combined report")
    fragment, data = render_embedded(manifest, report, records)
    document = document.replace("</style>", CSS + "</style>", 1)
    document = document.replace("</nav>", '<a href="#dyadic">Supplement · Two-member debate</a></nav>', 1)
    document = document.replace("<footer>", fragment + "<footer>", 1)
    return document, {
        **summary,
        "supplements": {
            **summary.get("supplements", {}),
            "dyadic": {
                **data,
                "source_directory": str(source),
                "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                "main_study_manifest_sha256": manifest["source"]["manifest_sha256"],
                "pooled_with_main_study": False,
            },
        },
    }
