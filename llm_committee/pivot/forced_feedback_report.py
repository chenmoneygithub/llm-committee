"""Readable, offline-only report for paired supplementary feedback cases."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .models import AGREEMENT
from .prompts import FILLER_SENTENCE

MODELS = ("GPT-5.6 Terra", "Qwen3.8-27B", "Inkling")
ARMS = ("natural", "forced")
TEXT_LABELS = ("unchanged", "adjusted", "conclusion_changed", "unjudgeable")


def esc(value):
    return html.escape(str(value))


def rate(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "— (0 valid)"


def number(value, *, signed=False):
    return "—" if value is None else format(value, "+.3f" if signed else ".3f")


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def group_order(dimensions, key):
    result = []
    for dimension, value in zip(dimensions, key, strict=True):
        ordered = AGREEMENT if dimension == "self_label" else ARMS if dimension == "arm" else ()
        result.append((0, ordered.index(value)) if value in ordered else (1, str(value)))
    return tuple(result)


def flatten(records):
    rows = []
    for record in records:
        case = record["case"]
        for arm in ARMS:
            data = record["arms"][arm]
            d = data["D_text"]
            values = {
                treatment: sum(i * dist["probabilities"][label] for i, label in enumerate("ABCDEFG", 1))
                for treatment in ("argument", "control")
                if d and (dist := d.get(treatment)) and dist.get("probabilities")
            }
            before, after = record.get("D_choice_before"), data.get("D_choice_after")
            initial_choice = (record.get("position_before") or {}).get("choice")
            text_label = (data.get("C_text") or {}).get("label")
            rows.append(
                {
                    **case,
                    "arm": arm,
                    "self_label": data["receiver_self_label"],
                    "feedback_label": (data.get("feedback") or {}).get("agreement"),
                    "choice_changed": data["C_choice_changed"],
                    "text_label": text_label,
                    "text_adjusted": float(text_label == "adjusted") if text_label in TEXT_LABELS[:3] else None,
                    "text_conclusion_changed": float(text_label == "conclusion_changed")
                    if text_label in TEXT_LABELS[:3]
                    else None,
                    "D_argument": values.get("argument"),
                    "D_control": values.get("control"),
                    "D_text_delta": d["mean_own_agreement_argument_minus_control"] if d else None,
                    "D_choice_pp": data.get("D_choice_pp"),
                    "choice_p_before": 100 * before["probabilities"][initial_choice]
                    if before and initial_choice in before.get("probabilities", {})
                    else None,
                    "choice_p_after": 100 * after["probabilities"][initial_choice]
                    if after and initial_choice in after.get("probabilities", {})
                    else None,
                }
            )
    return rows


def question_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[row["question_id"]].append(float(row[field]))
    values = [mean(v) for v in groups.values()]
    return (mean(values) if values else None, sum(map(len, groups.values())), len(values))


def paired_summary(rows, field, *, recipient=None):
    cases = defaultdict(dict)
    for row in rows:
        if recipient is None or row["recipient"] == recipient:
            cases[row["id"]][row["arm"]] = row
    differences = defaultdict(list)
    for arms in cases.values():
        if set(arms) == set(ARMS) and all(arms[a].get(field) is not None for a in ARMS):
            differences[arms["forced"]["question_id"]].append(arms["forced"][field] - arms["natural"][field])
    values = [mean(v) for v in differences.values()]
    if not values:
        return {"estimate": None, "interval": None, "questions": 0, "pairs": 0}
    rng = random.Random(f"forced-feedback-report-v1/{field}/{recipient}")
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(5000)) if len(values) > 1 else []
    return {
        "estimate": mean(values),
        "interval": [draws[124], draws[4874]] if draws else None,
        "questions": len(values),
        "pairs": sum(map(len, differences.values())),
    }


def label_groups(rows, dimensions):
    if "self_label" not in dimensions:
        raise ValueError("C/D result tables must retain the recipient self-label")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[d] if row[d] is not None else "missing" for d in dimensions)].append(row)
    index = dimensions.index("self_label")
    for key in list(groups):
        for label in AGREEMENT:
            groups.setdefault((*key[:index], label, *key[index + 1 :]), [])
    return groups


def c_table(rows, dimensions):
    groups = label_groups(rows, dimensions)
    data = []
    for key, group in sorted(groups.items(), key=lambda item: group_order(dimensions, item[0])):
        choices = [r["choice_changed"] for r in group if r["choice_changed"] is not None]
        labels = Counter(r["text_label"] for r in group)
        valid = sum(labels[label] for label in TEXT_LABELS[:3])
        names = [
            MODELS[v] if d == "recipient" else str(v).replace("_", " ") for d, v in zip(dimensions, key, strict=True)
        ]
        data.append(
            [
                *names,
                len(group),
                rate(sum(choices), len(choices)),
                len(group) - len(choices),
                *[rate(labels[label], valid) for label in TEXT_LABELS[:3]],
                labels["unjudgeable"],
                labels[None],
            ]
        )
    return table(
        [
            *[
                d.replace("receiver_T", "Recipient turn T")
                .replace("self_label", "Recipient self-label")
                .replace("_", " ")
                for d in dimensions
            ],
            "Cases",
            "Option changed / valid",
            "Option missing / unjudgeable",
            "Text unchanged / valid",
            "Text adjusted / valid",
            "Conclusion changed / valid",
            "Text unjudgeable",
            "Text missing",
        ],
        data,
    )


def d_table(rows, dimensions, *, kind="text"):
    groups = label_groups([row for row in rows if row["recipient"] in (1, 2)], dimensions)
    data = []
    for key, group in sorted(groups.items(), key=lambda item: group_order(dimensions, item[0])):
        names = [
            MODELS[v] if d == "recipient" else str(v).replace("_", " ") for d, v in zip(dimensions, key, strict=True)
        ]
        fields = (
            ("D_control", "D_argument", "D_text_delta")
            if kind == "text"
            else ("choice_p_before", "choice_p_after", "D_choice_pp")
        )
        # All three means in a row use exactly the same valid paired measurements.
        supported = [row for row in group if all(row.get(field) is not None for field in fields)]
        values = [question_mean(supported, field) for field in fields]
        data.append(
            [
                *names,
                f"{len(supported)}/{len(group)}",
                values[0][2],
                number(values[0][0]),
                number(values[1][0]),
                number(values[2][0], signed=True),
            ]
        )
    return table(
        [
            *[
                d.replace("receiver_T", "Recipient turn T")
                .replace("self_label", "Recipient self-label")
                .replace("_", " ")
                for d in dimensions
            ],
            "Valid pairs / cases",
            "Questions",
            *(
                ["Own-text score: filler", "Own-text score: feedback", "Feedback − filler (points)"]
                if kind == "text"
                else ["Prior option: before (%)", "Prior option: after (%)", "After − before (pp)"]
            ),
        ],
        data,
    )


def render(manifest, report, records):
    rows = flatten(records)
    # Preserve earlier paired aggregates in the machine-readable audit, not as the
    # answer to the self-label-conditional research question in the reading report.
    paired = {
        field: paired_summary(rows, field) for field in ("choice_changed", "text_adjusted", "text_conclusion_changed")
    }
    for member in (1, 2):
        for field in ("D_text_delta", "D_choice_pp"):
            paired[f"{member}/{field}"] = paired_summary(rows, field, recipient=member)
    labels = []
    for role, field in (("Feedback generator", "feedback_label"), ("Recipient's fresh reply", "self_label")):
        for arm in ARMS:
            selected = [r for r in rows if r["arm"] == arm]
            counts = Counter(r[field] for r in selected)
            valid = sum(counts[label] for label in AGREEMENT)
            labels.append(
                [role, arm, valid, *[rate(counts[label], valid) for label in AGREEMENT], len(selected) - valid]
            )
    findings = []
    for label in AGREEMENT:
        parts = []
        for arm in ARMS:
            selected = [row for row in rows if row["self_label"] == label and row["arm"] == arm]
            choices = [row["choice_changed"] for row in selected if row["choice_changed"] is not None]
            text = [row["text_label"] for row in selected if row["text_label"] in TEXT_LABELS[:3]]
            parts.append(
                f"{arm} feedback: option changed {rate(sum(choices), len(choices))}; "
                f"text conclusion changed {rate(text.count('conclusion_changed'), len(text))}"
            )
        findings.append(f"<li>Recipient self-label: {esc(label.replace('_', ' '))} — {esc('; '.join(parts))}.</li>")
    excerpts = []
    for record in records:
        case = record["case"]
        pieces = [
            f"<details><summary>{esc(case['id'])} · {esc(MODELS[case['recipient']])} · T={case['receiver_T']} · {esc(case['tone'])} · {esc(record['question']['text'])}</summary>"
        ]
        pieces.append(
            f"<p>Feedback generator: {esc(MODELS[case['challenger']])}. Recipient: {esc(MODELS[case['recipient']])}.</p>"
        )
        pieces.append(
            "<p>Original options: "
            + "; ".join(f"{chr(65 + i)} — {esc(option)}" for i, option in enumerate(record["question"]["options"]))
            + "</p>"
        )
        before = record.get("position_before") or {}
        pieces.append(
            f"<p><b>Before feedback (choice {esc(before.get('choice'))}):</b> {esc(before.get('position', 'Missing'))}</p>"
        )
        for arm in ARMS:
            data = record["arms"][arm]
            pieces.append(f"<h4>{arm.title()} feedback</h4>")
            for title, value in (
                ("Peer feedback", data.get("feedback")),
                ("Recipient reply", data.get("receiver_reply")),
                ("Position after", data.get("position_after")),
                ("Gemini text comparison", data.get("C_text")),
            ):
                if value is None:
                    display = "Missing measurement"
                elif "reply" in value:
                    display = f"[{value['agreement'].replace('_', ' ')}] {value['reply']}"
                elif "position" in value:
                    display = f"[Selected option: {value['choice'] or 'unjudgeable'}] {value['position']}"
                else:
                    display = f"[{value['label'].replace('_', ' ')}] {value['evidence']}"
                pieces.append(f"<p><b>{esc(title)}:</b> {esc(display)}</p>")
        pieces.append("</details>")
        excerpts.append("".join(pieces))
    mock = manifest["kind"] == "offline_mock"
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strong-disagreement feedback supplement</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#172838;background:#f5f7fa;margin:auto;max-width:1500px;padding:30px}}h1,h2,h3{{line-height:1.2}}section{{background:white;padding:24px;margin:20px 0;border:1px solid #dce2e8;border-radius:10px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde3e8;vertical-align:top}}th{{background:#eaf0f5}}details{{padding:12px;border-bottom:1px solid #dce2e8}}summary{{cursor:pointer}}.notice{{border-left:5px solid #d88a23;padding:12px;background:#fff4df}}p{{max-width:1150px}}
</style><h1>Strong-disagreement feedback supplement</h1>
<p>{"OFFLINE MOCK — synthetic outputs; not scientific results." if mock else "New supplementary data; the main experiment is unchanged."} Status: {esc(report["status"])}. {len(records)}/150 case records; {report["successful_cases"]} complete cases.</p>
<section><h2>What was tested?</h2><p>At 150 presampled points across 60 questions, a member receives either the peer's original feedback or a new response privately instructed to oppose its central conclusion. We generate a fresh recipient reply in both continuations. Only the exact earlier branch is retained; original later turns are not reused.</p>
<p>For example, after A speaks, B supplies natural or forced-opposition feedback, then A responds. The private instruction goes only to B. Both arms start with the same archived prefix and the same measured pre-feedback position.</p>
<p>Three separate questions: what label does A give its reply to B; does A then select a different survey option or express a changed position; and, for Qwen/Inkling, do the option probabilities or endorsement of A's own prior text shift?</p>
<p class="notice">The forced-feedback generator and the recipient are different roles. A peer being instructed to disagree does not force the recipient to disagree in return. No sample is excluded because it gave an unexpected label.</p></section>
<section><h2>Position-change summary, by the recipient's self-label</h2><ul>{"".join(findings)}</ul><p>These counts describe separate outcomes and must not be added together. The natural and forced groups within a label can contain different cases: the label is itself measured after feedback, not fixed before the intervention.</p></section>
<section><h2>1. Who agrees or disagrees?</h2><p>Each percentage is among available labels for that role and arm. The recipient labels below are the labels used to break down C/D. “Fully” and “leaning” retain the main experiment's definitions.</p>
{table(["Role", "Feedback condition", "Valid labels", *[v.replace("_", " ") for v in AGREEMENT], "Missing"], labels)}</section>
<section><h2>2. C: did the recipient's position change?</h2><p>Compare the same member's position immediately before feedback with its position after the new reply. Option change is a deterministic comparison of the two selected letters. A single judge, Gemini 3.8 Flash, separately compares the full position texts without seeing the arm, self-label, or forced instruction.</p>
<p>“Adjusted” means the reasons, scope, or qualifications changed but the main conclusion remained. “Conclusion changed” means the substantive conclusion changed. Wording alone counts as unchanged. These text categories cannot be added to the option-change count.</p>
<h3>By the recipient's self-label</h3>{c_table(rows, ("self_label", "arm"))}
<h3>Within each self-label: debate turn T</h3><p>T is the index of the recipient's new debate reply, not the number of times this member has spoken. No row mixes agreement levels.</p>{c_table(rows, ("self_label", "receiver_T", "arm"))}
<details><summary>Within each self-label: recipient model</summary>{c_table(rows, ("self_label", "recipient", "arm"))}</details>
<details><summary>Within each self-label: this member's participation count</summary><p>This counts the member's own debate replies up to and including its fresh return, excluding its initial answer.</p>{c_table(rows, ("self_label", "recipient_participation", "arm"))}</details>
<p>Label-conditioned tables are descriptive: the intervention can itself change the recipient's label. They are not randomized subgroups. Several cases from one question do not count as several independent questions.</p></section>
<section><h2>3. D: probability measurements of the same recipient</h2><p><b>Own-text endorsement:</b> fix the full position measured before feedback. In separate copies of the pre-reply context, add either peer feedback or procedural filler of approximately the same token length, and ask how much the model still agrees with its own quoted position. No later recipient reply enters this readout.</p>
<p>The filler repeats this actual sentence: “{esc(FILLER_SENTENCE)}” It extends the conversation without taking a position on the question.</p>
<p>The seven-option expected rating ranges from 1 (completely disagree with its own prior text) to 7 (completely agree). “Feedback − filler” compares the two expected ratings. A value of −0.2 means a drop of one fifth of one rating point, not a 20% drop and not necessarily acceptance of the peer's position. There is no validated threshold defining a meaningful change on this scale; use the actual reference scores, paired comparison and C results together.</p>
<p><b>Prior-option probability:</b> from the same pre/post C reads, track the probability assigned to the option selected before feedback. The option is held fixed within each comparison. Negative pp means less probability assigned to that option; the selected option can remain unchanged.</p>
<p>Each mean below first averages within question. Missing top-20 candidates are assigned zero and the returned option probabilities renormalized at temperature 1: this is a truncated-distribution approximation, not proof that omitted options have true probability zero. No top-two margin or entropy adjustment is used.</p>
<h3>Own-text endorsement, by recipient self-label and model</h3>{d_table(rows, ("self_label", "recipient", "arm"))}
<details><summary>Within each self-label and model: own-text endorsement by T</summary>{d_table(rows, ("self_label", "recipient", "receiver_T", "arm"))}</details>
<h3>Prior-option probability, by recipient self-label and model</h3>{d_table(rows, ("self_label", "recipient", "arm"), kind="choice")}
<details><summary>Within each self-label and model: prior-option probability by T</summary>{d_table(rows, ("self_label", "recipient", "receiver_T", "arm"), kind="choice")}</details>
<p>Each row describes its named recipient-label group. Differences between natural and forced rows with the same label are not paired causal estimates, because the intervention may change which cases belong to that group.</p></section>
<section><h2>4. Coverage, missingness and limits</h2><p>Planned: 50 cases per recipient model, 50 per original delivery tone; two or three cases per question. Feedback cuts are at T=2–4, returning to the immediately previous speaker at T=3–5. Samples were selected using route metadata, before examining responses. Natural feedback is reused from the archive; forced feedback is newly generated. This contrast does not separate stance assignment from resulting differences in wording, length, and argument quality.</p>
<p>Complete cases: {report["successful_cases"]}; finalized cases including partial failures: {report["completed_cases"]}. Missing/unjudgeable measurements remain separate in the tables. Technical failures have at most two additional identical attempts; no retry is based on the substantive result. Cost is a token-based estimate, not an invoice: ${report["charged_or_reserved_usd"]:.2f} including any outstanding reservations.</p>
<p>Three types of evidence are distinct: a reply's agreement with the peer, the recipient's own reported position, and context-dependent next-token probabilities. Agreement among them is not evidence of lasting belief change or of judge accuracy.</p></section>
<section><h2>5. Inspect each paired case</h2>{"".join(excerpts)}</section></html>"""
    return document, {"status": report["status"], "case_count": len(records), "rows": rows, "paired": paired}


def load_run(source):
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((source / "cases").glob("*.json"))]
    if report["status"] not in ("completed", "completed_with_failures", "preflight_completed"):
        raise ValueError("Run has not finished this phase")
    if len(records) != report["completed_cases"]:
        raise ValueError("Case coverage does not match report")
    return manifest, report, records


def render_embedded(manifest, report, records):
    """Reuse every result/example, but inherit the host report's document and styles."""
    document, data = render(manifest, report, records)
    _, separator, body = document.partition("</style>")
    if not separator or not body.endswith("</html>"):
        raise ValueError("Unexpected standalone report structure")
    body = body.removesuffix("</html>")
    # The host's h1 remains the only page title; the supplement is a peer of A–E.
    body = re.sub(r"<(\/?)[hH]([1-4])>", lambda m: f"<{m[1]}h{int(m[2]) + 1}>", body)
    return '<section id="forced-feedback" class="feedback-supplement">' + body + "</section>", data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest, report, records = load_run(args.run)
    document, data = render(manifest, report, records)
    companion = args.output.with_suffix(".json")
    if args.output.exists() or companion.exists():
        raise FileExistsError("Choose a new output path; do not overwrite a prior report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    companion.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
