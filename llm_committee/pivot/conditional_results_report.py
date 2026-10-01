"""B/C/D reporting is conditional on the current reply's self-label, never just T or model."""

from .agreement import AGREEMENT
from .models import TEXT_SHIFT
from .results_report import MODEL_NAMES, categories, changes, ci_cell, fraction, question_estimate, table

LABEL_NAMES = ("Fully agree", "Leaning agree", "Leaning disagree", "Fully disagree")


def analyze_conditional(questions):
    events = []
    for question in questions:
        for event in question["sampled_events"]:
            row = {
                "question_id": question["question_id"],
                **event,
                "B_label": event["B"]["label"],
                "text_adjacent": event["C_text_adjacent"]["label"],
                "text_final": event["C_text_final"]["label"] if event["C_text_final"] else None,
                "D_text_delta": event["D_text"]["mean_own_agreement_argument_minus_control"]
                if event["D_text"]
                else None,
            }
            for arm in ("argument", "control"):
                row[f"D_text_{arm}"] = (
                    sum(i * event["D_text"][arm]["probabilities"][label] for i, label in enumerate("ABCDEFG", 1))
                    if event["D_text"]
                    else None
                )
            events.append(row)
    labels = list(zip(AGREEMENT, LABEL_NAMES, strict=True))
    if any(event["A"] is None for event in events):
        labels.append((None, "Unreported"))

    def c_row(rows, label, name, *, final=False, **extra):
        return {
            "group": name,
            "self_label": label,
            "label_name": name,
            **extra,
            "questions": len({r["question_id"] for r in rows}),
            "choice": changes(rows, "C_choice_initial_changed" if final else "C_choice_adjacent_changed"),
            "initial_choice": changes(rows, "C_choice_initial_changed"),
            "text": categories(rows, "text_final" if final else "text_adjacent", TEXT_SHIFT),
        }

    b_rows, c_turn, c_final, c_participation, c_model, d_rows, d_turn = [], [], [], [], [], [], []
    for label, name in labels:
        selected = [row for row in events if row["A"] == label]
        b_rows.append(
            {
                "self_label": label,
                "label_name": name,
                **categories(selected, "B_label", (*AGREEMENT, "no_position", "unjudgeable")),
            }
        )
        for turn in range(1, 6):
            c_turn.append(c_row([row for row in selected if row["T"] == turn], label, name, T=turn))
        c_final.append(c_row([row for row in selected if row["final_pair"]], label, name, final=True))
        for count in sorted({row["participation_index"] for row in events}):
            c_participation.append(
                c_row(
                    [row for row in selected if row["participation_index"] == count], label, name, participation=count
                )
            )
        for member in range(3):
            c_model.append(
                c_row([row for row in selected if row["member"] == member], label, name, model=MODEL_NAMES[member])
            )
        for member in (1, 2):
            subset = [row for row in selected if row["member"] == member]
            fields = (
                "D_choice_adjacent_pp",
                "D_choice_initial_pp",
                "D_text_delta",
                "D_text_argument",
                "D_text_control",
            )
            d_rows.append(
                {
                    "model": MODEL_NAMES[member],
                    "self_label": label,
                    "label_name": name,
                    "n": len(subset),
                    **{field: question_estimate(subset, field) for field in fields},
                    "D_choice_final_pp": question_estimate(
                        [row for row in subset if row["final_pair"]], "D_choice_initial_pp"
                    ),
                }
            )
            for turn in range(1, 6):
                at_turn = [row for row in subset if row["T"] == turn]
                d_turn.append(
                    {
                        "model": MODEL_NAMES[member],
                        "self_label": label,
                        "label_name": name,
                        "T": turn,
                        "n": len(at_turn),
                        **{field: question_estimate(at_turn, field) for field in fields},
                    }
                )
    return {
        "grouping": "Current responding member's self-report about the incoming peer message; tone pooled",
        "B": b_rows,
        "C_by_label_turn": c_turn,
        "C_endpoint_by_label": c_final,
        "C_by_label_participation": c_participation,
        "C_by_label_model": c_model,
        "D_by_label": d_rows,
        "D_by_label_turn": d_turn,
        "human_annotation": {
            "status": "not_completed",
            "planned_events": 60,
            "planned_annotators": 2,
            "B_per_person": 60,
            "C_text_per_person": 60,
        },
    }


def render_b(rows):
    return table(
        "Table B1. For each self-reported label, how does Gemini label the reply text?",
        [
            "Member's self-report ↓ / Gemini's text rating →",
            "Sampled replies",
            *LABEL_NAMES,
            "No position",
            "Unjudgeable",
        ],
        [
            [
                r["label_name"],
                r["n"],
                *[
                    f"{r['counts'][label]} ({100 * r['counts'][label] / r['n']:.1f}%)" if r["n"] else "—"
                    for label in (*AGREEMENT, "no_position", "unjudgeable")
                ],
            ]
            for r in rows
        ],
        "Each row includes only replies with that self-reported label. Cells show count (percentage of that row), "
        "not a percentage of all sampled replies. Diagonal cells are exact matches; off-diagonal cells show how "
        "the text rating differs. An empty row is not evidence of perfect agreement.",
    )


def d_choice_table(rows, title, key="D_choice_adjacent_pp"):
    return table(
        title,
        ["Current self-label", "Member", "Valid comparisons / questions", "Probability change (pp) [95% interval]"],
        [[r["label_name"], r["model"], f"{r[key]['n']} / {r[key]['questions']}", ci_cell(r[key])] for r in rows],
        "Keep the reference option fixed within each comparison. Means first average within each supported question, "
        "then weight those questions equally. A dash means no available comparison, not zero change.",
    )


def d_text_table(rows, title):
    return table(
        title,
        [
            "Current self-label",
            "Member",
            "Text pairs / questions",
            "With filler",
            "With peer",
            "Peer − filler (points) [95% interval]",
        ],
        [
            [
                r["label_name"],
                r["model"],
                f"{r['D_text_delta']['n']} / {r['D_text_delta']['questions']}",
                f"{r['D_text_control']['mean']:.3f}" if r["D_text_control"]["mean"] is not None else "—",
                f"{r['D_text_argument']['mean']:.3f}" if r["D_text_argument"]["mean"] is not None else "—",
                ci_cell(r["D_text_delta"], digits=3),
            ]
            for r in rows
        ],
        "The score is endorsement of the same full prior position on the 1–7 scale. Self-labels are observed "
        "on the corresponding formal reply; these conditional associations are not causal effects of labels.",
    )


def b_finding(rows):
    return (
        "<li><strong>B · Does each self-label fit the reply text?</strong> Gemini's exact matches within each self-label category are "
        + "; ".join(f"{r['label_name']}: {fraction(r['counts'].get(r['self_label'], 0), r['n'])}" for r in rows)
        + ". See the row-normalized cross-table for the actual mismatches. The Fully disagree category is sparse; these are model–self-report comparisons, not human-validated accuracy.</li>"
    )
