"""Offline report arithmetic: correct units, missingness, reproducibility and HTML."""

import json

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.results_report import (
    analyze,
    categories,
    changes,
    ci_cell,
    fraction,
    question_estimate,
    table,
)


def test_question_mean_does_not_treat_repeated_events_as_independent_questions():
    records = [{"question_id": "a", "x": 1}] * 9 + [{"question_id": "b", "x": 0}]
    result = question_estimate(records, "x")
    assert result["n"] == 10
    assert result["questions"] == 2
    assert result["mean"] == 0.5  # Not the pooled event mean of 0.9.
    assert result == question_estimate(list(reversed(records)), "x")


def test_missing_measurements_do_not_become_zeros_or_new_questions():
    records = [{"question_id": "a", "x": None}, {"question_id": "b", "x": 0.25}]
    result = question_estimate(records, "x")
    assert result == {"n": 1, "questions": 1, "mean": 0.25, "ci": None}
    assert "one question" in ci_cell(result)
    assert question_estimate([], "x") == {"n": 0, "questions": 0, "mean": None, "ci": None}


def test_choice_denominator_excludes_unselectable_not_unchanged():
    records = [{"x": True}, {"x": False}, {"x": False}, {"x": None}]
    assert changes(records, "x") == {"n": 4, "valid": 3, "changed": 1, "unavailable": 1}
    assert fraction(1, 3) == "1/3 (33.3%)"
    assert "no eligible observations" in fraction(0, 0)


def test_empty_categories_and_unjudgeable_are_visible():
    result = categories([{"x": "yes"}, {"x": "unjudgeable"}], "x", ("yes", "no", "unjudgeable"))
    assert result == {"n": 2, "counts": {"yes": 1, "no": 0, "unjudgeable": 1}, "other": 0}
    assert categories([{"x": None}], "x", ("yes",))["other"] == 1


def test_bootstrap_is_on_whole_question_means_for_pooled_tones():
    records = [{"question_id": q, "x": x} for q in ("a", "b", "c") for x in (0, 0.5, 1)]
    result = question_estimate(records, "x")
    assert result["questions"] == 3 and result["n"] == 9
    assert result["mean"] == 0.5 and result["ci"] == [0.5, 0.5]


def test_nonfinite_data_is_not_rendered_as_a_result():
    with pytest.raises(ValueError, match="Nonfinite"):
        question_estimate([{"question_id": "a", "x": float("nan")}], "x")


def test_table_escapes_source_text_and_has_correct_cell_counts():
    document = table("<Unsafe>", ["Label", "Value"], [["<script>alert(1)</script>", "2 & 3"]], "A < B")
    parsed = BeautifulSoup(document, "html.parser")
    assert not parsed.find("script")
    assert parsed.caption.text == "<Unsafe>"
    assert len(parsed.tbody.find_all("tr")) == 1
    assert len(parsed.tbody.tr.find_all(["td", "th"], recursive=False)) == 2
    assert parsed.tbody.tr.th["scope"] == "row"


@pytest.mark.parametrize(
    "kind,status,count",
    [
        ("offline_mock", "completed", 60),
        ("paid_main_study", "completed_with_failures", 60),
        ("paid_main_study", "completed", 3),
    ],
)
def test_report_rejects_synthetic_incomplete_or_other_sized_runs(tmp_path, kind, status, count):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "config": {"roster": "mixed_family"},
                "questions": [{}] * count,
            }
        )
    )
    (tmp_path / "report.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match="real 60-question"):
        analyze(tmp_path)


def test_leaning_report_does_not_accept_reference_run(tmp_path):
    from llm_committee.pivot.agreement import LEGACY_PROMPT_VERSION
    from llm_committee.pivot.leaning_results_report import analyze_leaning

    (tmp_path / "manifest.json").write_text(json.dumps({"prompt_version": LEGACY_PROMPT_VERSION}))
    with pytest.raises(ValueError, match="legacy relabeling"):
        analyze_leaning(tmp_path)


def test_default_reference_report_still_rejects_new_rubric(tmp_path):
    from llm_committee.pivot.leaning_results_report import PROMPT_VERSION

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "paid_main_study",
                "config": {"roster": "mixed_family"},
                "questions": [{}] * 60,
                "prompt_version": PROMPT_VERSION,
            }
        )
    )
    (tmp_path / "report.json").write_text(json.dumps({"status": "completed"}))
    with pytest.raises(ValueError, match="protocol mismatch"):
        analyze(tmp_path)


def test_leaning_table_uses_new_labels_without_aliasing():
    from llm_committee.pivot.agreement import AGREEMENT, LEGACY_AGREEMENT
    from llm_committee.pivot.leaning_results_report import agreement_cells

    row = {"n": 10, "counts": dict(zip(AGREEMENT, (1, 2, 3, 4), strict=True))}
    assert agreement_cells(row) == [10, "10.0%", "20.0%", "30.0%", "40.0%"]
    with pytest.raises(KeyError):
        agreement_cells({"n": 10, "counts": dict(zip(LEGACY_AGREEMENT, (1, 2, 3, 4), strict=True))})


def test_leaning_label_comparison_explains_both_directions_and_denominator():
    from llm_committee.pivot.leaning_results_report import render_label_comparison

    ab = {"n": 480, "counts": {"same": 433, "more_disagreeing": 24, "less_disagreeing": 23}}
    parsed = BeautifulSoup(render_label_comparison(ab), "html.parser")
    rows = [[cell.get_text() for cell in row.find_all(["th", "td"])] for row in parsed.tbody.find_all("tr")]
    assert "480" in parsed.thead.get_text()
    assert rows[0][1:3] == ["433", "90.2%"]
    assert rows[1][1:3] == ["24", "5.0%"]
    assert "member says Fully agree" in rows[1][3]
    assert "reply as Leaning agree" in rows[1][3]
    assert rows[2][1:3] == ["23", "4.8%"]
    assert "member says Leaning agree" in rows[2][3]
    assert "reply as Fully agree" in rows[2][3]


def test_probability_explanation_does_not_quote_cross_label_model_means():
    from llm_committee.pivot.leaning_results_report import render_probability_interpretation

    summary = {
        "D": [
            {
                "model": "Qwen3.8-27B",
                "D_text_control": {"mean": 6.737588513101374},
                "D_text_argument": {"mean": 6.582634291914595},
                "D_text_delta": {"mean": -0.15495422118677854},
            }
        ],
    }
    text = BeautifulSoup(render_probability_interpretation(summary), "html.parser").get_text()
    assert "6.738" not in text and "6.583" not in text and "-0.155" not in text
    assert "within the same self-label and model row" in text
    assert "half a scale step, not a 50% drop in probability" in text
    assert "illustration, not a pooled result" in text
    assert "same self-label category in each layer" in text


def test_c_turn_breakdown_uses_current_self_label_and_preserves_missing_choices():
    from llm_committee.pivot.leaning_results_report import position_turn_breakdown

    questions = [
        {
            "question_id": "first",
            "sampled_events": [
                {
                    "T": 3,
                    "participation_index": 2,
                    "A": "leaning_disagree",
                    "B": {"label": "fully_agree"},
                    "C_choice_adjacent_changed": True,
                    "C_text_adjacent": {"label": "conclusion_changed"},
                    "final_trajectories": ["left", "right"],
                }
            ],
        },
        {
            "question_id": "second",
            "sampled_events": [
                {
                    "T": 3,
                    "participation_index": 1,
                    "A": "leaning_disagree",
                    "C_choice_adjacent_changed": None,
                    "C_text_adjacent": {"label": "adjusted"},
                }
            ],
        },
    ]
    rows = position_turn_breakdown(questions)
    assert len(rows) == 20
    row = next(r for r in rows if r["T"] == 3 and r["self_label"] == "leaning_disagree")
    assert row["questions"] == 2
    assert row["choice"] == {"n": 2, "valid": 1, "changed": 1, "unavailable": 1}
    assert row["text"]["n"] == 2  # Shared final paths do not duplicate an event.
    assert row["text"]["counts"]["adjusted"] == row["text"]["counts"]["conclusion_changed"] == 1
    assert sum(r["text"]["n"] for r in rows) == 2
    assert all(r["text"]["n"] == 0 for r in rows if r["self_label"] == "fully_agree")


def test_c_turn_breakdown_does_not_drop_an_unreported_self_label():
    from llm_committee.pivot.leaning_results_report import position_turn_breakdown

    rows = position_turn_breakdown(
        [
            {
                "question_id": "a",
                "sampled_events": [
                    {
                        "T": 2,
                        "A": None,
                        "C_choice_adjacent_changed": False,
                        "C_text_adjacent": {"label": "unchanged"},
                    }
                ],
            }
        ]
    )
    row = next(r for r in rows if r["T"] == 2 and r["self_label"] is None)
    assert row["label_name"] == "Unreported"
    assert row["choice"]["valid"] == row["text"]["n"] == 1
    assert sum(r["text"]["n"] for r in rows) == 1


def test_c_empty_turn_label_group_is_not_rendered_as_zero_percent_change():
    from llm_committee.pivot.leaning_results_report import position_group_cells, position_turn_breakdown

    empty = position_turn_breakdown([])[0]
    cells = position_group_cells(empty)
    assert cells == ["T = 1 · Fully agree", 0, 0, "—", 0, "—", "—", "—", 0]


def test_c_tables_keep_labels_including_turn_participation_and_endpoints():
    from llm_committee.pivot.leaning_results_report import position_turn_breakdown, render_position_tables

    empty = position_turn_breakdown([])[0]
    summary = {
        "C_by_self_label": [empty],
        "conditional_analysis": {
            "C_by_label_turn": [dict(row, initial_choice=row["choice"]) for row in position_turn_breakdown([])],
            "C_endpoint_by_label": [empty],
            "C_by_label_participation": [dict(empty, participation=1)],
            "C_by_label_model": [dict(empty, model="Qwen3.8-27B")],
        },
    }
    parsed = BeautifulSoup("".join(render_position_tables(summary)), "html.parser")
    tables = parsed.find_all("table")
    assert len(tables) == 8
    assert all(table.find_parent("details") is None for table in tables[:6])
    assert all(table.find_parent("details") is not None for table in tables[6:])
    assert all(len(t.tbody.find_all("tr")) == 5 for t in tables[1:5])
    assert "pooled counts" not in parsed.get_text().lower()
    for label, t in zip(
        ("Fully agree", "Leaning agree", "Leaning disagree", "Fully disagree"), tables[1:5], strict=True
    ):
        assert label in t.caption.get_text()
    for parsed_table in tables:
        width = len(parsed_table.thead.find_all("th"))
        assert all(len(row.find_all(["td", "th"])) == width for row in parsed_table.tbody.find_all("tr"))


def test_c_headline_conditions_on_self_label_not_just_on_turn():
    from llm_committee.pivot.leaning_results_report import render_position_finding

    summary = {
        "C_by_self_label": [
            {"group": label, "choice": {"changed": changed, "valid": valid}}
            for label, changed, valid in [
                ("Fully agree", 2, 207),
                ("Leaning agree", 12, 138),
                ("Leaning disagree", 10, 127),
                ("Fully disagree", 1, 2),
            ]
        ]
    }
    parsed = BeautifulSoup(render_position_finding(summary), "html.parser")
    text = parsed.get_text()
    assert "Fully agree: 2/207 (1.0%)" in text
    assert "Leaning agree: 12/138 (8.7%)" in text
    assert "Fully disagree: 1/2 (50.0%)" in text
    assert "3/92" not in text
    assert "25/474" not in text
    assert parsed.find("a", href="#c-by-turn-label")
