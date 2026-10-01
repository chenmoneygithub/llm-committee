"""Report estimands, units and incomplete cases remain explicit."""

import pytest

from llm_committee.pivot.forced_feedback_report import c_table, d_table, paired_summary, question_mean, rate


def test_question_weighting_does_not_pool_all_cases():
    rows = [{"question_id": "one", "v": 1.0}] + [{"question_id": "two", "v": 0.0}] * 3
    assert question_mean(rows, "v") == (0.5, 4, 2)


def test_paired_summary_requires_matched_cases_not_unpaired_arm_means():
    rows = [
        {"id": "a", "question_id": "one", "arm": "natural", "v": 0, "recipient": 1},
        {"id": "a", "question_id": "one", "arm": "forced", "v": 1, "recipient": 1},
        {"id": "b", "question_id": "one", "arm": "natural", "v": 0, "recipient": 1},
        {"id": "b", "question_id": "one", "arm": "forced", "v": 0, "recipient": 1},
        {"id": "c", "question_id": "two", "arm": "natural", "v": 1, "recipient": 2},
        {"id": "c", "question_id": "two", "arm": "forced", "v": 0, "recipient": 2},
        {"id": "d", "question_id": "two", "arm": "natural", "v": 0, "recipient": 2},
        {"id": "d", "question_id": "two", "arm": "forced", "v": None, "recipient": 2},
    ]
    result = paired_summary(rows, "v")
    assert result["estimate"] == -0.25  # Mean of question means +0.5 and -1.0.
    assert result["questions"] == 2 and result["pairs"] == 3
    assert paired_summary(rows, "v", recipient=1)["estimate"] == 0.5
    assert paired_summary(rows, "v", recipient=0)["estimate"] is None
    assert paired_summary(rows, "v") == result


def test_c_table_keeps_missing_and_unjudgeable_separate():
    rows = [
        {"arm": "forced", "choice_changed": True, "text_label": "adjusted"},
        {"arm": "forced", "choice_changed": False, "text_label": "unchanged"},
        {"arm": "forced", "choice_changed": None, "text_label": "unjudgeable"},
        {"arm": "forced", "choice_changed": None, "text_label": None},
    ]
    text = c_table([{**row, "self_label": "leaning_agree"} for row in rows], ("self_label", "arm"))
    assert "1/2 (50.0%)" in text
    assert "1/4" not in text
    assert "Text unjudgeable" in text and "Text missing" in text
    assert rate(0, 0) == "— (0 valid)"


@pytest.mark.parametrize("kind", ["text", "choice"])
def test_d_means_use_identical_paired_support(kind):
    rows = [
        {
            "question_id": "one",
            "recipient": 1,
            "D_control": 7,
            "D_argument": 6.5,
            "D_text_delta": -0.5,
            "choice_p_before": 70,
            "choice_p_after": 60,
            "D_choice_pp": -10,
        },
        {
            "question_id": "one",
            "recipient": 1,
            "D_control": 1,
            "D_argument": None,
            "D_text_delta": None,
            "choice_p_before": 1,
            "choice_p_after": None,
            "D_choice_pp": None,
        },
    ]
    text = d_table([{**row, "self_label": "leaning_agree"} for row in rows], ("self_label", "recipient"), kind=kind)
    assert "1/2" in text
    if kind == "text":
        assert "7.000" in text and "6.500" in text and "-0.500" in text
    else:
        assert "70.000" in text and "60.000" in text and "-10.000" in text


@pytest.mark.parametrize("renderer", [c_table, d_table])
@pytest.mark.parametrize("dimensions", [("arm",), ("receiver_T", "arm"), ("recipient", "arm")])
def test_c_d_tables_refuse_to_pool_across_self_labels(renderer, dimensions):
    with pytest.raises(ValueError, match="must retain the recipient self-label"):
        renderer([], dimensions)
