"""Pair/order decoding and question-clustered old/new comparison."""

import copy

import pytest

from llm_committee.pivot.models import digest
from scripts.report_process_neutral_judging import MODELS, analyze, change_pp, human_comparison, percent, summarize_rows


def test_model_summary_pools_assignments_within_questions_and_tracks_flips():
    rows = []
    for model in MODELS:
        for q, arm, old, new in [
            ("q1", "original", [False, False], [True, True]),
            ("q1", "alternate", [True, True], [True, True]),
            ("q2", "original", [False, False], [False, False]),
            ("q2", "alternate", [False, False], [False, False]),
        ]:
            rows.append(
                {
                    "question_id": q,
                    "assignment": arm,
                    "model": model,
                    "old_votes": old,
                    "new_votes": new,
                    "old_score": sum(old) / 2,
                    "new_score": sum(new) / 2,
                    "delta": (sum(new) - sum(old)) / 2,
                }
            )
    for s in summarize_rows(rows):
        assert s["old_score"]["mean"] == 0.25 and s["new_score"]["mean"] == 0.5
        assert s["delta"]["mean"] == 0.25 and s["delta"]["ci"] == [0.0, 0.5]
        assert s["old_score"]["questions"] == 2 and s["old_score"]["n"] == 4
        assert s["changed_decisions"] == 2 and s["changed_pair_verdicts"] == 1
        assert s["baseline_to_debate"] == 2 and s["debate_to_baseline"] == 0
        assert s["decisions"] == 8 and s["pairs"] == 4


def test_human_alignment_maps_answer_identity_across_both_orders():
    annotation = {
        "version": "blind-answer-review-v1",
        "dataset_id": "set",
        "responses": [{"item_id": "item", "preference": "A", "rating_language": "zh"}],
    }
    key = {
        "dataset_id": "set",
        "items": [
            {
                "item_id": "item",
                "question_id": "q",
                "assignment": "alternate",
                "debate_side": "A",
                "baseline_sha256": digest("base"),
                "debate_sha256": digest("debated"),
            }
        ],
    }
    questions = {
        "q": {
            "question": {"text": "Question"},
            "E2": [{"assignment": "alternate", "baseline": {"answer": "base"}, "debated": {"answer": "debated"}}],
        }
    }
    rows = [
        {
            "model": model,
            "question_id": "q",
            "assignment": "alternate",
            "orders": [
                {
                    "order": 0,
                    "target_side": "left",
                    "old": {"preference": "left", "evidence": "old"},
                    "new": {"preference": "right", "evidence": "new"},
                },
                {
                    "order": 1,
                    "target_side": "right",
                    "old": {"preference": "right", "evidence": "old"},
                    "new": {"preference": "right", "evidence": "new"},
                },
            ],
        }
        for model in MODELS
    ]
    result = human_comparison(annotation, key, rows, questions, {"item": {"question": "问题"}})
    assert result["human_debate_wins"] == 1 and result["excluded_nonbinary"] == 0
    case = result["cases"][0]
    assert case["answers"] == {"A": "debated", "B": "base"}
    for s in result["summary"]:
        assert s["binary_ratings"] == 1
        assert s["old"]["matched_order_agreement"] == s["old"]["both_orders_agree"] == 1
        assert s["new"]["matched_order_agreement"] == 0 and s["new"]["order_inconsistent"] == 1
    for orders in case["judges"].values():
        assert [o["old"]["choice"] for o in orders] == ["A", "A"]
        assert [o["new"]["choice"] for o in orders] == ["B", "A"]
    tied = copy.deepcopy(annotation)
    tied["responses"][0]["preference"] = "tie"
    result = human_comparison(tied, key, rows, questions, {"item": {}})
    assert result["excluded_nonbinary"] == 1 and all(s["binary_ratings"] == 0 for s in result["summary"])
    for invalid in ({**annotation, "dataset_id": "wrong"}, {**annotation, "responses": annotation["responses"] * 2}):
        with pytest.raises(ValueError):
            human_comparison(invalid, key, rows, questions, {"item": {}})


def test_incomplete_judging_cannot_be_published(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.report_process_neutral_judging.execution_report", lambda output: {"status": "incomplete"}
    )
    with pytest.raises(ValueError, match="incomplete"):
        analyze(tmp_path)


def test_display_distinguishes_percentage_rates_and_paired_percentage_point_changes():
    estimate = {"mean": 0.04, "ci": [-0.01, 0.08]}
    assert change_pp(estimate) == "+4.0 [-1.0, +8.0] pp"
    assert percent(estimate) == "4.0% [-1.0, 8.0]"
    assert percent({"mean": None}) == "—"
    assert percent({"mean": 0.5, "ci": None}) == "50.0%"
