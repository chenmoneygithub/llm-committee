"""Scaling appends disjoint questions without changing existing measurement slots."""

from collections import Counter
from pathlib import Path

import pytest

from llm_committee.pivot.models import TONES
from llm_committee.pivot.quality_study import plan_questions as quality_plans
from llm_committee.pivot.stateful_run import run
from llm_committee.pivot.stateful_study import prepare as prepare_dyadic
from llm_committee.pivot.triadic_study import prepare as prepare_triadic
from llm_committee.pivot.turn_tone import plan_questions
from llm_committee.pivot.turn_tone_fork import alternate_plans
from scripts.scale_public_history import BASE_DYADIC, prepare, read


def test_original_twenty_schedules_unchanged():
    old = read(BASE_DYADIC)
    plans = plan_questions([p["question"]["id"] for p in old["plans"]])
    alternate = {p["question_id"]: p for p in alternate_plans(plans)}
    for old_item, new_item in zip(old["plans"], plans, strict=True):
        qid = new_item["question_id"]
        assert old_item["question"]["id"] == qid
        assert old_item["original_tones"] == new_item["tone_schedule"]
        assert old_item["alternate_tones"] == alternate[qid]["tone_schedule"]
        assert set(old_item["sampled_node_ids"]) == {e["node_id"] for e in new_item["events"]}


def test_quality_twenty_plan_unchanged():
    root = Path(__file__).resolve().parents[1]
    old = read(root / "runs/public-history-quality-20-20260927/live/manifest.json")
    assert quality_plans([p["question_id"] for p in old["plans"]]) == old["plans"]


@pytest.fixture(scope="module")
def extension(tmp_path_factory):
    output = tmp_path_factory.mktemp("extension")
    selection, specs = prepare(output)
    return output, selection, specs


def test_selection_is_shared_disjoint_and_not_redrawn(extension):
    output, selection, _ = extension
    assert len(selection["new_question_ids"]) == 30
    assert len(selection["existing_question_ids"]) == 20
    assert not set(selection["new_question_ids"]) & set(selection["existing_question_ids"])
    assert len(selection["unused_question_ids"]) == 10
    assert prepare(output)[0] == selection


@pytest.mark.parametrize("roster", ["mixed_family", "same_family", "same_model"])
def test_scaled_counts_and_matching_layout(extension, roster):
    output, selection, _ = extension
    m, c = prepare_dyadic(output / "plans/dyadic-plan.json", roster=roster, mock=True, public_history=True)
    assert set(c) == set(selection["new_question_ids"])
    assert m["design"]["independent_question_count"] == 30
    counts = m["planned_counts"]
    assert counts["questions"] == 30 and counts["formal_replies"] == 540
    assert counts["branch_paths"] == 180 and counts["sampled_original_events"] == 240
    assert m["design"]["D_enabled"] == (roster == "mixed_family")
    tones = Counter(
        t
        for p in m["plans"]
        for arm, branch in p["arms"].items()
        for node, t in branch["tone_schedule"].items()
        if arm == "original" or int(node[-1]) > 2
    )
    assert tones == dict.fromkeys(TONES, 180)
    tri, tc = prepare_triadic(output / "plans/triadic-plan.json", roster=roster, mock=True)
    assert tc == c and tri["planned_counts"]["questions"] == 30
    assert tri["planned_counts"]["branch_paths"] == 360
    assert tri["planned_counts"]["primary_quality_comparisons"] == 60
    assert tri["planned_counts"]["D_calls"] == tri["planned_counts"]["local_C_calls"] == 0


def test_analysis_view_cannot_launch(extension, tmp_path):
    output, _, _ = extension
    m, c = prepare_dyadic(output / "plans/dyadic-plan.json", mock=True, public_history=True)
    m["analysis_only"] = True
    with pytest.raises(ValueError, match="cannot dispatch"):
        run(m, c, tmp_path / "should-not-exist")
    assert not (tmp_path / "should-not-exist").exists()
