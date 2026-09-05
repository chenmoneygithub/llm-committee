"""Toolkit tests: the shipped surface (engine config + evaluator layers) works as documented."""
from __future__ import annotations

import pytest

from llm_committee.committee_sync import DEBATE_STANCE_INSTRUCTIONS
from llm_committee.evaluators import layer_a


def test_predefined_tones_exist() -> None:
    assert set(DEBATE_STANCE_INSTRUCTIONS) == {"neutral", "friendly", "hostile"}
    assert DEBATE_STANCE_INSTRUCTIONS["neutral"] == ""
    for tone in ("friendly", "hostile"):
        assert DEBATE_STANCE_INSTRUCTIONS[tone].strip()


def _toy_graph() -> dict:
    return {
        "root-1": {"node_id": "root-1", "member_id": "m1", "iteration": 0},
        "turn-1": {
            "node_id": "turn-1",
            "member_id": "m1",
            "iteration": 1,
            "agreed_level": "fully_disagreed",
            "agreed_level_explicit": True,
            "opinion_shift": 1,
        },
        "turn-2": {
            "node_id": "turn-2",
            "member_id": "m2",
            "iteration": 1,
            "agreed_level": "fully_agreed",
            "agreed_level_explicit": True,
            "opinion_shift": 0,
        },
        "turn-3-defaulted": {
            "node_id": "turn-3-defaulted",
            "member_id": "m3",
            "iteration": 1,
            "agreed_level": "partially_agreed",
            "agreed_level_explicit": False,
        },
    }


def test_layer_a_reads_explicit_labels_only_by_default() -> None:
    rows = layer_a.turn_labels(_toy_graph())
    assert [r["node_id"] for r in rows] == ["turn-1", "turn-2"]
    assert rows[0]["severity"] == 3

    summary = layer_a.summarize(_toy_graph())
    assert summary["n_turns"] == 2
    assert summary["fully_agreed_rate"] == 0.5
    assert summary["dissent_rate"] == 0.5

    # Schema-defaulted labels are visible only on request, flagged as such.
    all_rows = layer_a.turn_labels(_toy_graph(), explicit_only=False)
    assert len(all_rows) == 3
    assert all_rows[-1]["agreed_level_explicit"] is False


def test_layer_c_requires_registered_task_spec() -> None:
    layer_c = pytest.importorskip("llm_committee.evaluators.layer_c")
    with pytest.raises(KeyError):
        layer_c.get_benchmark("unregistered-task")
    layer_c.register_task_spec("my-task", "Debate the question collaboratively", "")
    spec = layer_c.get_benchmark("my-task")
    assert spec.committee_task == "Debate the question collaboratively"


def test_stats_sign_flip_is_deterministic() -> None:
    stats = pytest.importorskip("llm_committee.evaluators.stats")
    a = stats.sign_flip_test([-0.5, -0.25, 0.0, -0.75], seed=7)
    b = stats.sign_flip_test([-0.5, -0.25, 0.0, -0.75], seed=7)
    assert a == b


def test_layer_b_and_d_importable() -> None:
    pytest.importorskip("dspy")
    from llm_committee.evaluators import layer_b

    assert hasattr(layer_b, "judge_pushback")

    from llm_committee.evaluators import layer_d

    assert hasattr(layer_d, "ENDPOINTS_JSON")
