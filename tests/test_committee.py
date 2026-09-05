"""Network-free unit tests for the current LLM Committee public API.

These tests exercise what actually exists today (the ``LLMCommitteeSync`` module, the
pure trajectory helpers, the Databricks LM factory, and the keyless web tools) without
making any network calls. LLM and Databricks interactions are either avoided entirely
(pure helpers) or stubbed via monkeypatching.
"""

from __future__ import annotations

import sys
import types

import pytest

from llm_committee import LLMCommitteeSync, TrajectoryAnalyzer
from llm_committee.committee_sync import AgreedLevel, TrajectoryNode


# --------------------------------------------------------------------------------------
# Imports / surface
# --------------------------------------------------------------------------------------
def test_public_symbols_import() -> None:
    """The current public classes import from the top-level package."""
    assert LLMCommitteeSync is not None
    assert TrajectoryAnalyzer is not None
    # They are dspy.Module subclasses.
    import dspy

    assert issubclass(LLMCommitteeSync, dspy.Module)
    assert issubclass(TrajectoryAnalyzer, dspy.Module)


# --------------------------------------------------------------------------------------
# Trajectory node helpers
# --------------------------------------------------------------------------------------
def _node(
    node_id: str,
    member_id: str,
    parent_id: str | None,
    agreed_level: AgreedLevel | None,
    *,
    iteration: int = 0,
    opinion_shift: int = 0,
) -> TrajectoryNode:
    """Build a TrajectoryNode with sensible defaults for testing."""
    return TrajectoryNode(
        node_id=node_id,
        member_id=member_id,
        iteration=iteration,
        parent_id=parent_id,
        children_ids=[],
        debate_opinion=f"opinion-{node_id}",
        reasoning=f"reasoning-{node_id}",
        agreed_level=agreed_level,
        opinion_shift=opinion_shift,
    )


def _committee() -> LLMCommitteeSync:
    """Construct a committee instance.

    Construction only instantiates DSPy modules (no LM call), so this is network-free.
    """
    return LLMCommitteeSync(num_members=3, max_iterations=1)


def test_select_ancestor_single_node_returns_none() -> None:
    """A trajectory with only the root node has no ancestors to debate with."""
    committee = _committee()
    trajectory = [_node("root", "member_0", None, None)]
    assert committee._select_ancestor_by_agreement(trajectory) is None


def test_select_ancestor_returns_valid_candidate() -> None:
    """Selection returns one of the trajectory's member ids and is weighted, not crashing."""
    committee = _committee()
    trajectory = [
        _node("n0", "member_0", None, None, iteration=0),
        _node("n1", "member_1", "n0", AgreedLevel.fully_disagreed, iteration=1),
        _node("n2", "member_2", "n1", AgreedLevel.partially_agreed, iteration=2),
    ]
    candidate_ids = {n.member_id for n in trajectory}
    # Run repeatedly: every result must be a real candidate (probabilistic selection).
    seen = {committee._select_ancestor_by_agreement(trajectory) for _ in range(50)}
    assert seen, "expected at least one selection"
    assert seen <= candidate_ids


def test_select_ancestor_excludes_member() -> None:
    """The excluded member is never selected, across many probabilistic draws."""
    committee = _committee()
    trajectory = [
        _node("n0", "member_0", None, None, iteration=0),
        _node("n1", "member_1", "n0", AgreedLevel.fully_disagreed, iteration=1),
        _node("n2", "member_2", "n1", AgreedLevel.fully_disagreed, iteration=2),
    ]
    results = {committee._select_ancestor_by_agreement(trajectory, exclude_member="member_2") for _ in range(100)}
    assert "member_2" not in results
    assert results <= {"member_0", "member_1"}


def test_select_ancestor_only_excluded_candidate_returns_none() -> None:
    """If excluding the only non-trivial candidate leaves nothing, return None."""
    committee = _committee()
    # Two-node trajectory; the only non-root member is member_1, which we exclude.
    trajectory = [
        _node("n0", "member_0", None, None, iteration=0),
        _node("n1", "member_1", "n0", AgreedLevel.fully_agreed, iteration=1),
    ]
    # member_0 (root) is still a candidate, so excluding member_1 leaves member_0.
    assert committee._select_ancestor_by_agreement(trajectory, exclude_member="member_1") == "member_0"
    # Excluding both members leaves no candidate.
    # Build a trajectory where every node belongs to the excluded member.
    trajectory_same = [
        _node("m0", "member_0", None, None, iteration=0),
        _node("m1", "member_0", "m0", AgreedLevel.fully_agreed, iteration=1),
    ]
    assert committee._select_ancestor_by_agreement(trajectory_same, exclude_member="member_0") is None


def test_compute_opinion_shifts() -> None:
    """Opinion-shift aggregation counts shifts, flags significant ones, and skips roots."""
    committee = _committee()
    # Root nodes (no shift) + debate nodes with assorted shift levels.
    graph: dict[str, TrajectoryNode] = {
        "r0": _node("r0", "member_0", None, None, iteration=0, opinion_shift=0),
        "r1": _node("r1", "member_1", None, None, iteration=0, opinion_shift=0),
        # member_0: a moderate shift (2) and a complete reversal (4).
        "a": _node("a", "member_0", "r0", AgreedLevel.partially_agreed, iteration=1, opinion_shift=2),
        "b": _node("b", "member_0", "a", AgreedLevel.fully_disagreed, iteration=2, opinion_shift=4),
        # member_1: no change (0) and a significant change (3).
        "c": _node("c", "member_1", "r1", AgreedLevel.fully_agreed, iteration=1, opinion_shift=0),
        "d": _node("d", "member_1", "c", AgreedLevel.partially_disagreed, iteration=2, opinion_shift=3),
    }

    out = committee._compute_opinion_shifts(graph)
    metrics = out["convergence_metrics"]

    # shift_counts indexes 0-4. Debate nodes only: one each of 2, 4, 0, 3.
    assert metrics["shift_counts"] == {0: 1, 1: 0, 2: 1, 3: 1, 4: 1}
    # total_shifts counts levels 1-4 (excludes 0): nodes a(2), b(4), d(3) -> 3.
    assert metrics["total_shifts"] == 3
    # significant_shifts counts levels 3-4: b(4), d(3) -> 2.
    assert metrics["significant_shifts"] == 2
    # Both members shifted at some point.
    assert metrics["members_who_shifted"] == ["member_0", "member_1"]

    # Per-member shift records exclude the (root) initial opinions.
    shifts = out["opinion_shifts"]
    assert {s["shift"] for s in shifts["member_0"]} == {2, 4}
    assert {s["shift"] for s in shifts["member_1"]} == {0, 3}


# --------------------------------------------------------------------------------------
# Databricks LM factory (no real auth / connection)
# --------------------------------------------------------------------------------------
@pytest.fixture()
def patched_connection(monkeypatch: pytest.MonkeyPatch):
    """Patch _resolve_connection so make_lm never touches Databricks."""
    from llm_committee import databricks_lm

    monkeypatch.setattr(
        databricks_lm,
        "_resolve_connection",
        lambda profile: ("http://x", "tok", "123"),
    )
    return databricks_lm


def test_make_lm_responses_for_gpt55pro(patched_connection) -> None:
    """gpt-5.5-pro is a Responses-API-only endpoint."""
    lm = patched_connection.make_lm("gpt-5.5-pro")
    assert lm.model == "openai/databricks-gpt-5-5-pro"
    assert lm.model_type == "responses"
    # Responses-only frontier models are forced to temperature=1.0.
    assert lm.kwargs.get("temperature") == 1.0


def test_make_lm_chat_for_opus(patched_connection) -> None:
    """opus-4.8 is a standard chat-completions endpoint."""
    lm = patched_connection.make_lm("opus-4.8")
    assert lm.model == "openai/databricks-claude-opus-4-8"
    assert lm.model_type == "chat"
    # The workspace-id header from the resolved connection is wired through.
    assert lm.kwargs.get("extra_headers", {}).get("X-Databricks-Workspace-Id") == "123"


# --------------------------------------------------------------------------------------
# Web tools never raise (return error strings instead)
# --------------------------------------------------------------------------------------
def test_web_search_returns_string_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """web_search swallows exceptions and returns an error string, never raising."""
    from llm_committee import tools

    boom = types.ModuleType("ddgs")

    class _DDGS:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("forced search failure")

    boom.DDGS = _DDGS
    monkeypatch.setitem(sys.modules, "ddgs", boom)

    result = tools.web_search("anything")
    assert isinstance(result, str)
    assert result.startswith("[web_search error:")


def test_fetch_page_returns_string_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_page swallows exceptions and returns an error string, never raising."""
    from llm_committee import tools

    req = types.ModuleType("requests")

    def _get(*args, **kwargs):
        raise RuntimeError("forced network failure")

    req.get = _get
    monkeypatch.setitem(sys.modules, "requests", req)

    result = tools.fetch_page("http://example.invalid")
    assert isinstance(result, str)
    assert result.startswith("[fetch_page error:")


def test_fetch_page_bad_input_does_not_raise() -> None:
    """A clearly malformed URL returns an error string rather than raising."""
    from llm_committee.tools import fetch_page

    result = fetch_page("not-a-real-url-at-all")
    assert isinstance(result, str)
    assert result.startswith("[fetch_page error:")
