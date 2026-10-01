"""Zero-fill is a declared truncation approximation, not fabricated log probabilities."""

import math
from copy import deepcopy

import pytest

from llm_committee.pivot.probabilities import TOPK_ZERO_FILL_POLICY, distribution, uses_topk_zero_fill


@pytest.fixture
def readout():
    return {
        "candidate_ids": {"A": 65, "B": 66, "C": 67},
        "candidate_logprobs": {"A": math.log(0.3), "B": math.log(0.6)},
        "source": "sample_topk",
        "prefix_sha256": "a" * 64,
        "read_temperature": 1.0,
        "sampling_temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
    }


def test_zero_fill_normalizes_only_observed_candidates_and_keeps_raw_values(readout):
    before = deepcopy(readout)
    result = distribution(readout, tuple("ABC"), missing_as_zero=True)
    assert result["probabilities"] == pytest.approx({"A": 1 / 3, "B": 2 / 3, "C": 0})
    assert result["candidate_mass"] == pytest.approx(0.9)
    assert result["missing_candidate_labels"] == ["C"]
    assert result["zero_fill_applied"] is True
    assert result["candidate_mass_is_lower_bound"] is True
    assert result["candidate_logprobs"] == before["candidate_logprobs"]
    assert readout == before


def test_complete_readout_retains_all_values_and_reports_no_zero_fill(readout):
    readout["candidate_logprobs"]["C"] = math.log(0.1)
    result = distribution(readout, tuple("ABC"), missing_as_zero=True)
    legacy = distribution(readout, tuple("ABC"))
    assert result["probabilities"] == legacy["probabilities"]
    assert result["missing_candidate_labels"] == []
    assert result["zero_fill_applied"] is False


def test_a_zero_logprob_is_probability_one_not_an_absent_candidate(readout):
    readout["candidate_logprobs"] = {"A": 0.0}
    result = distribution(readout, tuple("ABC"), missing_as_zero=True)
    assert result["probabilities"] == {"A": 1.0, "B": 0.0, "C": 0.0}


def test_very_small_observed_probs_still_normalize_stably(readout):
    readout["candidate_logprobs"] = {"A": -1000, "B": -1000 - math.log(2)}
    result = distribution(readout, tuple("ABC"), missing_as_zero=True)
    assert result["probabilities"] == pytest.approx({"A": 2 / 3, "B": 1 / 3, "C": 0})


@pytest.mark.parametrize("bad", ["empty", "nan", "infinity", "positive", "extra", "source", "mass"])
def test_zero_fill_does_not_bypass_other_validation(readout, bad):
    if bad == "empty":
        readout["candidate_logprobs"] = {}
    elif bad == "source":
        readout["source"] = "target_prompt_logprobs"
    elif bad == "extra":
        readout["candidate_logprobs"]["Z"] = -3
    elif bad == "mass":
        readout["candidate_logprobs"] = {"A": math.log(0.9), "B": math.log(0.9)}
    else:
        readout["candidate_logprobs"]["B"] = {"nan": math.nan, "infinity": -math.inf, "positive": 0.1}[bad]
    with pytest.raises(ValueError):
        distribution(readout, tuple("ABC"), missing_as_zero=True)


def test_manifest_policy_is_explicit_and_versioned():
    assert not uses_topk_zero_fill({})
    assert uses_topk_zero_fill({"probability_readout": dict(TOPK_ZERO_FILL_POLICY)})
    with pytest.raises(ValueError, match="Unknown"):
        uses_topk_zero_fill({"probability_readout": {"name": "unrecognized"}})
