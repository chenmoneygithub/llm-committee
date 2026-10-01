"""A/B share the new rubric; old data must not be relabeled or silently reused."""

import json
from copy import deepcopy

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.agreement import (
    AGREEMENT,
    AGREEMENT_DEFINITIONS,
    AGREEMENT_RUBRIC_TEXT,
    LEGACY_AGREEMENT,
    LEGACY_PROMPT_VERSION,
    agreement_manifest,
)
from llm_committee.pivot.continuation import load_checkpoint
from llm_committee.pivot.models import SCREEN_JUDGES, PilotConfig, Question, Screen
from llm_committee.pivot.runner import manifest_for


@pytest.fixture
def question():
    return Question(
        "fixture", "Should parks be free?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "synthetic-only", "0" * 64)
    )


def test_all_four_labels_have_shared_verbatim_definitions(question):
    assert AGREEMENT == ("fully_agree", "leaning_agree", "leaning_disagree", "fully_disagree")
    assert set(AGREEMENT_DEFINITIONS) == set(AGREEMENT)
    judge_system = prompts.b_messages(question, "Peer contribution", "Reply")[0].text
    assert AGREEMENT_RUBRIC_TEXT in prompts.REPLY_RULE
    assert AGREEMENT_RUBRIC_TEXT in judge_system
    for label, definition in AGREEMENT_DEFINITIONS.items():
        assert definition and f"{label}: {definition}" in judge_system
    assert "overall supports or opposes" in judge_system
    assert "not uncertainty" in judge_system
    assert prompts.PROMPT_VERSION != LEGACY_PROMPT_VERSION


@pytest.mark.parametrize("label", AGREEMENT)
def test_new_labels_are_accepted_in_both_output_schemas(label):
    assert (
        prompts.parse_json(json.dumps({"reply": "Text", "agreement": label}), prompts.REPLY_SCHEMA)["agreement"]
        == label
    )
    assert prompts.parse_json(json.dumps({"label": label, "evidence": "Text"}), prompts.B_SCHEMA)["label"] == label


@pytest.mark.parametrize("label", LEGACY_AGREEMENT)
def test_legacy_labels_are_not_mapped_to_new_ones(label):
    with pytest.raises(ValueError):
        prompts.parse_json(json.dumps({"reply": "Text", "agreement": label}), prompts.REPLY_SCHEMA)
    with pytest.raises(ValueError):
        prompts.parse_json(json.dumps({"label": label, "evidence": "Text"}), prompts.B_SCHEMA)


def test_nonjudgments_are_retained_without_forcing_a_leaning_label():
    assert prompts.parse_json('{"reply":"No position", "agreement":null}', prompts.REPLY_SCHEMA)["agreement"] is None
    for label in ("no_position", "unjudgeable"):
        assert (
            prompts.parse_json(json.dumps({"label": label, "evidence": "Cannot classify"}), prompts.B_SCHEMA)["label"]
            == label
        )


def test_manifest_freezes_definitions_and_does_not_mutate_them(question):
    manifest = manifest_for((question,), PilotConfig(judge_model="gemini-3.8-flash"), mock=True)
    assert manifest["agreement_rubric"] == agreement_manifest()
    manifest["agreement_rubric"]["definitions"]["leaning_agree"] = "tampered"
    assert AGREEMENT_DEFINITIONS["leaning_agree"] != "tampered"


@pytest.mark.parametrize("change", ["old_version", "changed_definition"])
def test_old_or_changed_rubric_cannot_be_imported_as_a_continuation(tmp_path, question, change):
    current = manifest_for((question,), PilotConfig(judge_model="gemini-3.8-flash"), mock=True)
    archived = deepcopy(current)
    field = "prompt_version" if change == "old_version" else "agreement_rubric"
    if change == "old_version":
        archived["prompt_version"] = LEGACY_PROMPT_VERSION
        archived.pop("agreement_rubric")
    else:
        archived["agreement_rubric"]["definitions"]["leaning_agree"] = "different rule"
    (tmp_path / "manifest.json").write_text(json.dumps(archived))
    # Rejected before even opening a journal, constructing a client, or reading credentials.
    with pytest.raises(ValueError, match=field):
        load_checkpoint(tmp_path, current)


def test_legacy_report_remains_bound_to_legacy_vocabulary():
    from llm_committee.pivot.results_report import AGREEMENT as REPORT_LABELS

    assert REPORT_LABELS == LEGACY_AGREEMENT
    assert REPORT_LABELS != AGREEMENT
