"""Checkpoint reuse must not resample, hide failures, or double-charge old calls."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from llm_committee.pivot.continuation import load_checkpoint
from llm_committee.pivot.models import SCREEN_JUDGES, Completion, PilotConfig, Question, Screen
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.runner import expected_counts, manifest_for, run_pilot
from llm_committee.pivot.storage import Journal, ProbabilityReadBlocked, RunBlocked


class MismatchProvider(MockProvider):
    def generate(self, request):
        if request.scoring:
            self.calls.append(request)
            scores = {k: v - 0.5 for k, v in request.scoring["fixture_scores"].items()}
            return Completion(json.dumps({**request.scoring, "candidate_logprobs": scores}), 100, 1)
        result = super().generate(request)
        if result.readout:
            full = result.readout["candidate_logprobs"]
            return replace(
                result, readout={**result.readout, "candidate_logprobs": {"A": full["A"]}, "fixture_scores": full}
            )
        return result


@pytest.fixture
def setup():
    q = Question("test", "Should parks be free?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
    config = PilotConfig(judge_model="gemini-3.8-flash")
    manifest = manifest_for((q,), config, mock=True)
    manifest.pop("probability_readout")  # Exercise archived exact-score continuation semantics.
    return (q,), config, manifest


@pytest.fixture
def stopped(tmp_path, setup):
    questions, config, manifest = setup
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    journal = Journal(source / "requests.sqlite3", manifest, 100)
    try:
        with pytest.raises(ProbabilityReadBlocked):
            run_pilot(questions, config, manifest, journal, MismatchProvider())
        count = len(journal.audit()["calls"])
        with pytest.raises(ProbabilityReadBlocked):
            run_pilot(questions, config, manifest, journal, MismatchProvider())
        assert len(journal.audit()["calls"]) == count
        charge = journal.charged_usd
    finally:
        journal.close()
    return source, charge


def test_continuation_preserves_failed_scores_and_finishes_independent_work(tmp_path, setup, stopped):
    questions, config, manifest = setup
    source, prior_cost = stopped
    manifest["execution"]["record_probability_failures"] = True
    descriptor, rows = load_checkpoint(source, manifest)
    manifest["continuation"] = descriptor
    journal = Journal(tmp_path / "new.sqlite3", manifest, 100)
    provider = MismatchProvider()
    try:
        journal.import_checkpoint(descriptor, rows)
        assert journal.charged_usd == 0
        result = run_pilot(questions, config, manifest, journal, provider)
        assert result["status"] == "completed_with_measurement_failures"
        assert result["prior_continuation_cost_usd"] == prior_cost
        assert not ({r["key"] for r in rows} & {r.key for r in provider.calls})
        question = result["questions"][0]
        assert question["D"]["status"] == "incomplete"
        assert question["D"]["choice_readings"] == {}
        assert (
            len(result["measurement_failures"])
            == expected_counts(manifest)["D_choice_readings"] + expected_counts(manifest)["D_calls"]
        )
        for event in question["sampled_events"]:
            assert event["B"] and event["C_text_adjacent"]
            assert event["D_choice_adjacent_pp"] is None and event["D_choice_initial_pp"] is None
            if event["D_text"]:
                assert event["D_text"]["mean_own_agreement_argument_minus_control"] is None
                assert "probabilities" not in event["D_text"]["argument"]
        assert len(question["E"]["preferences"]) == 3
        assert sum(r.purpose == "judge_b" for r in provider.calls) == expected_counts(manifest)["B_judgments"]
        n = len(provider.calls)
        journal.import_checkpoint(descriptor, rows)
        assert run_pilot(questions, config, manifest, journal, provider) == result
        assert len(provider.calls) == n
        audit = journal.audit()
        assert sum(c["reused_without_new_call"] for c in audit["calls"]) == len(rows)
        assert any(c["status"] == "invalid" and c["reused_without_new_call"] for c in audit["calls"])
    finally:
        journal.close()
    assert load_checkpoint(source, manifest)[0] == descriptor  # Source was not edited.


def test_zero_fill_reanalysis_is_explicit_offline_and_does_not_repair_old_scores(tmp_path, setup):
    from llm_committee.pivot.reanalysis import reanalyze_topk

    questions, config, manifest = setup
    manifest["execution"]["record_probability_failures"] = True
    source = tmp_path / "legacy-completed"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    journal = Journal(source / "requests.sqlite3", manifest, 100)
    try:
        original = run_pilot(questions, config, manifest, journal, MismatchProvider())
        source_audit = journal.audit()
        assert original["status"] == "completed_with_measurement_failures"
    finally:
        journal.close()
    new_manifest = manifest_for(questions, config, mock=True)
    with pytest.raises(ValueError, match="probability_readout"):
        load_checkpoint(source, new_manifest)
    revised = reanalyze_topk(source, tmp_path / "zero-fill", token_counter=MockProvider().token_count)
    assert revised["status"] == "completed"
    assert revised["offline_reanalysis"]["api_calls"] == 0
    assert revised["charged_or_reserved_usd"] == 0
    assert revised["offline_reanalysis"]["source_invalid_records"]
    for key in ("A", "initial_answers", "formal_replies", "C_readings", "C_text_pairs", "positions", "E"):
        assert revised["questions"][0][key] == original["questions"][0][key]
    for before, after in zip(
        original["questions"][0]["sampled_events"], revised["questions"][0]["sampled_events"], strict=True
    ):
        assert {k: v for k, v in before.items() if not k.startswith("D_")} == {
            k: v for k, v in after.items() if not k.startswith("D_")
        }
    for d in revised["questions"][0]["D"]["choice_readings"].values():
        assert d["probabilities"] == {"A": 1.0, "B": 0.0}
    journal = Journal(source / "requests.sqlite3", manifest, 100)
    try:
        assert journal.audit() == source_audit
    finally:
        journal.close()


def test_offline_reanalysis_stops_on_unsaved_request_without_reserving_or_dispatching(tmp_path, setup, stopped):
    import sqlite3

    from llm_committee.pivot.reanalysis import reanalyze_topk

    output = tmp_path / "incomplete-reanalysis"
    with pytest.raises(RunBlocked, match="dispatch is disabled"):
        reanalyze_topk(stopped[0], output, token_counter=MockProvider().token_count)
    db = sqlite3.connect(output / "requests.sqlite3")
    try:
        assert db.execute("SELECT SUM(charge) FROM calls").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize("field", ["config", "prompt_version", "plans", "runtime_packages", "questions"])
def test_continuation_rejects_protocol_changes(setup, stopped, field):
    manifest = deepcopy(setup[2])
    manifest[field] = "CHANGED"
    with pytest.raises(ValueError, match=field):
        load_checkpoint(stopped[0], manifest)


def test_imported_key_still_requires_identical_request(tmp_path, setup, stopped):
    from llm_committee.pivot.models import Request

    manifest = setup[2]
    descriptor, rows = load_checkpoint(stopped[0], manifest)
    journal = Journal(tmp_path / "new.sqlite3", manifest, 100)
    try:
        journal.import_checkpoint(descriptor, rows)
        provider = MockProvider()
        with pytest.raises(RunBlocked, match="Request changed"):
            journal.call(Request(rows[0]["key"], "initial", "gpt-5.6-terra", (), "medium", 42), provider, json.loads)
        assert not provider.calls
        with pytest.raises(ValueError, match="checkpoint changed"):
            journal.import_checkpoint({**descriptor, "reused_calls": 999}, rows)
    finally:
        journal.close()


@pytest.mark.parametrize("state", ["pending", "uncertain", "billing_unknown", "received", "invalid"])
def test_continuation_does_not_bypass_unrelated_failures(setup, stopped, state):
    import sqlite3

    source, _ = stopped
    db = sqlite3.connect(source / "requests.sqlite3")
    try:
        db.execute("UPDATE calls SET status=?,error='Different failure' WHERE status='invalid'", (state,))
        db.commit()
    finally:
        db.close()
    with pytest.raises(ValueError, match="checkpoint status"):
        load_checkpoint(source, setup[2])


def test_diagnostic_policy_does_not_swallow_format_failure(tmp_path, setup):
    class Malformed(MockProvider):
        def generate(self, request):
            if request.purpose == "d_text":
                self.calls.append(request)
                return Completion("NOT A RATING", 100, 1)
            return super().generate(request)

    questions, config, manifest = setup
    manifest["execution"]["record_probability_failures"] = True
    journal = Journal(tmp_path / "format.sqlite3", manifest, 100)
    try:
        with pytest.raises(RunBlocked, match="Invalid response"):
            run_pilot(questions, config, manifest, journal, Malformed())
    finally:
        journal.close()


def test_received_gemini_usage_can_be_reconciled_offline_without_new_call(tmp_path, setup):
    class OldAccounting(MockProvider):
        def generate(self, request):
            result = super().generate(request)
            if request.purpose != "judge_b":
                return result
            response = {
                "choices": [{"message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 603, "completion_tokens": 77, "reasoning_tokens": 95, "total_tokens": 775},
            }
            return Completion(
                result.text,
                603,
                77,
                reasoning_tokens=95,
                raw={"provider": "databricks", "http_status": 200, "response": response},
            )

    questions, config, manifest = setup
    source = tmp_path / "old-accounting"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    journal = Journal(source / "requests.sqlite3", manifest, 100)
    try:
        with pytest.raises(RunBlocked, match="reconcile"):
            run_pilot(questions, config, manifest, journal, OldAccounting())
        before = journal.audit()
        with pytest.raises(ValueError, match="checkpoint status"):
            load_checkpoint(source, manifest)
        descriptor, rows = load_checkpoint(source, manifest, reconcile_databricks_usage=True)
        (correction,) = descriptor["offline_usage_reconciliations"]
        fixed = next(r for r in rows if r["key"] == correction["key"])
        assert fixed["status"] == "completed"
        assert json.loads(fixed["response"])["output_tokens"] == 172
        assert fixed["charge"] == pytest.approx(0.001206975)
        assert descriptor["prior_charged_or_reserved_usd"] == pytest.approx(
            before["charged_or_reserved_usd"] - correction["source_reserved_usd"] + fixed["charge"]
        )
        assert journal.audit() == before  # Source journal remains exactly as it was.
        new = Journal(tmp_path / "new-accounting.sqlite3", manifest, 100)
        try:
            new.import_checkpoint(descriptor, rows)
            provider = MockProvider()
            result = run_pilot(questions, config, manifest, new, provider)
            assert result["status"] == "completed"
            assert not ({r.key for r in provider.calls} & {r["key"] for r in rows})
        finally:
            new.close()
    finally:
        journal.close()


def test_identical_json_duplicates_reconcile_offline_with_raw_and_cost_preserved(tmp_path, setup):
    from llm_committee.pivot import prompts
    from llm_committee.pivot.models import Request, digest

    _, _, manifest = setup
    source = tmp_path / "duplicates"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    request = Request("test/debate", "debate", "gpt-5.6-terra", (), "medium", 100, prompts.REPLY_SCHEMA)

    class DuplicateProvider(MockProvider):
        def generate(self, request):
            self.calls.append(request)
            return Completion('{"agreement":"leaning_agree","reply":"Saved reply",'
                              '"agreement":"leaning_agree"}', 100, 30)

    def old_parse(text):
        raise ValueError("Duplicate JSON field: agreement")

    journal = Journal(source / "requests.sqlite3", manifest, 100)
    try:
        with pytest.raises(RunBlocked, match="Invalid response"):
            journal.call(request, DuplicateProvider(), old_parse)
        before = journal.audit()
        with pytest.raises(ValueError, match="checkpoint status"):
            load_checkpoint(source, manifest)
        descriptor, rows = load_checkpoint(source, manifest, reconcile_identical_json_duplicates=True)
        (correction,) = descriptor["offline_json_reconciliations"]
        (fixed,) = rows
        assert correction["identical_duplicate_fields"] == ["agreement"]
        assert correction["source_response_sha256"] == digest(before["calls"][0]["response"])
        assert json.loads(fixed["response"]) == before["calls"][0]["response"]
        assert fixed["charge"] == descriptor["prior_charged_or_reserved_usd"] == before["charged_or_reserved_usd"]
        assert journal.audit() == before
        new = Journal(tmp_path / "recovered.sqlite3", manifest, 100)
        try:
            new.import_checkpoint(descriptor, rows)
            provider = DuplicateProvider()
            parsed = new.call(request, provider, lambda text: prompts.parse_json(text, prompts.REPLY_SCHEMA))
            assert parsed == {"agreement": "leaning_agree", "reply": "Saved reply"}
            assert not provider.calls and new.charged_usd == 0
        finally:
            new.close()
    finally:
        journal.close()


@pytest.mark.parametrize("text", [
    '{"agreement":"leaning_agree","reply":"x","agreement":"fully_disagree"}',
    '{"agreement":"leaning_agree","agreement":"leaning_agree"}',
    '{"agreement":"leaning_agree","reply":"x"}',
    '{"agreement":"invalid","reply":"x","agreement":"invalid"}',
    '{"agreement":"leaning_agree","reply":"x","agreement":"leaning_agree","extra":"x"}',
])
def test_json_reconciliation_rejects_conflicts_and_other_invalid_outputs(text):
    from dataclasses import asdict

    from llm_committee.pivot import prompts
    from llm_committee.pivot.continuation import reconcile_json_duplicates
    from llm_committee.pivot.models import Request, canonical

    row = {
        "status": "invalid", "error": "Duplicate JSON field: agreement",
        "request": canonical(Request("test", "debate", "gpt-5.6-terra", (), "medium", 100,
                                     prompts.REPLY_SCHEMA).document()),
        "response": canonical(asdict(Completion(text, 100, 20))),
    }
    before = deepcopy(row)
    with pytest.raises(ValueError):
        reconcile_json_duplicates(row)
    assert row == before
