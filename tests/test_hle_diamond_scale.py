"""Synthetic-only tests for outcome-blind expansion and explicit reasoning fallback."""

import copy
import json
from dataclasses import replace

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.failures import FORMAT_RETRY_POLICY, LocalTaskFailure
from llm_committee.pivot.models import Completion, Message, Request, canonical, digest
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.storage import RunStopped
from llm_committee.pivot.strong_run import run as old_run
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import SuperGPQAMockProvider
from scripts.hle_diamond_scale import (
    FALLBACK,
    LOW_SUFFIX,
    MODEL,
    FallbackJournal,
    all_rows,
    audit,
    fallback_summary,
    prepare,
    run,
    select_extension,
)
from scripts.hle_diamond_scale_report import publish
from scripts.hle_diamond_study import TEMPLATE, PairedGraph, read, sha
from scripts.hle_diamond_study import prepare as old_prepare


def request():
    return Request(
        "synthetic/initial/1",
        "initial",
        MODEL,
        (Message("user", "Which is even? A. 2 B. 3"),),
        "medium",
        16384,
        {"type": "object"},
    )


def manifest():
    return {
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": FORMAT_RETRY_POLICY,
            "qwen_low_fallback": FALLBACK,
        }
    }


def truncated(req):
    return Completion(
        "unfinished",
        10,
        req.max_output_tokens,
        status="incomplete",
        raw={"provider": "tinker", "model": req.model, "effort": req.effort, "stop_reason": "length"},
    )


class FailedMedium:
    def __init__(self, *, low_fails=False):
        self.calls, self.low_fails = [], low_fails

    def generate(self, req):
        self.calls.append(req)
        if req.effort == "medium" or self.low_fails:
            return truncated(req)
        return Completion('{"choice":"A","position":"Synthetic answer"}', 10, 20)


def test_uniform_extension_retains_old_ids_without_outcome_selection():
    bank = {"pool_ids": [str(i) for i in range(75)], "selected": [{"id": str(i)} for i in range(20)]}
    selected = select_extension(bank)
    assert len(selected["new_ids"]) == 30
    assert not set(selected["new_ids"]) & set(selected["existing_ids"])
    for row in bank["selected"]:
        row["answer"] = "ALTERED_EVALUATION_KEY"
    assert selected == select_extension(bank)


def test_fallback_exact_context_immutable_attempts_and_resume(tmp_path):
    provider = FailedMedium()
    req = request()
    journal = FallbackJournal(tmp_path / "calls.sqlite3", manifest(), None)
    try:
        expected = journal.call(req, provider, json.loads)
        assert len(provider.calls) == 4
        for item in provider.calls[:3]:
            assert item == req
        low = provider.calls[-1]
        assert low == replace(req, key=req.key + LOW_SUFFIX, effort="low")
        assert journal.call(req, provider, json.loads) == expected
        assert len(provider.calls) == 4
        journal.should_stop = lambda: True
        assert journal.call(req, OfflineProvider(lambda *_: 0), json.loads) == expected
        rows = journal.db.execute("SELECT status FROM calls ORDER BY rowid").fetchall()
        assert [r[0] for r in rows] == ["invalid", "invalid", "invalid", "completed"]
    finally:
        journal.close()


def test_fallback_exhaustion_is_local_and_bounded(tmp_path):
    provider = FailedMedium(low_fails=True)
    journal = FallbackJournal(tmp_path / "calls.sqlite3", manifest(), None)
    try:
        for _ in range(2):
            with pytest.raises(LocalTaskFailure) as exc:
                journal.call(request(), provider, json.loads)
            assert exc.value.kind == "failed_after_low_fallback"
            assert len(exc.value.attempts) == 6
        assert len(provider.calls) == 6
    finally:
        journal.close()


def test_resume_waits_for_low_without_dispatching_offline(tmp_path):
    provider = FailedMedium()
    journal = FallbackJournal(tmp_path / "calls.sqlite3", manifest(), None)
    try:
        # Stop immediately after the third medium response has been saved.
        journal.should_stop = lambda: len(provider.calls) >= 3
        with pytest.raises(RunStopped):
            journal.call(request(), provider, json.loads)
        assert len(provider.calls) == 3
        journal.should_stop = lambda: True
        with pytest.raises(RunStopped):
            journal.call(request(), OfflineProvider(lambda *_: 0), json.loads)
        journal.should_stop = lambda: False
        assert journal.call(request(), provider, json.loads)["choice"] == "A"
        assert len(provider.calls) == 4
    finally:
        journal.close()


@pytest.mark.parametrize("mode", ["format", "other_model", "not_length", "off"])
def test_no_unauthorized_effort_change(tmp_path, mode):
    req = request()
    if mode == "other_model":
        req = replace(req, model="thinkingmachines/Inkling")
    if mode == "off":
        req = replace(req, effort="none", purpose="d_choice")

    class Provider:
        calls = 0

        def generate(self, r):
            self.calls += 1
            if mode == "format":
                return Completion("not-json", 10, 10)
            response = truncated(r)
            if mode == "not_length":
                response = replace(response, raw={**response.raw, "stop_reason": "stop"})
            return response

    provider = Provider()
    journal = FallbackJournal(tmp_path / "calls.sqlite3", manifest(), None)
    try:
        with pytest.raises(LocalTaskFailure):
            journal.call(req, provider, json.loads)
        assert provider.calls == 3
    finally:
        journal.close()


def make_source(tmp_path, provider=None):
    tone_template = read(TEMPLATE)
    tone_template["plans"] = tone_template["plans"][:1]
    tone_template["design_sha256"] = digest({k: v for k, v in tone_template.items() if k != "design_sha256"})
    tone_path = tmp_path / "old-plan/tones.json"
    atomic_json(tone_path, tone_template)
    row = {
        "id": "z-old-synthetic",
        "question": "Which number is prime?\nA. 4\nB. 7\nC. 9",
        "answer": "B",
        "category": "Math",
    }
    bank = {"dataset": "SYNTHETIC", "revision": "SYNTHETIC", "selected": [row]}
    bank_path = tmp_path / "old-plan/bank.json"
    atomic_json(bank_path, bank)
    m, c = old_prepare(bank_path, mock=True, template_file=tone_path)
    source = tmp_path / "old-live"
    old_run(
        m,
        c,
        source,
        question_limit=1,
        request_limit=8,
        workers=1,
        graph_type=PairedGraph,
        mock_provider_type=SuperGPQAMockProvider,
        provider_factory=lambda: provider or SuperGPQAMockProvider(),
        token_count=lambda *_: 0,
    )
    # New ID sorts before the original; careless all-ID re-zipping would change its tone schedule.
    new_row = {**row, "id": "a-new-synthetic"}
    bank["selected"].append(new_row)
    new_bank_path = tmp_path / "new-plan/bank.json"
    atomic_json(new_bank_path, bank)
    tone_template["plans"][0]["question"]["id"] = "new-tone-template-id"
    tone_template["design_sha256"] = digest({k: v for k, v in tone_template.items() if k != "design_sha256"})
    new_tone_path = tmp_path / "new-plan/tones.json"
    atomic_json(new_tone_path, tone_template)
    result = prepare(new_bank_path, plan=tmp_path / "new-plan", source=source, template=new_tone_path)
    return m, result, source


def test_expansion_preserves_schedules_reuses_successes_and_reports_dynamic_size(tmp_path):
    old, (m, c, rows), source = make_source(tmp_path)
    assert m["plans"][0] == old["plans"][0]
    assert m["planned_counts"]["logical_calls"] == 84
    old_hash = sha(source / "requests.sqlite3")
    provider = SuperGPQAMockProvider()
    output = tmp_path / "new-live"
    result = run(m, c, output, provider_factory=lambda: provider, request_limit=8, workers=2)
    assert result["status"] == "completed" and len(provider.calls) == 42
    assert audit(output)["requests_reconstructed"] == 84
    run(m, c, output, provider_factory=lambda: provider, request_limit=8, workers=2)
    assert len(provider.calls) == 42 and sha(source / "requests.sqlite3") == old_hash
    page = tmp_path / "report.html"
    report = publish(output, page)
    assert report["questions"] == 2 and report["coverage"]["complete_trajectories"] == 12
    soup = BeautifulSoup(page.read_text(), "html.parser")
    assert "2 题" in soup.get_text() and "2/2" in soup.get_text()
    assert "请勿公开" in soup.get_text()
    for table in soup.select("table"):
        width = len(table.select("thead th"))
        assert all(len(tr.select("td")) == width for tr in table.select("tbody tr"))
    for link in soup.select("nav a"):
        assert soup.find(id=link["href"][1:])


def test_restores_failed_medium_then_resumes_real_dependents_with_low(tmp_path):
    class FailingQwen(SuperGPQAMockProvider):
        def generate(self, req):
            if req.purpose == "initial" and req.model == MODEL and req.effort == "medium":
                self.calls.append(req)
                return truncated(req)
            return super().generate(req)

    _, (m, c, _), source = make_source(tmp_path, FailingQwen())
    provider = FailingQwen()
    output = tmp_path / "new-live"
    result = run(m, c, output, provider_factory=lambda: provider, request_limit=8, workers=2)
    assert result["status"] == "completed"
    assert sum(req.effort == "low" for req in provider.calls) == 2
    assert not any(req.purpose == "initial" and req.effort == "medium" and "z-old" in req.key for req in provider.calls)
    checked = audit(output)
    assert checked["requests_reconstructed"] == 84
    assert fallback_summary(all_rows(output))["recovered_requests"] == 2
    before = len(provider.calls)
    run(m, c, output, provider_factory=lambda: provider, request_limit=8, workers=2)
    assert len(provider.calls) == before
    page = tmp_path / "report.html"
    publish(output, page)
    assert "medium 长度失败后的 low 回退" in page.read_text()
    # The truth key remains scoring-only under fallback as well.
    changed = copy.deepcopy(c)
    for record in changed.values():
        record["answer_letter"] = "C"
    first = m["plans"][0]
    original = PairedGraph(c[first["question_id"]], first, m)
    other = PairedGraph(changed[first["question_id"]], first, m)
    journal = FallbackJournal(output / "requests.sqlite3", m, None, should_stop=lambda: True)
    try:
        original.restore(journal, OfflineProvider(lambda *_: 0))
        for key, task in original.tasks.items():
            assert canonical(task.build(original.values, lambda *_: 0).document()) == canonical(
                other.tasks[key].build(original.values, lambda *_: 0).document()
            )
    finally:
        journal.close()
