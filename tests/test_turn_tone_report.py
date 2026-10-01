"""Separate cohorts; preserve old reports and both agreement-label dimensions."""

import copy
import json
import sqlite3
from dataclasses import asdict

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot import turn_tone_report as reporting
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY
from llm_committee.pivot.models import SCREEN_JUDGES, PilotConfig, Question, Screen, digest
from llm_committee.pivot.turn_tone import VERSION, plan_questions
from llm_committee.pivot.turn_tone_run import run


@pytest.fixture(scope="module")
def completed_mock(tmp_path_factory):
    root = tmp_path_factory.mktemp("turn-tone-report")
    q = Question("q00", "Policy topic?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
    contexts = json.loads(
        json.dumps({q.id: {"question": asdict(q), "initial_answers": {str(m): "Prior opinion" for m in range(3)}}})
    )
    manifest = {
        "kind": "offline_mock",
        "protocol_version": VERSION,
        "config": asdict(PilotConfig(judge_model="gemini-3.8-flash")),
        "source": {
            "directory": "/nonexistent/frozen-main-study",
            "contexts_sha256": digest(contexts),
            "manifest_sha256": "fixture",
        },
        "plans": plan_questions([f"q{i:02}" for i in range(20)])[:1],
        "planned_counts": {},
        "question_selection": {"seed": 20260927},
        "sampling": {},
        "design": {},
        "execution": {"budget_policy": "no_limit_user_requested", "failure_policy": FORMAT_RETRY_POLICY},
    }
    run(manifest, contexts, root, question_limit=1, request_limit=16)
    return root


def test_render_preserves_current_tone_and_both_label_dimensions(completed_mock):
    manifest, report, records = reporting.load_run(completed_mock)
    html, summary = reporting.render(manifest, report, records, supplementary_href="archive.html")
    soup = BeautifulSoup(html, "html.parser")
    assert len(soup.select("html")) == 1
    assert "OFFLINE MOCK" in soup.get_text()
    assert not summary["pooled_with_archived_studies"]
    assert sum(r[1] for r in summary["A_by_current_tone"]) == 12
    assert len(summary["rows"]) == 8
    assert len(soup.select(".history")) == 1
    assert {r[0] for r in summary["A_by_current_tone"]} == {"friendly", "neutral", "hostile"}
    for row in summary["rows"]:
        assert row["tone"] == "turn_level" and row["probe_tone"] == "neutral"
        assert row["current_tone"] in ("friendly", "neutral", "hostile")
    for node in soup.select("#c table, #d table"):
        headers = [h.get_text() for h in node.select("thead th")]
        assert "Peer's preceding self-label" in headers
        assert "Receiver's current self-label" in headers
        assert "Turn T" in headers
    for node in soup.select("#d table"):
        assert "Receiver model" in node.get_text()
    for node in soup.select("table"):
        width = len(node.select("thead th"))
        assert all(len(r.select("td")) == width for r in node.select("tbody tr"))
    ids = [n["id"] for n in soup.select("[id]")]
    assert len(ids) == len(set(ids))
    assert all(a["href"][1:] in ids for a in soup.select('a[href^="#"]'))
    assert "Both probes are neutral" in soup.get_text()
    assert "not 160 independent questions" in soup.get_text()


def test_loading_rejects_changed_tone_or_label(completed_mock, tmp_path):
    for name in ("manifest.json", "report.json"):
        (tmp_path / name).write_bytes((completed_mock / name).read_bytes())
    (tmp_path / "questions").mkdir()
    (tmp_path / "requests.sqlite3").write_bytes((completed_mock / "requests.sqlite3").read_bytes())
    original = json.loads((completed_mock / "questions" / "q00.json").read_text())
    for field, value in (("current_tone", "invalid"), ("previous_peer_self_label", "invalid")):
        record = copy.deepcopy(original)
        record["events"][0][field] = value
        (tmp_path / "questions" / "q00.json").write_text(json.dumps(record))
        with pytest.raises(AssertionError):
            reporting.load_run(tmp_path)


def test_unknown_billing_is_not_reported_as_spent(completed_mock, tmp_path):
    for name in ("manifest.json", "report.json", "requests.sqlite3"):
        (tmp_path / name).write_bytes((completed_mock / name).read_bytes())
    (tmp_path / "questions").mkdir()
    (tmp_path / "questions" / "q00.json").write_bytes((completed_mock / "questions" / "q00.json").read_bytes())
    with sqlite3.connect(tmp_path / "requests.sqlite3") as db:
        db.execute(
            "INSERT INTO calls(key,request,request_hash,status,charge) VALUES('unknown','{}','fixture','uncertain',10)"
        )
    manifest, report, records = reporting.load_run(tmp_path)
    assert report["report_cost_accounting"]["unresolved_reservations_usd"] == 10
    assert report["report_cost_accounting"]["unresolved_attempts"] == 1
    html, _ = reporting.render(manifest, report, records, supplementary_href="archive.html")
    assert "US$10.00 is retained as a conservative reservation, not confirmed spending" in html


def test_archive_is_a_separate_copy_with_unchanged_metrics(tmp_path):
    source, output = tmp_path / "old.html", tmp_path / "supplementary.html"
    html = "<html><head><title>Old</title></head><body><h1>Old</h1><table><tr><td>12/34</td></tr></table></body></html>"
    data = {"scores": [0.1, 0.2], "supplements": {"dyadic": {"n": 2160}}}
    source.write_text(html)
    source.with_suffix(".json").write_text(json.dumps(data))
    before = source.read_bytes(), source.with_suffix(".json").read_bytes()
    reporting.archive_fixed_tone(source, output)
    assert before == (source.read_bytes(), source.with_suffix(".json").read_bytes())
    archived = json.loads(output.with_suffix(".json").read_text())
    assert all(archived[k] == v for k, v in data.items())
    assert "12/34" in output.read_text() and "Supplementary archive" in output.read_text()
    with pytest.raises(ValueError, match="already exists"):
        reporting.archive_fixed_tone(source, output)
    reporting.archive_fixed_tone(source, output, replace=True)
    with pytest.raises(ValueError, match="canonical"):
        reporting.archive_fixed_tone(source, source, replace=True)
    output.with_suffix(".json").write_text('{"source":"unrelated"}')
    with pytest.raises(ValueError, match="unrelated"):
        reporting.archive_fixed_tone(source, output, replace=True)
