"""Equal output-budget E must not alter debates, leak labels, or import the third member."""

import json
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from llm_committee.pivot import prompts
from llm_committee.pivot.dyadic import PAIRS
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.quality_run import run
from llm_committee.pivot.quality_study import (
    ARMS,
    CALIBRATION_KINDS,
    LENGTH,
    VERBOSE_LENGTH,
    QualityGraph,
    QualityMockProvider,
    parse_answer,
    prepare,
    synthesis_messages,
    words,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from llm_committee.pivot.strong_run import run as run_debate
    from llm_committee.pivot.strong_study import prepare as prepare_debate

    template = Path(__file__).resolve().parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"
    source = tmp_path_factory.mktemp("quality-debate-source")
    manifest, contexts = prepare_debate(template, mock=True)
    assert run_debate(manifest, contexts, source, question_limit=20)["status"] == "completed"
    return prepare(source, mock=True)


def test_counts_and_frozen_inputs(prepared):
    m, contexts = prepared
    assert m["planned_counts"]["logical_calls"] == 540
    assert m["planned_counts"]["by_purpose"] == {"synthesis": 216, "judge_e": 312, "calibration_validation": 12}
    assert len(contexts) == 20
    assert sum(bool(p["calibration"]) for p in m["plans"]) == 12
    assert Counter(p["calibration"]["pair"] for p in m["plans"] if p["calibration"]) == dict.fromkeys(PAIRS, 4)
    for pair in PAIRS:
        for arm in ARMS:
            assert Counter(p["orders"][pair][arm][0] for p in m["plans"]) == {False: 10, True: 10}
            assert all(p["orders"][pair][arm][0] != p["orders"][pair][arm][1] for p in m["plans"])
    for context in contexts.values():
        frozen = digest(context)
        for pair, members in PAIRS.items():
            requests = [synthesis_messages(context, pair, arm) for arm in ("baseline", *ARMS)]
            assert len({r[0].text for r in requests}) == 1
            assert len({r[1].text for r in requests}) == 1
            for arm, request in zip(("baseline", *ARMS), requests, strict=True):
                material = json.loads(request[-1].text)
                assert {v["member"] for v in material["initial_answers"]} == set(members)
                assert all(
                    v["text"] == context["initial_answers"][str(v["member"])] for v in material["initial_answers"]
                )
                assert len(material["discussion"]) == (0 if arm == "baseline" else 4)
                assert all(n["id"].startswith(pair + "-") for n in material["discussion"])
                assert all(set(n) == {"id", "member", "reply_to", "text"} for n in material["discussion"])
                assert '"agreement"' not in request[-1].text and '"tone_schedule"' not in request[-1].text
            original, alternate = [json.loads(r[-1].text)["discussion"] for r in requests[1:]]
            assert original[:2] == alternate[:2]
        assert digest(context) == frozen


def test_stateful_source_uses_formal_initial_positions_without_private_peer_fields(tmp_path):
    from llm_committee.pivot.stateful_run import run as run_debate
    from llm_committee.pivot.stateful_study import prepare as prepare_debate

    template = Path(__file__).resolve().parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"
    source = tmp_path / "stateful-source"
    manifest, contexts = prepare_debate(template, mock=True)
    assert run_debate(manifest, contexts, source, question_limit=20)["status"] == "completed"
    m, quality_contexts = prepare(source, mock=True)
    assert m["planned_counts"]["logical_calls"] == 540
    for qid, context in quality_contexts.items():
        record = json.loads((source / "questions" / f"{qid}.json").read_text())
        assert context["initial_answers"] == {k: v["position"] for k, v in record["initial_positions"].items()}
        for pair in PAIRS:
            for arm in ARMS:
                material = json.loads(synthesis_messages(context, pair, arm)[-1].text)
                assert all(set(n) == {"id", "member", "reply_to", "text"} for n in material["discussion"])
                for n in material["discussion"]:
                    assert n["text"] == record["branches"][arm]["formal_replies"]["turn_level"][n["id"]]["reply"]


@pytest.mark.parametrize("n,valid", [(189, False), (190, True), (200, True), (210, True), (211, False)])
def test_hard_word_count(n, valid):
    text = canonical({"answer": " ".join(["word"] * n)})
    if valid:
        assert words(parse_answer(text)["answer"]) == n
    else:
        with pytest.raises(ValueError, match="Length contract failed"):
            parse_answer(text)


@pytest.fixture(scope="module")
def completed(prepared, tmp_path_factory):
    m, contexts = prepared
    output = tmp_path_factory.mktemp("quality-completed")
    report = run(m, contexts, output, question_limit=20)
    assert report["status"] == "completed"
    assert report["call_status_counts"] == {"completed": 540}
    assert report["comparison_statuses"] == {"success": 120}
    return output, m, contexts


def test_snapshot_audit_writes_output_before_temporary_package_cleanup(completed, tmp_path):
    source, _, _ = completed
    output = tmp_path / "snapshot-audit.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_frozen_run",
            str(source),
            "--kind",
            "quality_study",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    result = json.loads(output.read_text())
    assert result["used_archived_implementation"]
    assert result["request_contexts_reconstructed_exactly"] == 540


def test_fresh_synthesis_blind_judge_and_calibration(completed):
    output, m, contexts = completed
    with sqlite3.connect(output / "requests.sqlite3") as db:
        calls = {
            json.loads(r)["key"]: (json.loads(r), json.loads(p))
            for r, p in db.execute("SELECT request,parsed FROM calls")
        }
    for request, value in calls.values():
        assert request["purpose"] in ("synthesis", "judge_e", "calibration_validation")
        assert not request["candidate_labels"]
        if request["purpose"] == "synthesis":
            bounds = VERBOSE_LENGTH if "/calibration/verbosity/" in request["key"] else LENGTH
            assert bounds[0] <= words(value["answer"]) <= bounds[1]
            assert request["model"] == "gpt-5.6-terra" and request["effort"] == "medium"
        if request["purpose"] == "judge_e":
            assert request["model"] == "gemini-3.8-flash"
            material = json.loads(request["messages"][-1]["text"])
            assert set(material) == {"question", "options", "left", "right"}
    for plan in m["plans"]:
        graph = QualityGraph(contexts[plan["question_id"]], plan, m)
        for pair in PAIRS:
            for arm in ARMS:
                a, b = [calls[graph.key(f"{pair}/{arm}/order{i}")][0] for i in range(2)]
                aa, bb = [json.loads(r["messages"][-1]["text"]) for r in (a, b)]
                assert aa["left"] == bb["right"] and aa["right"] == bb["left"]
                assert a["messages"][0]["text"] == prompts.e_messages(graph.question, "left", "right")[0].text
        record = json.loads((output / "questions" / f"{plan['question_id']}.json").read_text())
        for result in record["E"]:
            votes = [o["judgment"]["preference"] == o["target_side"] for o in result["preferences"]["orders"]]
            assert result["debate_score"] == sum(votes) / 2
        if record["calibration"]:
            assert set(record["calibration"]["variants"]) == set(CALIBRATION_KINDS)


def test_length_retry_is_bounded_symmetric_and_resume_is_free(prepared, tmp_path):
    m, contexts = prepared
    plan = m["plans"][0]
    graph = QualityGraph(contexts[plan["question_id"]], plan, m)
    target = graph.key("AB/baseline/synthesis")
    seen = Counter()

    class TooShort(QualityMockProvider):
        def generate(self, request):
            seen[request.key] += 1
            value = super().generate(request)
            return replace(value, text=canonical({"answer": "Too short"})) if request.key == target else value

    result = run(m, contexts, tmp_path / "failure", question_limit=1, provider_factory=TooShort)
    assert seen[target] == 3
    assert result["comparison_statuses"] == {"failed": 2, "success": 4}
    assert not result["fatal_errors"]
    before = seen.copy()
    run(m, contexts, tmp_path / "failure", question_limit=1, provider_factory=TooShort)
    assert seen == before
    from scripts.audit_quality_study import audit

    checked = audit(tmp_path / "failure")
    assert checked["question_reports_reconstructed_exactly"] == 1
    assert checked["failed_logical_requests_verified"] == 1
    assert checked["blocked_logical_requests_verified"] == 4
    assert checked["primary_pairs_both_orders_verified"] == 4
    assert checked["primary_pairs_unavailable"] == 2


def test_report_counts_lengths_question_intervals_and_cases(completed):
    from bs4 import BeautifulSoup

    from llm_committee.pivot.quality_report import load_run, render

    output, _, _ = completed
    m, r, records = load_run(output)
    html, summary = render(m, r, records)
    soup = BeautifulSoup(html, "html.parser")
    assert len(summary["primary_rows"]) == 120
    assert len(summary["primary_summary"]) == 6
    assert len(summary["calibration_rows"]) == 36
    assert len(soup.select(".e-case")) == 60
    assert len(soup.select(".e-calibration-case")) == 12
    assert "OFFLINE MOCK" in soup.get_text()
    assert "Debate wins (%) [95% interval]" in [th.get_text() for th in soup.select("th")]
    assert "Each complete pair contributes two decisions" in soup.get_text()
    assert "not the percentage improvement in answer quality" in soup.get_text()
    assert "Debate score" not in soup.get_text()
    for group in summary["primary_summary"]:
        assert group["estimate"]["questions"] == group["valid"] == group["planned"] == 20
        assert group["baseline_words"] == group["debate_words"] == 200
    for t in soup.select("table"):
        assert all(len(row.select("td")) == len(t.select("thead th")) for row in t.select("tbody tr"))


def test_embedding_is_cohort_checked_and_preserves_all_existing_data(completed, monkeypatch):
    from bs4 import BeautifulSoup

    from llm_committee.pivot import quality_report

    output, _, _ = completed
    m, r, records = quality_report.load_run(output)
    summary = {
        "source_directory": m["source"]["directory"],
        "manifest_sha256": m["source"]["manifest_sha256"],
        "rows": ["unchanged A-D rows"],
    }
    document = '<html><body><nav><a href="#paired">Paired</a></nav><section id="c"><table id="old"><tr><td>original</td></tr></table></section><section id="paired">paired cases</section></body></html>'
    with pytest.raises(ValueError, match="mock"):
        quality_report.include_quality(document, summary, output)
    monkeypatch.setattr(
        quality_report, "load_run", lambda source: ({**m, "kind": "paid_turn_tone_quality"}, r, records)
    )
    html, combined = quality_report.include_quality(document, summary, output)
    assert all(combined[k] == v for k, v in summary.items())
    soup = BeautifulSoup(html, "html.parser")
    assert len(soup.select("#e")) == 1 and len(soup.select("#paired")) == 1
    assert str(soup.select_one("#old")) == str(BeautifulSoup(document, "html.parser").select_one("#old"))
    with pytest.raises(ValueError, match="already included"):
        quality_report.include_quality(html, combined, output)
    with pytest.raises(ValueError, match="different"):
        quality_report.include_quality(document, {**summary, "manifest_sha256": "wrong"}, output)
