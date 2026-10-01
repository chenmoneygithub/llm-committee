"""New reports must use integrated positions and preserve paired-label conditioning."""

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.stateful_run import run
from llm_committee.pivot.stateful_study import prepare
from scripts.audit_stateful_dyadic import audit
from scripts.report_stateful_dyadic import collect, load_run, publish


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    plan = Path(__file__).resolve().parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"
    output = tmp_path_factory.mktemp("stateful-report-source")
    manifest, contexts = prepare(plan, mock=True)
    assert run(manifest, contexts, output, question_limit=20)["status"] == "completed"
    return output


def test_frozen_reconstruction_and_probabilities(completed):
    result = audit(completed)
    assert result["completed_requests"] == result["requests_reconstructed_exactly"] == 1676
    assert result["questions_reconstructed_exactly"] == 20
    assert result["argument_control_pairs_differ_only_in_incoming"] == 143
    assert result["native_checks_are_synthetic"]


def test_report_conditioning_dedup_and_full_prompts(completed, tmp_path):
    _, _, records = load_run(completed)
    formal, rows = collect(records)
    assert len(formal) == 360 and len(rows) == 267
    output = publish(completed, tmp_path / "report.html")
    summary = json.loads(output.with_suffix(".json").read_text())
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert "OFFLINE MOCK" in soup.get_text()
    assert len(soup.select(".case")) == 60
    assert len(soup.select(".probe-case")) == 10
    assert sum(g["replies"] for g in summary["C_adjacent"]) == 267
    assert sum(g["pairs"] for g in summary["D2"]) == 143
    assert sum(g["compared"] for g in summary["D1_adjacent"]) == 143
    assert all(
        {"previous_peer_self_label", "current_self_label"} <= set(g)
        for key in ("C_adjacent", "C_endpoint", "D1_adjacent", "D2")
        for g in summary[key]
    )
    assert '"your_current_position"' in soup.select_one("#probes").get_text()
    assert '"incoming_peer_message"' in soup.select_one("#probes").get_text()
    for t in soup.select("table"):
        assert all(len(row.select("td")) == len(t.select("thead th")) for row in t.select("tbody tr"))


def test_matching_E_embeds_without_old_cohort_reuse(completed, tmp_path, monkeypatch):
    from llm_committee.pivot import quality_report
    from llm_committee.pivot.quality_run import run as run_quality
    from llm_committee.pivot.quality_study import prepare as prepare_quality

    m, contexts = prepare_quality(completed, mock=True)
    source = tmp_path / "quality"
    assert run_quality(m, contexts, source, question_limit=20)["status"] == "completed"
    with pytest.raises(ValueError, match="mock"):
        publish(completed, tmp_path / "combined.html", quality_source=source)
    qm, qr, records = quality_report.load_run(source)
    monkeypatch.setattr(quality_report, "load_run", lambda _: ({**qm, "kind": "paid_turn_tone_quality"}, qr, records))
    output = publish(completed, tmp_path / "combined.html", quality_source=source)
    summary = json.loads(output.with_suffix(".json").read_text())
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert len(soup.select("#e")) == 1 and len(soup.select("#paired")) == 1
    assert len(summary["length_controlled_E"]["primary_rows"]) == 120
    assert sum(g["pairs"] for g in summary["D2"]) == 143
