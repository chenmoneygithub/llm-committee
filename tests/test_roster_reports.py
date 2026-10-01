"""Same-family reports must not inherit mixed-family model names or D tables."""

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.models import ROSTERS
from llm_committee.pivot.stateful_run import run
from llm_committee.pivot.stateful_study import prepare
from llm_committee.pivot.triadic_report import publish as publish_triadic
from llm_committee.pivot.triadic_run import run as run_triadic
from llm_committee.pivot.triadic_study import prepare as prepare_triadic
from scripts.audit_stateful_dyadic import audit
from scripts.report_roster_triadic import publish as publish_roster_triadic
from scripts.report_stateful_dyadic import publish

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["same_family", "same_model"])
def dyadic(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("same-family-report")
    m, c = prepare(
        ROOT / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json", mock=True, roster=request.param, public_history=True
    )
    result = run(m, c, root / "run", question_limit=20)
    assert result["status"] == "completed"
    return root, publish(root / "run", root / "dyadic.html")


def test_roster_labels_and_no_D(dyadic):
    root, output = dyadic
    data = json.loads(output.with_suffix(".json").read_text())
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert data["models"] == list(ROSTERS[data["roster"]])
    assert data["D_enabled"] is False
    assert data["D1_adjacent"] == data["D1_initial"] == data["D2"] == []
    assert data["formal_replies"] == 360 and data["sampled_events"] == 267
    assert not soup.select('#d, #probes, a[href="#d"], a[href="#probes"]')
    assert len(soup.select(".case")) == 60
    assert "Qwen" not in soup.get_text() and "Inkling" not in soup.get_text()
    assert all(model in soup.select_one("#member-endpoints").get_text() for model in ROSTERS[data["roster"]])
    if data["roster"] == "same_model":
        labels = [f"{chr(65 + m)} (member {m}) · gpt-5.6-terra" for m in range(3)]
        assert [row.select_one("td").get_text() for row in soup.select("#member-endpoints tbody tr")] == labels
        assert all(name in soup.select_one("#paired").get_text() for name in labels)
    assert soup.select_one("#c") and soup.select_one("#b")
    for link in soup.select('a[href^="#"]'):
        assert soup.find(id=link["href"][1:]) is not None
    result = audit(root / "run")
    assert result["completed_requests"] == result["requests_reconstructed_exactly"] == 1168
    assert result["native_token_checks"] == []


@pytest.mark.parametrize("roster", ["same_family", "same_model"])
def test_triadic_roster_and_reference_link(tmp_path, roster):
    m, c = prepare_triadic(ROOT / "docs/turn-tone-triadic-shared-plan-2026-09-28.json", roster=roster, mock=True)
    assert run_triadic(m, c, tmp_path / "run", question_limit=1)["status"] == "preflight_completed"
    original = publish_triadic(tmp_path / "run", tmp_path / "original.html")
    output = publish_roster_triadic(tmp_path / "run", tmp_path / "triadic.html", dyadic_report=tmp_path / "dyadic.html")
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert ("GPT same-family" if roster == "same_family" else "Same-model") in soup.h1.get_text()
    assert soup.find("a", href="dyadic.html").get_text() == "Same roster · two-member report"
    assert not soup.find("a", href="turn-tone-dyadic-public-history-2026-09-27.html")
    assert not soup.select('#c, #d, a[href="#c"], a[href="#d"]')
    # Wrapping changes labels/navigation only, never scientific tables or summaries.
    before = BeautifulSoup(original.read_text(), "html.parser")

    def without_member_prefixes(text):
        for m in range(3):
            text = text.replace(f"{chr(65 + m)} (member {m}) · ", "")
        return text

    assert [without_member_prefixes(t.get_text()) for t in soup.select("table")] == [
        t.get_text() for t in before.select("table")
    ]
    if roster == "same_model":
        labels = [f"{chr(65 + i)} (member {i}) · gpt-5.6-terra" for i in range(3)]
        assert [row.select_one("td").get_text() for row in soup.select("#e1 tbody tr")] == labels
        assert all(name in soup.select_one("#cases").get_text() for name in labels)
    new_data = json.loads(output.with_suffix(".json").read_text())
    old_data = json.loads(original.with_suffix(".json").read_text())
    assert {k: v for k, v in new_data.items() if k not in ("roster", "models")} == old_data
