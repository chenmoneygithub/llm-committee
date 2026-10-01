"""Offline distributions retain negative events, paired labels and cohort identity."""

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from scripts.stateful_probability_diagnostics import build, distribution_plot, grouped, include_diagnostics, render


def row(qid, node, delta, *, member=1, previous="leaning_disagree", current="leaning_disagree"):
    return {
        "question_id": qid,
        "node_id": node,
        "assignment": "original",
        "member": member,
        "T": 4,
        "previous_peer_self_label": previous,
        "current_self_label": current,
        "D_choice_adjacent": delta,
        "choice_p_before": 25,
        "choice_p_after": 25 + delta,
        "previous_reading": "turn_level/BC-2",
        "C_choice_changed": False,
    }


def test_distribution_keeps_negative_events_and_uses_question_equal_mean():
    groups = grouped([row("q1", "a", -25), row("q1", "b", 75), row("q2", "a", -15)])
    assert len(groups) == 1
    g = groups[0]
    assert g["question_equal_mean_pp"] == 5  # mean(25, -15), not mean(-25, 75, -15)
    assert g["any_negative"] == g["drop_over_1pp"] == 2
    assert g["rise_over_1pp"] == 1 and g["event_median_pp"] == -15
    soup = BeautifulSoup(distribution_plot(groups, 1), "html.parser")
    assert len(soup.select("circle.event-dot")) == 3
    assert len(soup.select("path.mean-diamond")) == 1
    assert "-25.000000" in soup.get_text()


def test_groups_do_not_pool_labels_or_models_and_small_changes_stay_visible():
    groups = grouped(
        [
            row("q1", "a", -0.002, member=2),
            row("q1", "b", 0.003, member=2),
            row("q2", "a", 0, current="leaning_agree"),
            row("q3", "a", 5, previous="strongly_agree"),
        ]
    )
    assert len(groups) == 3
    ink = next(g for g in groups if g["member"] == 2)
    assert ink["within_1pp"] == 2 and ink["any_negative"] == 1
    assert len(ink["points"]) == 2
    assert all(g["drop_over_1pp"] + g["within_1pp"] + g["rise_over_1pp"] == g["events"] for g in groups)


def test_embedding_checks_cohort_and_does_not_change_existing_results(tmp_path):
    path = tmp_path / "diagnostic.html"
    data = {
        "kind": "offline_stateful_D1_diagnostic",
        "manifest_sha256": "sha",
        "source_directory": "/frozen",
        "groups": grouped([row("q1", "a", -25)]),
        "checks": {},
    }
    path.with_suffix(".json").write_text(json.dumps(data))
    document = "<h3>D2 · Endorsement of the fixed pre-turn position: argument vs filler</h3><p>Original data</p>"
    summary = {"manifest_sha256": "sha", "source_directory": "/frozen", "A": ["unchanged"]}
    html, result = include_diagnostics(document, summary, path)
    assert result["A"] == summary["A"] and "Original data" in html
    assert result["D1_diagnostic"]["new_model_calls"] == 0
    with pytest.raises(ValueError, match="different cohort"):
        include_diagnostics(document, {**summary, "manifest_sha256": "wrong"}, path)
    with pytest.raises(ValueError, match="already included"):
        include_diagnostics(html, result, path)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/stateful-dyadic-20-20260927/live"


@pytest.mark.skipif(
    not (SOURCE / "requests.sqlite3").exists(), reason="Read-only integration check requires local paid archive"
)
def test_archived_close_reading_is_exact_and_never_calls_a_model():
    data = build(SOURCE, ROOT / "docs/stateful-d1-case-notes-2026-09-27.json")
    assert data["new_model_calls"] == 0
    assert data["checks"] == {"unique_comparisons_recomputed": 141, "verbatim_quotes_checked": 28}
    assert len(data["cases"]) == 10
    assert sum(g["events"] for g in data["groups"]) == 141
    changed = next(c for c in data["cases"] if c["question_id"] == "global-row-1269")
    assert changed["row"]["D_choice_adjacent"] > 0 and changed["row"]["C_choice_changed"]
    soup = BeautifulSoup(render(data), "html.parser")
    assert len(soup.select(".diagnostic-case")) == 10 and len(soup.select(".full-prompt")) == 40
    assert len(soup.select("circle.event-dot")) == 141
    for table in soup.select("table"):
        assert all(len(r.select("td")) == len(table.select("thead th")) for r in table.select("tbody tr"))
