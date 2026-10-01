"""Supplement reporting stays conditional and preserves all main-study results."""

import copy

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.models import digest
from llm_committee.pivot.results_report import question_estimate
from llm_committee.pivot.stateful_study import rating_metrics
from llm_committee.pivot.study import atomic_json
from scripts import report_forced_feedback_public_history as report
from scripts.scale_public_history import read, sha


def case(qid="q0", cut=1, pair="BC"):
    control = {"probabilities": dict(zip("ABCDEFG", (0, 0, 0, 0, 0, 0.3, 0.7), strict=True))}
    argument = {"probabilities": dict(zip("ABCDEFG", (0, 0, 0, 0, 0, 0.6, 0.4), strict=True))}
    reply = {"reply": "Public argument", "position": "New position", "choice": "B", "agreement": "leaning_disagree"}
    arm = {
        "feedback": {**reply, "agreement": "strongly_disagree"},
        "return": reply,
        "challenger_label": "strongly_disagree",
        "recipient_label": "leaning_disagree",
        "B_feedback": {"label": "strongly_disagree", "evidence": "Opposing conclusion"},
        "B_return": {"label": "leaning_disagree", "evidence": "Rejects premise"},
        "C_choice_changed": True,
        "C_text": {"label": "adjusted", "evidence": "Narrower scope"},
        "position_before": {"choice": "A", "position": "Old position"},
        "D1_before": {"probabilities": {"A": 0.8, "B": 0.2}},
        "D1_after": {"probabilities": {"A": 0.5, "B": 0.5}},
        "D1_prior_choice_change_pp": -30,
        "D2": {"argument": argument, "control": control, **rating_metrics(argument, control)},
        "synthesis": {"answer": "Answer words"},
    }
    return {
        "question_id": qid,
        "id": f"{pair}-T{cut}",
        "pair": pair,
        "cut_T": cut,
        "return_T": cut + 1,
        "recipient": 1,
        "challenger": 2,
        "source_assignment": "original",
        "model": "Qwen/Qwen3.8-27B",
        "arms": {"natural": copy.deepcopy(arm), "forced": copy.deepcopy(arm)},
        "E_forced_wins": 0.5,
        "E_order_inconsistent": True,
        "E_orders": [
            {"target_side": "left", "judgment": {"preference": "left", "evidence": "First"}},
            {"target_side": "right", "judgment": {"preference": "left", "evidence": "Second"}},
        ],
    }


@pytest.fixture
def completed_fixture(tmp_path):
    """Synthetic report inputs only: never dispatch a model or touch live records."""
    output = tmp_path / "synthetic-report"
    records, plans = [], []
    for i in range(50):
        qid = f"q{i}"
        records.append(
            {
                "question_id": qid,
                "question": {"text": "Question <safe>", "options": ["Yes", "No"]},
                "initial_positions": {"1": {"position": "Initial B"}, "2": {"position": "Initial C"}},
                "cases": [case(qid, cut, pair) for cut in (1, 3) for pair in ("AB", "CA", "BC")],
            }
        )
        plans.append({"question_id": qid})
        atomic_json(output / "questions" / f"{qid}.json", records[-1])
    manifest = {"kind": "paid_public_history_forced_feedback", "protocol_version": report.VERSION, "plans": plans}
    run = {
        "status": "completed",
        "completed_questions": 50,
        "completed_cases": 300,
        "successful_cases": 300,
        "completed_new_calls_by_purpose": {"debate": 600},
        "reused_measurements": 100,
        "cost_accounting": {"received_response_estimate_usd": 1, "unresolved_reservations_usd": 0},
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "report.json", run)
    audit = {
        "status": "passed",
        "cases_reconstructed": 300,
        "mock": False,
        "manifest_sha256": sha(output / "manifest.json"),
        "report_sha256": sha(output / "report.json"),
        "question_records_sha256": digest(records),
    }
    atomic_json(output / "audit.json", audit)
    return output


def test_conditional_cells_and_missing_not_zero():
    rows = report.flatten([case()])
    extra = copy.deepcopy(rows[0])
    extra.update(C_choice_changed=None, C_text={"label": "unjudgeable"})
    other = copy.deepcopy(rows[0])
    other.update(challenger_label="strongly_agree", recipient_label="strongly_agree")
    assert [len(group) for _, group in report.cells([*rows, extra, other])] == [1, 3]
    soup = BeautifulSoup(report.c_table([rows[0], extra]), "html.parser")
    values = [td.get_text() for td in soup.select("tbody tr td")]
    assert values[:3] == ["Strongly disagree", "Leaning disagree", "2 / 1"]
    assert values[3:] == ["1/1 (100.0%)", "1/1 (100.0%)", "0/1 (0.0%)"]


def test_probability_signs_and_three_distinct_d2_measures():
    row = report.flatten([case()])[0]
    assert row["D1_before_pp"] == 80 and row["D1_after_pp"] == 50
    assert row["D1_prior_choice_change_pp"] == -30
    assert row["modal_rating_change"] == -1
    assert row["reference_probability_change_pp"] == pytest.approx(-30)
    assert row["mean_own_agreement_argument_minus_control"] == pytest.approx(-0.3)
    rendered = report.d_tables([row])
    assert "-30.00" in rendered and "-0.30" in rendered
    assert "1 / 0 / 0（n=1）" in rendered


def test_question_weighting_not_flat_event_weighting():
    rows = [{"question_id": "a", "score": 1}] * 3 + [{"question_id": "b", "score": 0}]
    assert report.qmean(rows, "score") == 0.5
    estimate = question_estimate(rows, "score")
    assert estimate["mean"] == 0.5 and estimate["questions"] == 2 and estimate["n"] == 4


def test_analyze_and_render_full_fixture(completed_fixture):
    result = report.analyze(completed_fixture)
    assert [s["cases"] for s in result["summary"]] == [150, 150]
    assert all(s["E_forced_wins"]["mean"] == 0.5 for s in result["summary"])
    assert "C_by_labels" in result["summary"][0]["arms"]["forced"]
    soup = BeautifulSoup(report.render(result), "html.parser")
    assert len(soup.select(".ff-case")) == 300
    assert "Question &lt;safe&gt;" in str(soup)
    assert all(len(t.select("thead th")) <= 6 for t in soup.select("table"))
    assert "选择 强制反驳" in soup.get_text() and "选择 自然对照" in soup.get_text()
    assert "0.5 不是真实平局" in soup.get_text()


@pytest.mark.parametrize("change", ["mock", "partial", "short_audit", "mock_audit", "stale_question", "stale_report"])
def test_refuse_incomplete_mock_or_stale_audit(completed_fixture, change):
    output = completed_fixture
    if change == "mock":
        path = output / "manifest.json"
        data = read(path)
        data["kind"] = "offline_mock"
    elif change in ("partial", "stale_report"):
        path = output / "report.json"
        data = read(path)
        data["status" if change == "partial" else "successful_cases"] = "running" if change == "partial" else 299
    elif change == "stale_question":
        path = output / "questions/q0.json"
        data = read(path)
        data["cases"][0]["E_forced_wins"] = 1
    else:
        path = output / "audit.json"
        data = read(path)
        data["cases_reconstructed" if change == "short_audit" else "mock"] = 6 if change == "short_audit" else True
    atomic_json(path, data)
    with pytest.raises(ValueError):
        report.analyze(output)


def test_publish_is_idempotent_and_preserves_main_results(completed_fixture, tmp_path):
    output = completed_fixture
    contexts = {
        f"q{i}": {
            "record": {
                "branches": {
                    "original": {
                        "formal_replies": {
                            "turn_level": {
                                f"{pair}-{t}": {"reply": f"Public prefix {pair} {t}"}
                                for pair in ("AB", "CA", "BC")
                                for t in (1, 2)
                            }
                        }
                    }
                }
            }
        }
        for i in range(50)
    }
    atomic_json(output / "source-contexts.json", contexts)
    page = tmp_path / "report.html"
    baseline = '<!DOCTYPE html><html><head></head><body><nav></nav><main><section id="e"><h2>Main E</h2><table><tr><td>42</td></tr></table></section></main></body></html>'
    page.write_text(baseline)
    atomic_json(page.with_suffix(".json"), {"original_scores": [1, 2, 3]})
    report.publish(output, page, refresh_dashboard=False)
    first = page.read_bytes()
    report.publish(output, page, refresh_dashboard=False)
    assert first == page.read_bytes()
    soup = BeautifulSoup(first, "html.parser")
    assert len(soup.select(f"#{report.SECTION}")) == 1
    assert len(soup.select("#ff-supplement-link")) == 1
    assert soup.select_one("#e").get_text() == "Main E42"
    assert read(page.with_suffix(".json"))["original_scores"] == [1, 2, 3]
    assert (output / "base-report-before-supplement.html").read_text() == baseline
    assert "Public prefix" in soup.get_text()
