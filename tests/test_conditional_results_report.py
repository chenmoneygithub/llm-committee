"""Regression contract: the current A label is the primary B/C/D grouping key."""

from bs4 import BeautifulSoup

from llm_committee.pivot.conditional_results_report import analyze_conditional, b_finding, render_b


def event(label="leaning_disagree", **overrides):
    probs = {letter: float(letter == "G") for letter in "ABCDEFG"}
    return {
        "A": label,
        "B": {"label": "fully_agree"},
        "T": 3,
        "participation_index": 2,
        "member": 1,
        "C_choice_adjacent_changed": True,
        "C_choice_initial_changed": False,
        "C_text_adjacent": {"label": "conclusion_changed"},
        "C_text_final": {"label": "unchanged"},
        "final_pair": "endpoint-pair",
        "D_choice_adjacent_pp": -10,
        "D_choice_initial_pp": 1,
        "D_text": {
            "argument": {"probabilities": probs},
            "control": {"probabilities": probs},
            "mean_own_agreement_argument_minus_control": 0,
        },
        **overrides,
    }


def test_grouping_uses_current_self_label_not_judge_label_or_previous_speaker():
    data = analyze_conditional([{"question_id": "q", "sampled_events": [event()]}])
    b = next(r for r in data["B"] if r["self_label"] == "leaning_disagree")
    assert b["n"] == b["counts"]["fully_agree"] == 1
    assert next(r for r in data["B"] if r["self_label"] == "fully_agree")["n"] == 0
    c = next(r for r in data["C_by_label_turn"] if r["self_label"] == "leaning_disagree" and r["T"] == 3)
    assert c["choice"]["changed"] == 1 and c["initial_choice"]["changed"] == 0
    final = next(r for r in data["C_endpoint_by_label"] if r["self_label"] == "leaning_disagree")
    assert final["choice"]["changed"] == 0 and final["text"]["counts"]["unchanged"] == 1
    assert sum(r["text"]["n"] for r in data["C_by_label_turn"]) == 1
    d = next(r for r in data["D_by_label"] if r["self_label"] == "leaning_disagree" and r["model"] == "Qwen3.8-27B")
    assert d["D_choice_adjacent_pp"]["mean"] == -10 and d["D_text_control"]["mean"] == 7
    assert all(r["D_text_delta"]["mean"] is None for r in data["D_by_label"] if r["self_label"] == "fully_disagree")


def test_b_rows_include_unjudgeable_without_inflating_matching_percentage():
    rows = [event("fully_agree"), event("fully_agree", B={"label": "unjudgeable"}), event("leaning_agree")]
    data = analyze_conditional([{"question_id": "q", "sampled_events": rows}])
    parsed = BeautifulSoup(render_b(data["B"]), "html.parser")
    first = parsed.tbody.find_all("tr")[0].get_text(" ")
    assert first.count("1 (50.0%)") == 2
    headline = BeautifulSoup(b_finding(data["B"]), "html.parser").get_text()
    assert "Fully agree: 1/2 (50.0%)" in headline
    assert "Leaning agree: 0/1 (0.0%)" in headline
    assert "Fully disagree: —" in headline


def test_unreported_label_is_preserved_and_human_validation_is_not_fabricated():
    data = analyze_conditional([{"question_id": "q", "sampled_events": [event(None)]}])
    assert next(r for r in data["B"] if r["self_label"] is None)["n"] == 1
    assert sum(r["text"]["n"] for r in data["C_by_label_turn"]) == 1
    assert data["human_annotation"]["status"] == "not_completed"
    assert data["human_annotation"]["planned_events"] == 60


def test_main_d_tables_never_fall_back_to_unconditioned_model_results():
    from llm_committee.pivot.leaning_results_report import render_probabilities

    summary = {"conditional_analysis": analyze_conditional([]), "probability_audit": [], "low_mass_choice_reads": []}
    page = BeautifulSoup("".join(render_probabilities(summary)), "html.parser")
    for table in page.find_all("table"):
        if "diagnostics" not in table.caption.get_text():
            assert "self-label" in table.caption.get_text().lower()
            assert "self-label" in table.thead.get_text().lower()
    assert "model-wide" not in page.get_text()
