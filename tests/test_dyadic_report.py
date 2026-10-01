"""Do not accidentally collapse the two incoming/outgoing agreement dimensions."""

import json

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.dyadic_report import REQUIRED, c_table, conditional_groups, d_table, flatten


def row():
    return {
        "question_id": "q",
        "T": 3,
        "member": 1,
        "previous_peer_self_label": "leaning_disagree",
        "current_self_label": "fully_agree",
        "C_choice_changed": True,
        "C_initial_choice_changed": True,
        "text_label": "conclusion_changed",
        "endpoint_text_label": "conclusion_changed",
        "final_pair": "initial->end",
        "D_control": 6.5,
        "D_argument": 6.0,
        "D_text_delta": -0.5,
        "choice_p_before": 80.0,
        "choice_p_after": 70.0,
        "D_choice_adjacent": -10.0,
    }


@pytest.mark.parametrize(
    "dimensions", [("T",), ("member",), ("current_self_label", "T"), ("previous_peer_self_label", "member")]
)
def test_c_d_require_both_labels(dimensions):
    for render in (c_table, d_table):
        with pytest.raises(ValueError, match="BOTH"):
            render([row()], dimensions)


def test_conditional_tables_show_references_not_just_deltas():
    rows = [row()]
    assert len(conditional_groups(rows, (*REQUIRED, "T"))) == 1
    c = c_table(rows)
    assert "Leaning disagree" in c and "Fully agree" in c and "1/1 (100.0%)" in c
    d = d_table(rows)
    assert "6.500" in d and "6.000" in d and "-0.500" in d
    d = d_table(rows, kind="choice")
    assert "80.000" in d and "70.000" in d and "-10.000" in d


def test_missing_measurement_is_not_zero_change():
    r = {**row(), "C_choice_changed": None, "text_label": None, "D_argument": None}
    assert "0 valid" in c_table([r])
    d = d_table([r])
    assert "0/1" in d and "6.500" not in d and "-0.500" not in d


def test_flatten_preserves_both_label_targets():
    e = {
        **row(),
        "D_text": None,
        "D_choice_before": None,
        "D_choice_after": None,
        "D_choice_initial": None,
        "position_before": None,
        "position_initial": None,
        "C_text": None,
        "C_initial_to_endpoint": None,
    }
    result = flatten([{"question_id": "q", "events": [e]}])[0]
    assert result[REQUIRED[0]] == "leaning_disagree" and result[REQUIRED[1]] == "fully_agree"
    assert result["D_text_delta"] is None


@pytest.mark.parametrize("invalid", [None, "mock", "unfinished", "different-source", "duplicate"])
def test_combining_preserves_main_and_other_supplement(tmp_path, monkeypatch, invalid):
    from llm_committee.pivot import dyadic_report

    manifest = {"kind": "paid_dyadic_supplement", "source": {"manifest_sha256": "main-hash"}}
    report = {"status": "completed"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(dyadic_report, "load_run", lambda source: (manifest, report, []))
    monkeypatch.setattr(
        dyadic_report,
        "render_embedded",
        lambda *args: ('<section id="dyadic"><h2>Two-member</h2></section>', {"rows": []}),
    )
    base = '<html><body><nav></nav><section id="main">Main</section><footer>Footer</footer></body></html>'
    summary = {
        "manifest_sha256": "main-hash",
        "counts": {"questions": 60},
        "supplements": {"forced_feedback": {"case_count": 150}},
    }
    if invalid == "mock":
        manifest["kind"] = "offline_mock"
    elif invalid == "unfinished":
        report["status"] = "preflight_completed"
    elif invalid == "different-source":
        manifest["source"]["manifest_sha256"] = "wrong"
    elif invalid == "duplicate":
        base = base.replace("<footer>", '<section id="dyadic"></section><footer>')
    if invalid:
        with pytest.raises(ValueError):
            dyadic_report.include_dyadic(base, summary, tmp_path)
        return
    document, combined = dyadic_report.include_dyadic(base, summary, tmp_path)
    assert combined["counts"] == summary["counts"]
    assert combined["supplements"]["forced_feedback"] == summary["supplements"]["forced_feedback"]
    assert not combined["supplements"]["dyadic"]["pooled_with_main_study"]
    parsed = BeautifulSoup(document, "html.parser")
    assert len(parsed.select("#dyadic")) == 1 and len(parsed.select("html")) == 1
    assert parsed.select_one("#main").get_text() == "Main"
