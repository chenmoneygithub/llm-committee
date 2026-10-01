"""A single offline page, without pooling cohorts or nesting whole HTML documents."""

import json

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot import forced_feedback_report
from llm_committee.pivot.leaning_results_report import include_feedback


def test_embedded_report_has_no_document_wrapper_or_global_styles():
    manifest = {"kind": "offline_mock"}
    report = {"status": "completed", "successful_cases": 0, "completed_cases": 0, "charged_or_reserved_usd": 0}
    fragment, data = forced_feedback_report.render_embedded(manifest, report, [])
    parsed = BeautifulSoup(fragment, "html.parser")
    assert parsed.select_one("section#forced-feedback h2")
    assert not parsed.find(["html", "head", "body", "style", "h1", "title"])
    assert "OFFLINE MOCK" in parsed.get_text()
    assert data["case_count"] == 0


@pytest.fixture
def supplement(tmp_path, monkeypatch):
    manifest = {"kind": "paid_forced_feedback_supplement", "source": {"manifest_sha256": "main-hash"}}
    report = {"status": "completed", "completed_cases": 150}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(forced_feedback_report, "load_run", lambda source: (manifest, report, []))
    monkeypatch.setattr(
        forced_feedback_report,
        "render_embedded",
        lambda *args: ('<section id="forced-feedback"><h2>Supplement</h2></section>', {"case_count": 150}),
    )
    base = (
        "<!doctype html><html><head><style>body{margin:0}</style></head><body><main>"
        '<header><h1>Main results</h1></header><nav><a href="#a">A</a></nav>'
        '<section id="a"><h2>A</h2><table><tr><td>Original score</td></tr></table></section>'
        "<footer>Original footer</footer></main></body></html>"
    )
    summary = {"manifest_sha256": "main-hash", "counts": {"questions": 60}, "A": ["original"]}
    return tmp_path, manifest, report, base, summary


def test_one_page_preserves_original_sections_and_separates_summary_data(supplement):
    path, _, _, base, summary = supplement
    document, combined = include_feedback(base, summary, path)
    old, new = BeautifulSoup(base, "html.parser"), BeautifulSoup(document, "html.parser")
    assert str(new.select_one("#a")) == str(old.select_one("#a"))
    assert new.footer.get_text() == old.footer.get_text()
    assert len(new.find_all("html")) == len(new.find_all("h1")) == 1
    assert len(new.select("#forced-feedback")) == 1
    assert new.select_one('nav a[href="#forced-feedback"]')
    assert not new.find(["iframe", "script"])
    assert "not pooled into the main A–E scores" in new.get_text()
    assert {k: v for k, v in combined.items() if k != "supplements"} == summary
    assert "supplements" not in summary
    assert combined["supplements"]["forced_feedback"]["pooled_with_main_study"] is False


@pytest.mark.parametrize("invalid", ["different-source", "mock", "unfinished", "already-embedded"])
def test_incompatible_or_duplicate_supplements_are_rejected(supplement, invalid):
    path, manifest, report, base, summary = supplement
    if invalid == "different-source":
        manifest["source"]["manifest_sha256"] = "different"
    elif invalid == "mock":
        manifest["kind"] = "offline_mock"
    elif invalid == "unfinished":
        report["status"] = "preflight_completed"
    else:
        base = base.replace("<footer>", '<section id="forced-feedback"></section><footer>')
    with pytest.raises(ValueError):
        include_feedback(base, summary, path)
