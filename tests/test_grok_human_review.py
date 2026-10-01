"""Independent judging reuses exact blinded inputs; human pages cannot decode sources."""

import json
import shutil
import sqlite3
from collections import Counter

from bs4 import BeautifulSoup

from llm_committee.pivot.databricks_provider import databricks_payload
from llm_committee.pivot.models import canonical
from scripts.grok_quality_check import MODEL, SOURCE, prepare, register_provider, request_from, summarize
from scripts.human_answer_review import build
from scripts.report_grok_quality import REPORT, attach
from scripts.scale_public_history import read


def test_frozen_judge_requests_are_identical_except_model_and_journal_key(tmp_path):
    manifest, tasks = prepare(SOURCE, tmp_path / "judge")
    register_provider()
    assert manifest["planned_calls"] == 200 and manifest["planned_pairs"] == 100
    assert Counter(t["assignment"] for t in tasks) == {"original": 100, "alternate": 100}
    assert len({t["question_id"] for t in tasks}) == 50
    with sqlite3.connect(f"file:{SOURCE}/requests.sqlite3?mode=ro", uri=True) as db:
        for t in tasks:
            old = json.loads(
                db.execute(
                    "SELECT request FROM calls WHERE status='completed' AND json_extract(request,'$.key')=?",
                    (t["source_key"],),
                ).fetchone()[0]
            )
            new = dict(t["request"])
            assert new["model"] == MODEL
            new.update(model=old["model"], key=old["key"])
            assert new == old
            assert databricks_payload(request_from(old)) == databricks_payload(request_from(t["request"]))
            assert set(json.loads(new["messages"][-1]["text"])) == {"question", "options", "left", "right"}
    assert prepare(SOURCE, tmp_path / "judge") == (manifest, tasks)


def test_pair_scores_count_two_orders_and_preserve_missingness(tmp_path):
    output = tmp_path / "judge"
    _, tasks = prepare(SOURCE, output)
    with sqlite3.connect(output / "requests.sqlite3") as db:
        db.execute("CREATE TABLE calls(key TEXT, request TEXT, parsed TEXT, status TEXT, charge REAL, error TEXT)")
        for t in tasks[:3]:
            side = t["target_side"] if t["order"] == 0 else ("right" if t["target_side"] == "left" else "left")
            db.execute(
                "INSERT INTO calls VALUES(?,?,?,'completed',0.01,NULL)",
                (t["request"]["key"], canonical(t["request"]), canonical({"preference": side, "evidence": "fixture"})),
            )
    report = summarize(output)
    assert report["completed_calls"] == 3 and report["status"] == "incomplete"
    complete = [r for r in report["rows"] if r["grok_score"] is not None]
    assert len(complete) == 1 and complete[0]["grok_score"] == 0.5
    assert sum(r["complete_pairs"] for r in report["summary"]) == 1
    assert report["cost_accounting"]["received_response_estimate_usd"] == 0.03


def test_human_sample_is_blind_balanced_repeatable_and_outcome_independent(tmp_path):
    output, private = tmp_path / "review.html", tmp_path / "private"
    build(output=output, private=private)
    document = output.read_text()
    soup = BeautifulSoup(document, "html.parser")
    public = json.loads(soup.select_one("#blind-data").string)
    key = read(private / "researcher-key.json")
    assert len(public["items"]) == 20 and len({r["question_id"] for r in key["items"]}) == 20
    assert Counter((r["assignment"], r["debate_side"]) for r in key["items"]) == {
        ("original", "A"): 5,
        ("original", "B"): 5,
        ("alternate", "A"): 5,
        ("alternate", "B"): 5,
    }
    assert all(set(r) == {"id", "question", "options", "answers"} for r in public["items"])
    for row, hidden in zip(public["items"], key["items"], strict=True):
        assert row["id"] == hidden["item_id"]
        record = read(SOURCE / "questions" / f"{hidden['question_id']}.json")
        e2 = next(r for r in record["E2"] if r["assignment"] == hidden["assignment"])
        assert row["answers"][hidden["debate_side"]] == e2["debated"]["answer"]
        assert row["answers"]["B" if hidden["debate_side"] == "A" else "A"] == e2["baseline"]["answer"]
        assert hidden["question_id"] not in document
    for forbidden in ("debate_side", "target_side", "gemini_judgment", "grok_score", "researcher-key", str(SOURCE)):
        assert forbidden not in document
    assert not soup.select("script[src],link[rel=stylesheet],a[href]")
    assert {r["value"] for r in soup.select("[name=preference]")} == {"A", "B", "tie", "unjudgeable"}
    assert "connect-src 'none'" in soup.select_one('meta[http-equiv="Content-Security-Policy"]')["content"]
    build(output=output, private=private)
    assert output.read_text() == document


def test_report_adds_one_judge_column_idempotently(tmp_path):
    output = tmp_path / "judge"
    _, tasks = prepare(SOURCE, output)
    with sqlite3.connect(output / "requests.sqlite3") as db:
        db.execute("CREATE TABLE calls(key TEXT, request TEXT, parsed TEXT, status TEXT, charge REAL, error TEXT)")
        for t in tasks:
            db.execute(
                "INSERT INTO calls VALUES(?,?,?,'completed',0.001,NULL)",
                (
                    t["request"]["key"],
                    canonical(t["request"]),
                    canonical({"preference": t["target_side"], "evidence": "fixture"}),
                ),
            )
    summarize(output)
    page = tmp_path / "report.html"
    for suffix in (".html", ".json"):
        shutil.copyfile(REPORT.with_suffix(suffix), page.with_suffix(suffix))
    for _ in range(2):
        attach(page, output)
        soup = BeautifulSoup(page.read_text(), "html.parser")
        table = soup.select_one("#e2 table")
        assert len(table.select("thead th")) == 7
        assert len(table.select("tbody tr")) == 2
        assert all(len(r.select("td")) == 7 for r in table.select("tbody tr"))
        assert [r.get_text() for r in table.select('tbody [data-judge="grok"]')] == ["100.0% [100.0, 100.0]"] * 2
        assert len(soup.select("#independent-grok-note")) == 1
