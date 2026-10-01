"""Fresh strong-label generations retain the frozen paired experimental design."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from llm_committee.pivot import prompts, strong_prompts
from llm_committee.pivot.models import Completion, canonical
from llm_committee.pivot.strong_agreement import AGREEMENT, AGREEMENT_RUBRIC_TEXT
from llm_committee.pivot.strong_run import run
from llm_committee.pivot.strong_study import StrongGraph, StrongMockProvider, prepare

PLAN = Path(__file__).resolve().parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"


def test_frozen_design_and_fresh_outputs():
    manifest, contexts = prepare(PLAN, mock=True)
    design = json.loads(PLAN.read_text())
    assert manifest["planned_counts"]["by_purpose"] == {
        "initial": 60,
        "debate": 360,
        "position": 387,
        "judge_b": 267,
        "judge_c": 481,
        "d_text": 286,
    }
    assert manifest["planned_counts"]["logical_calls"] == 1841
    for item, plan in zip(design["plans"], manifest["plans"], strict=True):
        assert contexts[plan["question_id"]] == {"question": item["question"]}
        graph = StrongGraph(contexts[plan["question_id"]], plan, manifest)
        assert not graph.values
        for arm in ("original", "alternate"):
            assert plan["arms"][arm]["tone_schedule"] == item[f"{arm}_tones"]
            assert [e["node_id"] for e in plan["arms"][arm]["events"]] == item["sampled_node_ids"]
        for pair in ("AB", "CA", "BC"):
            for t in (1, 2, 3, 4):
                suffix = f"turn_level/debate/{pair}-{t}"
                assert (graph.key("original", suffix) == graph.key("alternate", suffix)) == (t <= 2)
        for e in plan["arms"]["original"]["events"]:
            if e["member"] in (1, 2):
                suffix = f"Dtext/{e['id']}/argument"
                assert (graph.key("original", suffix) == graph.key("alternate", suffix)) == (e["T"] <= 3)


def test_rubric_rejects_legacy_labels():
    for label in AGREEMENT:
        assert prompts.parse_json(canonical({"reply": "test", "agreement": label}), strong_prompts.REPLY_SCHEMA)
    for label in ("fully_agree", "fully_disagree"):
        with pytest.raises(ValueError):
            prompts.parse_json(canonical({"reply": "test", "agreement": label}), strong_prompts.REPLY_SCHEMA)
    assert "fully_agree" in canonical(prompts.REPLY_SCHEMA)  # Archived protocol unchanged.
    assert "strongly_agree" not in canonical(prompts.REPLY_SCHEMA)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    manifest, contexts = prepare(PLAN, mock=True)
    output = tmp_path_factory.mktemp("strong-completed")
    report = run(manifest, contexts, output, question_limit=20)
    assert report["status"] == "completed"
    assert report["call_status_counts"] == {"completed": 1841}
    assert report["trajectory_statuses"] == {"success": 120}
    return output, manifest, contexts


def test_contexts_neutral_probes_and_paired_requests(completed):
    output, manifest, contexts = completed
    with sqlite3.connect(output / "requests.sqlite3") as db:
        calls = {json.loads(r)["key"]: json.loads(r) for (r,) in db.execute("SELECT request FROM calls")}
    count = Counter()
    for req in calls.values():
        purpose, messages = req["purpose"], req["messages"]
        count[purpose] += 1
        if purpose in ("debate", "judge_b"):
            assert AGREEMENT_RUBRIC_TEXT in messages[0]["text"]
            assert "fully_agree" not in canonical(req)
        if purpose in ("position", "d_text"):
            assert messages[0]["text"] == prompts.BASE + "\n"
            assert req["effort"] == "none"
        if purpose in ("debate", "position", "d_text"):
            for message in messages[1:]:
                assert all(prompts.TONE_TEXT[t] not in message["text"] for t in ("friendly", "hostile"))
                assert "SYNTHETIC evidence" not in message["text"]
                assert '"agreement":' not in message["text"] or message == messages[-1]
    assert dict(count) == manifest["planned_counts"]["by_purpose"]
    for plan in manifest["plans"]:
        graph = StrongGraph(contexts[plan["question_id"]], plan, manifest)
        for pair in ("AB", "CA", "BC"):
            a, b = [calls[graph.key(arm, f"turn_level/debate/{pair}-3")] for arm in ("original", "alternate")]
            assert a["messages"][1:] == b["messages"][1:]
            assert a["messages"][0] != b["messages"][0]
        record = json.loads((output / "questions" / f"{plan['question_id']}.json").read_text())
        a, b = (record["branches"][arm] for arm in ("original", "alternate"))
        for first, second in zip(a["events"], b["events"], strict=True):
            if first["T"] <= 3:
                assert first["D_text"] == second["D_text"]
            if first["T"] == 2:
                assert first == second


def test_resume_and_failure_isolation(tmp_path):
    manifest, contexts = prepare(PLAN, mock=True)
    graph = StrongGraph(contexts[manifest["plans"][0]["question_id"]], manifest["plans"][0], manifest)
    target = graph.key("alternate", "turn_level/debate/AB-3")
    seen = []

    class Broken(StrongMockProvider):
        def generate(self, request):
            seen.append(request.key)
            return Completion("INVALID", 100, 2) if request.key == target else super().generate(request)

    result = run(manifest, contexts, tmp_path / "fail", question_limit=1, provider_factory=Broken)
    assert result["trajectory_statuses"] == {"success": 5, "failed": 1}
    assert not result["fatal_errors"]
    assert seen.count(target) == 3
    frozen = list(seen)
    run(manifest, contexts, tmp_path / "fail", question_limit=1, provider_factory=Broken)
    assert seen == frozen


def test_report_unique_counts_conditioning_and_no_legacy_conversion(completed):
    from bs4 import BeautifulSoup

    from llm_committee.pivot.strong_report import load_run, render

    output, _, _ = completed
    manifest, report, records = load_run(output)
    document, summary = render(manifest, report, records, legacy_href="legacy.html", supplementary_href="fixed.html")
    soup = BeautifulSoup(document, "html.parser")
    assert len(summary["unique_formal_replies"]) == 360
    assert Counter(r["current_tone"] for r in summary["unique_formal_replies"]) == {
        "friendly": 120,
        "neutral": 120,
        "hostile": 120,
    }
    assert len(summary["rows"]) == 267
    assert sum(r["assignment"] == "Shared T1/T2" for r in summary["rows"]) == 53
    assert sum(r["assignment"] == "Alternate" for r in summary["rows"]) == 107
    assert not any(r["assignment"] == "Alternate" and r["T"] == 3 for r in summary["D_text_counted_rows"])
    assert len(soup.select(".fork-case")) == 60
    assert len(soup.select(".fork-grid")) == 120
    assert "OFFLINE MOCK" in soup.get_text()
    assert "fully_agree" not in document and "Fully agree" not in document
    assert "Strongly agree" in document and "Strongly disagree" in document
    c_tables = soup.select("#c table")
    assert len(c_tables) == 2  # Adjacent changes plus a collapsed initial-to-final view.
    for node in c_tables:
        assert [h.get_text() for h in node.select("thead th")] == [
            "Peer's preceding label",
            "Receiver's current label",
            "Replies",
            "Option changed",
            "Reasons adjusted",
            "Conclusion changed",
        ]
        assert len(node.select("tbody tr")) <= 16
    assert sum(g["replies"] for g in summary["C_by_label_pair"]) == 267
    assert sum(g["replies"] for g in summary["C_initial_to_endpoint_by_label_pair"]) == sum(
        bool(r["final_pair"]) for r in summary["rows"]
    )
    for g in summary["C_by_label_pair"]:
        expected = [
            r
            for r in summary["rows"]
            if r["previous_peer_self_label"] == g["previous_peer_self_label"]
            and r["current_self_label"] == g["current_self_label"]
        ]
        assert g["replies"] == len(expected)
        assert g["option_changed"] == sum(r["C_choice_changed"] is True for r in expected)
        assert g["text_unchanged"] + g["reasons_adjusted"] + g["conclusion_changed"] == g["text_compared"]
    assert soup.select_one('#c a[href="#cases"]')
    assert "Text unjudgeable" not in soup.select_one("#c").get_text()
    assert " / valid" not in soup.select_one("#c").get_text()
    d_tables = soup.select("#d table")
    assert len(d_tables) == 6  # Two models each for D1, D2 and the initial-reference view.
    for node in d_tables:
        headers = [h.get_text() for h in node.select("thead th")]
        assert len(headers) == 5
        assert headers[:3] == ["Peer's preceding label", "Receiver's current label", "Sample"]
        assert len(node.select("tbody tr")) <= 16
        assert not {"Turn T", "assignment", "Receiver model", "Valid pairs / sampled events"} & set(headers)
    for key, rows in (
        ("D_choice_by_label_pair_model", summary["rows"]),
        ("D_text_by_label_pair_model", summary["D_text_counted_rows"]),
        ("D_initial_by_label_pair_model", summary["rows"]),
    ):
        groups = summary[key]
        assert sum(g["sampled"] for g in groups) == sum(r["member"] in (1, 2) for r in rows)
        for g in groups:
            assert g["member"] in (1, 2)
            expected = [
                r
                for r in rows
                if all(r[k] == g[k] for k in ("member", "previous_peer_self_label", "current_self_label"))
            ]
            assert g["sampled"] == len(expected)
            if g["compared"]:
                assert g["change_mean"] == pytest.approx(g["updated_mean"] - g["reference_mean"])
    d_text = soup.select_one("#d").get_text()
    assert "filler versus peer, not before versus after a reply" in d_text
    assert "give each represented question equal weight" in d_text
    assert "Change (percentage points)" in d_text
    assert "100.00% is not proof of absolute certainty" in d_text
    assert soup.select_one('#d a[href="#cases"]')
    for node in soup.select("table"):
        headers = [h.get_text() for h in node.select("thead th")]
        assert all(len(tr.select("td")) == len(headers) for tr in node.select("tbody tr"))


def test_compact_c_uses_reply_counts_and_retains_missing_denominators():
    from copy import deepcopy

    from bs4 import BeautifulSoup

    from llm_committee.pivot.strong_report import c_summary, compact_c_table

    rows = [
        {
            "previous_peer_self_label": "strongly_agree",
            "current_self_label": "leaning_agree",
            "question_id": question,
            "T": turn,
            "assignment": assignment,
            "C_choice_changed": choice,
            "text_label": text,
            "final_pair": final,
            "C_initial_choice_changed": initial_choice,
            "endpoint_text_label": endpoint,
        }
        for question, turn, assignment, choice, text, final, initial_choice, endpoint in (
            ("q1", 2, "Shared T1/T2", True, "adjusted", False, None, None),
            ("q1", 3, "Original", False, "unchanged", True, True, "conclusion_changed"),
            ("q1", 4, "Alternate", False, "conclusion_changed", True, False, "unchanged"),
            ("q2", 3, "Original", None, "unjudgeable", True, None, None),
            ("q3", 4, "Original", False, None, True, False, "unjudgeable"),
        )
    ]
    original = deepcopy(rows)
    groups = c_summary(rows)
    assert groups == [
        {
            "previous_peer_self_label": "strongly_agree",
            "current_self_label": "leaning_agree",
            "questions": 3,
            "replies": 5,
            "option_compared": 4,
            "option_changed": 1,
            "text_compared": 3,
            "text_unchanged": 1,
            "reasons_adjusted": 1,
            "conclusion_changed": 1,
            "option_excluded": 1,
            "text_excluded": 2,
        }
    ]
    soup = BeautifulSoup(compact_c_table(groups), "html.parser")
    assert [td.get_text() for td in soup.select("tbody td")] == [
        "Strongly agree",
        "Leaning agree",
        "5",
        "1/4 (25.0%)",
        "1/3 (33.3%)",
        "1/3 (33.3%)",
    ]
    assert "1 option comparison(s)" in soup.get_text()
    assert "2 text comparison(s)" in soup.get_text()
    assert "not counted as unchanged" in soup.get_text()
    endpoint = c_summary(rows, endpoint=True)[0]
    assert endpoint["replies"] == 4
    assert endpoint["option_compared"] == 3
    assert endpoint["option_changed"] == 1
    assert endpoint["text_compared"] == 2
    assert endpoint["reasons_adjusted"] == 0
    assert endpoint["conclusion_changed"] == 1
    assert rows == original


def test_compact_c_does_not_turn_unavailable_measurements_into_zero():
    from bs4 import BeautifulSoup

    from llm_committee.pivot.strong_report import c_summary, compact_c_table

    groups = c_summary(
        [
            {
                "previous_peer_self_label": None,
                "current_self_label": "strongly_disagree",
                "question_id": "q1",
                "C_choice_changed": None,
                "text_label": "unjudgeable",
            }
        ]
    )
    soup = BeautifulSoup(compact_c_table(groups), "html.parser")
    assert [td.get_text() for td in soup.select("tbody td")] == [
        "Not reported",
        "Strongly disagree",
        "1",
        "—",
        "—",
        "—",
    ]
    assert "0.0%" not in soup.get_text()
    assert "valid" not in soup.get_text()


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("choice", ("choice_p_before", "choice_p_after", "D_choice_adjacent")),
        ("text", ("D_control", "D_argument", "D_text_delta")),
        ("initial", ("initial_p_before", "initial_p_after", "D_choice_initial_to_current")),
    ],
)
def test_compact_d_reweights_questions_and_keeps_models_separate(kind, fields):
    from copy import deepcopy

    from bs4 import BeautifulSoup

    from llm_committee.pivot.strong_report import compact_d_tables, d_summary

    rows = []
    scale = 20 if kind == "text" else 1
    for question, member, t, assignment, before, after, changed in (
        ("q1", 1, 3, "Original", 90, 80, False),
        ("q1", 1, 4, "Alternate", 90, 70, False),
        ("q2", 1, 2, "Shared T1/T2", 100, 60, True),
        ("q3", 1, 4, "Original", 10, None, None),
        ("q1", 2, 3, "Original", 80, 85, False),
        ("q1", 0, 3, "Original", 80, 90, False),
    ):
        rows.append(
            {
                "question_id": question,
                "member": member,
                "previous_peer_self_label": "leaning_disagree",
                "current_self_label": "leaning_agree",
                "T": t,
                "assignment": assignment,
                "C_choice_changed": changed,
                fields[0]: before / scale,
                fields[1]: after / scale if after is not None else None,
                fields[2]: (after - before) / scale if after is not None else None,
            }
        )
    original = deepcopy(rows)
    groups = d_summary(rows, kind=kind)
    assert len(groups) == 2
    first, second = groups
    assert first["sampled"] == 4 and first["compared"] == 3
    assert first["questions"] == 2 and first["excluded"] == 1
    assert first["reference_mean"] == pytest.approx(95 / scale)
    assert first["updated_mean"] == pytest.approx(67.5 / scale)
    assert first["change_mean"] == pytest.approx(-27.5 / scale)
    assert first["adjacent_choice_changed"] == 1
    assert first["adjacent_choice_compared"] == 3
    assert second["member"] == 2 and second["change_mean"] == pytest.approx(5 / scale)
    assert rows == original
    soup = BeautifulSoup(compact_d_tables(groups, kind=kind), "html.parser")
    assert [h.get_text() for h in soup.select("h4")] == ["Qwen3.8-27B", "Inkling"]
    assert "3 comparisons (2 questions)" in soup.get_text()
    assert "1 comparison (1 question)" in soup.get_text()
    assert "1 sampled comparison(s)" in soup.get_text()
    assert "not counted as zero change" in soup.get_text()
    if kind != "text":
        assert "95.00% → 67.50%" in soup.get_text()
        assert "−27.50" in soup.get_text()


def test_compact_d_missing_pairs_and_rounding_are_not_exact_zero():
    from bs4 import BeautifulSoup

    from llm_committee.pivot.strong_report import compact_d_tables, d_number, d_summary

    groups = d_summary(
        [
            {
                "member": 1,
                "question_id": "q1",
                "previous_peer_self_label": "strongly_agree",
                "current_self_label": "strongly_agree",
                "choice_p_before": 99.0,
                "choice_p_after": None,
                "D_choice_adjacent": None,
                "C_choice_changed": None,
            }
        ],
        kind="choice",
    )
    assert groups[0]["compared"] == 0
    assert groups[0]["reference_mean"] is None
    assert groups[0]["updated_mean"] is None
    assert groups[0]["change_mean"] is None
    soup = BeautifulSoup(compact_d_tables(groups, kind="choice"), "html.parser")
    assert [td.get_text() for td in soup.select("tbody td")][-2:] == ["—", "—"]
    assert d_number(-0.001, signed=True) == "≈0.00"
    assert d_number(0.001, signed=True) == "≈0.00"
    assert d_number(-0.01, signed=True) == "−0.01"
    assert d_number(1.01, signed=True) == "+1.01"


def test_publish_archives_both_files_exactly(completed, tmp_path, monkeypatch):
    from llm_committee.pivot import strong_report

    output, _, _ = completed
    manifest, report, records = strong_report.load_run(output)
    with pytest.raises(ValueError, match="mock"):
        # Test the mock guard after other path preconditions.
        (tmp_path / "fixed.html").write_text("fixed")
        strong_report.publish(output, tmp_path / "main.html", tmp_path / "legacy.html", tmp_path / "fixed.html")
    main, archive = tmp_path / "main.html", tmp_path / "legacy.html"
    main.write_text("EXACT OLD FULLY HTML\n")
    main.with_suffix(".json").write_text('{"schema":"turn_tone_dyadic_pilot_report_v1"}\n')
    old = {ext: main.with_suffix(ext).read_bytes() for ext in (".html", ".json")}
    monkeypatch.setattr(
        strong_report, "load_run", lambda source: ({**manifest, "kind": "paid_strong_dyadic_pilot"}, report, records)
    )
    strong_report.publish(output, main, archive, tmp_path / "fixed.html")
    assert all(archive.with_suffix(ext).read_bytes() == content for ext, content in old.items())
    assert "strongly_agree" in main.with_suffix(".json").read_text()
    strong_report.publish(output, main, archive, tmp_path / "fixed.html")
    assert all(archive.with_suffix(ext).read_bytes() == content for ext, content in old.items())
