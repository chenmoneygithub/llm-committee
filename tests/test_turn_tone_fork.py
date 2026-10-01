"""A shared prefix is reused byte-for-byte, never regenerated as a new replicate."""

import copy
import json
import sqlite3
from collections import Counter
from dataclasses import asdict

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot import prompts, turn_tone
from llm_committee.pivot.agreement import agreement_manifest
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY
from llm_committee.pivot.models import SCREEN_JUDGES, Completion, PilotConfig, Question, Screen, digest
from llm_committee.pivot.probabilities import TOPK_ZERO_FILL_POLICY
from llm_committee.pivot.providers import MockProvider
from llm_committee.pivot.turn_tone_fork import VERSION, ForkGraph, alternate_plans, prepare, reusable_suffixes
from llm_committee.pivot.turn_tone_fork_run import run
from llm_committee.pivot.turn_tone_run import run as run_original


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("fork-source")
    questions = [
        Question(f"q{i:02}", f"Policy {i}?", ("Yes", "No"), Screen(SCREEN_JUDGES, 3, "fixture", "0" * 64))
        for i in range(20)
    ]
    contexts = json.loads(
        json.dumps(
            {
                q.id: {"question": asdict(q), "initial_answers": {str(m): f"INITIAL-{m}" for m in range(3)}}
                for q in questions
            }
        )
    )
    manifest = {
        "kind": "offline_mock",
        "protocol_version": turn_tone.VERSION,
        "implementation_sha256": turn_tone.implementation_hash(),
        "base_prompt_version": prompts.PROMPT_VERSION,
        "config": asdict(PilotConfig(judge_model="gemini-3.8-flash")),
        "source": {"directory": "/nonexistent/source", "contexts_sha256": digest(contexts)},
        "plans": turn_tone.plan_questions(list(contexts)),
        "planned_counts": {"formal_replies": 240},
        "execution": {"budget_policy": "no_limit_user_requested", "failure_policy": FORMAT_RETRY_POLICY},
        "agreement_rubric": agreement_manifest(),
        "probability_readout": TOPK_ZERO_FILL_POLICY,
        "question_selection": {"count": 20},
        "sampling": {},
        "design": {},
    }
    run_original(manifest, contexts, root, question_limit=20)
    return root


def test_alternate_plan_same_prefix_different_suffix_balanced():
    source = turn_tone.plan_questions([f"q{i:02}" for i in range(20)])
    frozen = copy.deepcopy(source)
    plans = alternate_plans(source)
    assert source == frozen and plans == alternate_plans(list(reversed(source)))
    for node in source[0]["tone_schedule"]:
        before = [p["tone_schedule"][node] for p in source]
        after = [p["tone_schedule"][node] for p in plans]
        assert Counter(before) == Counter(after)
        assert all((a == b) == (int(node[-1]) <= 2) for a, b in zip(before, after, strict=True))
    assert all(
        [e["id"] for e in a["events"]] == [e["id"] for e in b["events"]] for a, b in zip(source, plans, strict=True)
    )


def test_reuse_and_current_tone_only(source):
    manifest, contexts = prepare(source, mock=True)
    assert manifest["planned_counts"]["formal_replies"] == 120
    assert manifest["planned_counts"]["total_unique_replies_with_source"] == 360
    assert manifest["planned_counts"]["logical_calls"] + manifest["planned_counts"]["reused_requests"] == 1161
    count = MockProvider().token_count
    for plan in manifest["plans"]:
        graph = ForkGraph(contexts[plan["question_id"]], plan, manifest)
        graph.verify_reuse(count)
        for pair in ("AB", "CA", "BC"):
            task = graph.tasks[graph.key(f"turn_level/debate/{pair}-3")]
            request = task.build(graph.values, count)
            assert (
                request.messages[0].text
                == prompts.BASE
                + "\n"
                + prompts.TONE_TEXT[plan["tone_schedule"][f"{pair}-3"]]
                + "\n"
                + prompts.REPLY_RULE
            )
            original_plan = copy.deepcopy(plan)
            original_plan["tone_schedule"] = plan["original_tone_schedule"]
            original = ForkGraph(contexts[plan["question_id"]], original_plan, manifest)
            original_request = original.tasks[task.key].build(original.values, count)
            assert request.messages[1:] == original_request.messages[1:]
            assert request.messages[0] != original_request.messages[0]
        for event in plan["events"]:
            if event["member"] in (1, 2):
                assert (graph.key(f"Dtext/{event['id']}/argument") in graph.reused_keys) == (event["T"] <= 3)


def test_gate_recovery_never_dispatches_reused_requests(source, tmp_path):
    manifest, contexts = prepare(source, mock=True)
    plan = manifest["plans"][0]
    seen = []

    class Recording(MockProvider):
        def generate(self, request):
            assert request.key.split(f"/{VERSION}/")[1] not in reusable_suffixes(plan)
            seen.append(request.key)
            return super().generate(request)

    report = run(manifest, contexts, tmp_path / "gate", question_limit=1, provider_factory=Recording)
    assert report["status"] == "preflight_completed" and report["trajectory_statuses"] == {"success": 3}
    assert sum("/debate/" in k for k in seen) == 6
    before = list(seen)
    run(manifest, contexts, tmp_path / "gate", question_limit=1, provider_factory=Recording)
    assert seen == before
    record = json.loads((tmp_path / "gate/questions" / f"{plan['question_id']}.json").read_text())
    original = json.loads((source / "questions" / f"{plan['question_id']}.json").read_text())
    for node in original["formal_replies"]["turn_level"]:
        if int(node[-1]) <= 2:
            assert record["formal_replies"]["turn_level"][node] == original["formal_replies"]["turn_level"][node]
    for a, b in zip(record["events"], original["events"], strict=True):
        if a["T"] <= 3:
            assert a["D_text"] == b["D_text"]
        if a["T"] == 2:
            for field in b.keys() - {"current_tone", "previous_tone"}:
                assert a[field] == b[field]


def test_changed_reuse_is_rejected(source):
    manifest, contexts = prepare(source, mock=True)
    plan = manifest["plans"][0]
    context = copy.deepcopy(contexts[plan["question_id"]])
    context["reused_requests"]["turn_level/debate/AB-1"]["parsed"]["reply"] = "CHANGED PREFIX"
    graph = ForkGraph(context, plan, manifest)
    with pytest.raises(ValueError, match="changed request"):
        graph.verify_reuse(MockProvider().token_count)


def test_failed_new_branch_is_isolated(source, tmp_path):
    manifest, contexts = prepare(source, mock=True)
    qid = manifest["plans"][0]["question_id"]
    target = f"{qid}/{VERSION}/turn_level/debate/AB-3"

    class Broken(MockProvider):
        def generate(self, request):
            return Completion("INVALID", 100, 2) if request.key == target else super().generate(request)

    result = run(manifest, contexts, tmp_path / "fail", question_limit=1, provider_factory=Broken)
    assert result["trajectory_statuses"] == {"failed": 1, "success": 2}
    assert not result["fatal_errors"]
    with sqlite3.connect(tmp_path / "fail/requests.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM calls WHERE key LIKE ?", (target + "%",)).fetchone()[0] == 3


@pytest.fixture(scope="module")
def completed_fork(source, tmp_path_factory):
    output = tmp_path_factory.mktemp("fork-completed")
    manifest, contexts = prepare(source, mock=True)
    report = run(manifest, contexts, output, question_limit=20)
    assert report["status"] == "completed" and report["call_status_counts"] == {"completed": 620}
    return output


def test_paired_report_no_duplicated_prefix_or_t3_effect(completed_fork):
    from llm_committee.pivot.turn_tone_fork_report import load_run, render

    m, r, a, b = load_run(completed_fork)
    frozen = digest(a)
    html, summary = render(m, r, a, b)
    soup = BeautifulSoup(html, "html.parser")
    assert digest(a) == frozen
    assert len(summary["rows"]) == 214 and len(summary["paired_events"]) == 107
    assert all(row["T"] >= 3 for row in summary["rows"])
    assert len(soup.select(".fork-case")) == 60 and len(soup.select(".fork-grid")) == 120
    assert "360 unique replies" in soup.get_text()
    assert "D-text at T3 is shared, not a new observation" in soup.get_text()
    assert "OFFLINE MOCK" in soup.get_text()
    for node in soup.select("table"):
        headers = [h.get_text() for h in node.select("thead th")]
        assert all(len(tr.select("td")) == len(headers) for tr in node.select("tbody tr"))
        if "Option changed / valid" in headers or "Valid pairs / sampled events" in headers:
            assert {"Peer's preceding self-label", "Receiver's current self-label", "Turn T", "assignment"} <= set(
                headers
            )
        if "Own-text score: filler" in headers:
            t_index = headers.index("Turn T")
            assert all(tr.select("td")[t_index].get_text() == "4" for tr in node.select("tbody tr"))


def test_embedding_keeps_original_tables_and_rejects_wrong_cohort(completed_fork, monkeypatch):
    from llm_committee.pivot import turn_tone_fork_report as module

    loaded = module.load_run(completed_fork)
    m, r, a, b = loaded
    base = '<html><head></head><body><h1>Pilot</h1><nav></nav><table id="old"><tr><td>original</td></tr></table><footer>End</footer></body></html>'
    summary = {
        "source_directory": m["source"]["directory"],
        "manifest_sha256": m["source"]["manifest_sha256"],
        "rows": ["original metrics"],
    }
    with pytest.raises(ValueError, match="mock"):
        module.include_fork(base, summary, completed_fork)
    # Exercise publication guards with synthetic fixtures only, never write a report.
    monkeypatch.setattr(module, "load_run", lambda source: ({**m, "kind": "paid_turn_tone_fork"}, r, a, b))
    document, combined = module.include_fork(base, summary, completed_fork)
    assert all(combined[k] == v for k, v in summary.items())
    soup = BeautifulSoup(document, "html.parser")
    assert len(soup.select("html")) == 1 and len(soup.select("#paired")) == 1
    assert str(soup.select_one("#old")) == str(BeautifulSoup(base, "html.parser").select_one("#old"))
    with pytest.raises(ValueError, match="already included"):
        module.include_fork(document, combined, completed_fork)
    with pytest.raises(ValueError, match="different pilot"):
        module.include_fork(base, {**summary, "manifest_sha256": "wrong"}, completed_fork)
