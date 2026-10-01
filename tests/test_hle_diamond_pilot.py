"""Synthetic-only verification: protected benchmark text is not test data."""

import copy
from collections import Counter
from dataclasses import replace

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.prompts import TONE_TEXT
from llm_committee.pivot.strong_run import run
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import SuperGPQAMockProvider
from scripts.hle_diamond_pilot import audit, publish
from scripts.hle_diamond_study import ARMS, PairedGraph, parse_options, prepare


@pytest.fixture
def prepared(tmp_path):
    from scripts.hle_diamond_study import TEMPLATE, read

    template = read(TEMPLATE)
    template["plans"] = template["plans"][:1]
    template["design_sha256"] = digest({k: v for k, v in template.items() if k != "design_sha256"})
    t = tmp_path / "plan/tones.json"
    b = tmp_path / "plan/bank.json"
    atomic_json(t, template)
    atomic_json(
        b,
        {
            "dataset": "SYNTHETIC",
            "revision": "SYNTHETIC",
            "selected": [
                {
                    "id": "synthetic-only",
                    "question": "Which number is prime?\nA. 4\nB. 7\nC. 9",
                    "answer": "B",
                    "category": "Math",
                }
            ],
        },
    )
    return prepare(b, mock=True, template_file=t)


def test_parser_preserves_native_order():
    raw = "Stem\nwith two lines.\nA. choice one\ncontinued\nB. choice two\nC. choice three"
    assert parse_options(raw) == ("Stem\nwith two lines.", ("choice one\ncontinued", "choice two", "choice three"))
    with pytest.raises(ValueError):
        parse_options("No native choices; do not invent them")
    with pytest.raises(ValueError):
        parse_options("Stem\nA. first\nC. missing B")


def execute(graph, provider):
    while not graph.complete:
        task = graph.ready()
        request = task.build(graph.values, lambda *_: 0)
        response = provider.generate(request)
        parsed = task.parse(response.text)
        if request.candidate_labels:
            parsed["_readout"] = response.readout
        graph.accept(task, request, parsed)


def test_counts_exact_goqa_tones_and_sharing(prepared):
    m, c = prepared
    assert m["planned_counts"]["by_purpose"] == {"initial": 3, "debate": 18, "d_choice": 12, "synthesis": 9}
    assert m["planned_counts"]["logical_calls"] == 42
    p = m["plans"][0]
    g = PairedGraph(c[p["question_id"]], p, m)
    provider = SuperGPQAMockProvider()
    execute(g, provider)
    assert len(provider.calls) == 42
    for node in g.route.nodes:
        for prefix, limit in [("debate", 2), ("D1", 3)]:
            assert (g.key("original", f"{prefix}/{node.id}") == g.key("alternate", f"{prefix}/{node.id}")) == (
                node.depth <= limit
            )
        for arm in ARMS:
            request = g.tasks[g.key(arm, f"debate/{node.id}")].build(g.values, lambda *_: 0)
            instruction = TONE_TEXT[p["tones"][arm][node.id]]
            if instruction:
                assert instruction in request.messages[-1].text
                assert instruction not in canonical(request.document()["messages"][:-1])
            d = g.tasks.get(g.key(arm, f"D1/{node.id}"))
            if d:
                probe = d.build(g.values, lambda *_: 0)
                assert request.messages[:-1] == probe.messages[:-1]
                assert probe.effort == "none"
    record = g.report(None)
    assert len(record["events"]) == 18 and len(record["endpoints"]) == 12 and len(record["quality"]) == 6
    assert Counter(e["T"] for e in record["events"]) == {1: 3, 2: 3, 3: 6, 4: 6}
    assert len({e["request_key"] for e in record["events"]}) == 18


def test_key_and_private_outputs_never_change_request(prepared):
    m, c = prepared
    p = m["plans"][0]
    g = PairedGraph(c[p["question_id"]], p, m)
    alt = {**c[p["question_id"]], "answer_letter": "C", "original_question": "EVAL_ONLY_SECRET"}
    other = PairedGraph(alt, p, m)
    execute(g, SuperGPQAMockProvider())
    changed = copy.deepcopy(g.values)
    for k, v in changed.items():
        if "/debate/" in k:
            v.update(position="PRIVATE_SENTINEL", choice="C", agreement="strongly_disagree")
    for key, task in g.tasks.items():
        original = task.build(g.values, lambda *_: 0)
        assert original == other.tasks[key].build(g.values, lambda *_: 0)
        assert original == task.build(changed, lambda *_: 0)
        assert "PRIVATE_SENTINEL" not in canonical(original.document())


def test_native_tone_change_does_not_change_shared_d1(prepared):
    m, c = prepared
    p = m["plans"][0]
    g = PairedGraph(c[p["question_id"]], p, m)
    execute(g, SuperGPQAMockProvider())
    # Independently reconstruct each arm, including the builders omitted by key de-duplication.
    for node in g.route.nodes:
        for arm in ARMS:
            base = g.bases[arm]
            mapping = g.maps[arm]
            local = {old: g.values[new] for old, new in mapping.items()}
            key = base.key(f"D1/{node.id}")
            if key in base.tasks:
                direct = base.tasks[key].build(local, lambda *_: 0)
                composed = g.tasks[mapping[key]].build(g.values, lambda *_: 0)
                assert replace(direct, key=mapping[key]) == composed


def test_mock_resume_audit_and_private_html(prepared, tmp_path):
    m, c = prepared
    output = tmp_path / "live"
    provider = SuperGPQAMockProvider()
    kwargs = {
        "question_limit": 1,
        "request_limit": 8,
        "workers": 1,
        "graph_type": PairedGraph,
        "mock_provider_type": SuperGPQAMockProvider,
        "provider_factory": lambda: provider,
        "token_count": lambda *_: 0,
    }
    result = run(m, c, output, **kwargs)
    assert result["status"] == "completed"
    assert audit(output)["requests_reconstructed"] == 42
    run(m, c, output, **kwargs)
    assert len(provider.calls) == 42
    page = tmp_path / "report.html"
    result = publish(output, page)
    assert result["coverage"]["replies"] == 18
    soup = BeautifulSoup(page.read_text(), "html.parser")
    assert "请勿发布" in soup.get_text()
    for table in soup.select("table"):
        width = len(table.select("thead th"))
        assert all(len(tr.select("td")) == width for tr in table.select("tbody tr"))
    for a in soup.select("nav a"):
        assert soup.find(id=a["href"][1:])


def test_failure_blocks_only_dependent_pairs(prepared, tmp_path):
    from llm_committee.pivot.models import Completion

    class BrokenInkling(SuperGPQAMockProvider):
        def generate(self, request):
            if request.purpose == "initial" and request.key.endswith("initial/2"):
                return Completion("not JSON", 10, 10, raw={"mock": True})
            return super().generate(request)

    manifest, contexts = prepared
    output = tmp_path / "failed-live"
    result = run(
        manifest,
        contexts,
        output,
        question_limit=1,
        request_limit=8,
        workers=1,
        graph_type=PairedGraph,
        mock_provider_type=BrokenInkling,
        token_count=lambda *_: 0,
    )
    assert result["status"] == "completed_with_failures"
    assert audit(output)["questions_reconstructed"] == 1
    summary = publish(output, tmp_path / "failed.html")
    assert summary["coverage"]["complete_trajectories"] == 2
    assert summary["coverage"]["replies"] == 6
    assert summary["coverage"]["failed_tasks"] == 1
    assert next(r for r in summary["models"] if r["model"] == "thinkingmachines/Inkling")["initial_n"] == 0
