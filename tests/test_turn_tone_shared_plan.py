"""The cross-roster template preserves design, never a previous roster's outputs."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from llm_committee.pivot.dyadic import make_route
from llm_committee.pivot.models import TONES, digest

PLAN = Path(__file__).parents[1] / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"


def load():
    return json.loads(PLAN.read_text())


def test_frozen_template_integrity_and_question_identity():
    data = load()
    expected = data.pop("design_sha256")
    assert digest(data) == expected
    assert len(data["plans"]) == len({p["question"]["id"] for p in data["plans"]}) == 20
    assert data["route_template"] == json.loads(json.dumps([asdict(n) for n in make_route().nodes]))
    assert data["scope"]["unique_debate_replies_per_roster"] == 20 * 3 * (2 + 2 + 2)


def test_same_prefix_changed_suffix_and_frozen_balancing():
    data = load()
    original, alternate = Counter(), Counter()
    for plan in data["plans"]:
        for nid, tone in plan["original_tones"].items():
            other = plan["alternate_tones"][nid]
            t = int(nid.rsplit("-", 1)[1])
            assert (tone == other) == (t <= 2)
            original[tone] += 1
            if t >= 3:
                alternate[other] += 1
    assert original == dict.fromkeys(TONES, 80)
    assert alternate == dict.fromkeys(TONES, 40)
    for node in data["route_template"]:
        nid = node["id"]
        assert Counter(p["original_tones"][nid] for p in data["plans"]) == Counter(
            p["alternate_tones"][nid] for p in data["plans"]
        )


def test_same_sample_and_position_references():
    data = load()
    route = make_route()
    shared = new = 0
    for plan in data["plans"]:
        nodes = plan["sampled_node_ids"]
        assert len(nodes) == len(set(nodes)) == 8
        assert plan["omitted_node_id"] not in nodes
        assert all(route.get(n).depth in (2, 3, 4) for n in nodes)
        shared += sum(route.get(n).depth == 2 for n in nodes)
        new += sum(route.get(n).depth >= 3 for n in nodes)
        for nid in nodes:
            n = route.get(nid)
            before = route.previous_own(nid)
            ref = data["reference_template"][nid]
            assert ref["previous_own_node"] == (before.id if before else None)
            assert ref["initial_member"] == n.receiver
            assert ref["compare_initial_to_endpoint"] == (n.depth >= 3)
    assert (shared, new) == (53, 107)


def test_no_generated_outputs_or_implicit_new_execution():
    data = load()
    forbidden = {"initial_answers", "formal_replies", "parsed", "_readout", "reused_requests"}

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden & value.keys()
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(data)
    for roster in ("same_model", "same_family"):
        assert data["existing_roster_slot_mappings"][roster]["status"] == "planned_not_started"
        assert not data["existing_roster_slot_mappings"][roster]["D_enabled"]
    assert data["existing_roster_slot_mappings"]["mixed_family"]["D_enabled"]
    assert not data["scope"]["three_member_run"] and not data["scope"]["layer_E"]
