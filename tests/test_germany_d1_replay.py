"""Diagnostic repeats are exact archived requests, not new prompts or debate."""

from llm_committee.pivot.models import canonical
from scripts.forced_feedback_public_history import FORCE, request_identity
from scripts.replay_germany_d1 import load_sources, repeat_request


def test_exact_request_reconstruction_and_no_formal_return_leak():
    _, case, sources = load_sources()
    assert set(sources) == {"before", "after"}
    for name, source in sources.items():
        new = repeat_request(source, name)
        assert new.key != source["request"]["key"]
        assert request_identity(new.document()) == request_identity(source["request"])
        assert new.effort == "none" and new.candidate_labels == tuple("ABCDE")
        strings = "\n".join(m.text for m in new.messages)
        assert case["arms"]["forced"]["return"]["position"] not in strings
        assert case["arms"]["forced"]["return"]["reply"] not in strings
        assert FORCE not in strings
    after = canonical(sources["after"]["request"]["messages"])
    before = canonical(sources["before"]["request"]["messages"])
    assert "incoming_peer_message" in after and "incoming_peer_message" not in before
    assert [m["role"] for m in sources["after"]["request"]["messages"]] == [
        "developer",
        "user",
        "assistant",
        "user",
        "user",
    ]
