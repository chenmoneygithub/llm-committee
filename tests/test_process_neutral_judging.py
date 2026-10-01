"""The E2 rubric addition must not alter answers or leak reviewer preferences."""

from collections import Counter
from dataclasses import replace

import pytest

from llm_committee.pivot.models import Message
from scripts.grok_quality_check import OUTPUT as PREVIOUS
from scripts.grok_quality_check import request_from
from scripts.prepare_process_neutral_judging import ADDENDUM, VERSION, prepare, revised_request
from scripts.scale_public_history import read, sha


def test_revision_only_appends_developer_instruction_and_namespaces_key():
    old = request_from(read(PREVIOUS / "tasks.json")[0]["request"])
    new = revised_request(old)
    assert new.key == f"{VERSION}/{old.key}"
    assert new.messages[0] == Message("developer", old.messages[0].text + "\n\n" + ADDENDUM)
    assert new.messages[1:] == old.messages[1:]
    assert replace(new, key=old.key, messages=old.messages) == old
    assert "Do not treat agreement among contributors as evidence" not in new.messages[0].text
    with pytest.raises(ValueError, match="already"):
        revised_request(new)
    with pytest.raises(ValueError, match="already"):
        revised_request(replace(new, key=old.key))
    with pytest.raises(ValueError, match="only"):
        revised_request(replace(old, purpose="synthesis"))
    with pytest.raises(ValueError, match="developer"):
        revised_request(replace(old, messages=(Message("user", "not a rubric"),)))


def test_offline_preparation_preserves_every_input_and_both_models(tmp_path):
    previous_files = ("manifest.json", "tasks.json", "report.json", "requests.sqlite3")
    before = {name: sha(PREVIOUS / name) for name in previous_files}
    output = tmp_path / "revision"
    manifest, tasks = prepare(output=output)
    assert manifest["status"] == "prepared_not_dispatched"
    assert manifest["questions"] == 50 and manifest["answer_pairs"] == 100
    assert manifest["planned_calls_if_executed"] == len(tasks) == 400
    assert Counter(t["request"]["model"] for t in tasks) == {"gemini-3.8-flash": 200, "grok-4-6": 200}
    previous = {(t["question_id"], t["assignment"], t["order"]): t for t in read(PREVIOUS / "tasks.json")}
    for task in tasks:
        old = previous[task["question_id"], task["assignment"], task["order"]]
        new = task["request"]
        assert new["messages"][1:] == old["request"]["messages"][1:]
        assert new["messages"][0]["text"] == old["request"]["messages"][0]["text"] + "\n\n" + ADDENDUM
        assert task["target_side"] == old["target_side"]
        assert new["purpose"] == "judge_e"
        assert new["schema"] == old["request"]["schema"]
        assert new["effort"] == old["request"]["effort"]
        assert new["max_output_tokens"] == old["request"]["max_output_tokens"]
        assert "gemini_judgment" not in task and "human" not in task
    assert prepare(output=output) == (manifest, tasks)
    assert {name: sha(PREVIOUS / name) for name in previous_files} == before
    assert not (output / "requests.sqlite3").exists()


def test_revision_cannot_overwrite_or_nest_in_original_judge_directory(tmp_path):
    for output in (PREVIOUS, PREVIOUS / "revision"):
        with pytest.raises(ValueError, match="separate"):
            prepare(output=output)
