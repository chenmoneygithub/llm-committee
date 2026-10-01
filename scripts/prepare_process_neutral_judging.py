"""Prepare an append-only E2 judge-rubric revision; never dispatch model calls.

The frozen Gemini and Grok inputs are reused without changing chairman outputs,
answer order, model settings, schema, or the forced-choice task. A new namespace
prevents the revised requests from overwriting or reusing old judgments.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from llm_committee.pivot.models import Message, canonical, digest
from scripts.grok_quality_check import OUTPUT as PREVIOUS_JUDGING
from scripts.grok_quality_check import request_from, verify_source
from scripts.scale_public_history import ROOT, freeze, read, sha

VERSION = "E2-process-reference-neutral-v2"
OUTPUT = ROOT / "runs/E2-process-reference-neutral-v2-20260928"
ADDENDUM = (
    "Do not reward or penalize references to a committee, its members, or a discussion process "
    "merely because they appear. Judge the substantive reasoning rather than the presence or "
    "absence of these references. If an answer has a genuine evidential or logical gap, identify "
    "that specific gap; do not infer one solely from committee-related wording."
)


def revised_request(request):
    """Change only the judge's developer instruction and its versioned cache key."""
    if request.purpose != "judge_e":
        raise ValueError("This revision applies only to the final-answer preference judge")
    if not request.messages or request.messages[0].role != "developer":
        raise ValueError("Expected the original developer rubric as the first message")
    if request.key.startswith(VERSION + "/") or any(ADDENDUM in m.text for m in request.messages):
        raise ValueError("Request already uses the revised rubric")
    return replace(
        request,
        key=f"{VERSION}/{request.key}",
        messages=(Message("developer", request.messages[0].text + "\n\n" + ADDENDUM), *request.messages[1:]),
    )


def prepare(previous=PREVIOUS_JUDGING, output=OUTPUT):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    old_manifest, old_tasks = read(previous / "manifest.json"), read(previous / "tasks.json")
    source = Path(old_manifest["source_directory"]).resolve()
    for protected in (previous, source):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Use a separate revision directory")
    if digest(old_tasks) != old_manifest["tasks_sha256"]:
        raise ValueError("Previous judge tasks changed")
    verify_source(old_manifest)
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        originals = {
            json.loads(req)["key"]: json.loads(req)
            for (req,) in db.execute("SELECT request FROM calls WHERE status='completed'")
        }
    tasks = []
    for old in old_tasks:
        gemini, grok = originals[old["source_key"]], old["request"]
        # The prior study changed only model/key when sending Gemini's inputs to Grok.
        if {**grok, "model": gemini["model"], "key": gemini["key"]} != gemini:
            raise ValueError("Previous judges do not share identical inputs/settings")
        for document in (gemini, grok):
            request = revised_request(request_from(document))
            tasks.append(
                {
                    "question_id": old["question_id"],
                    "assignment": old["assignment"],
                    "order": old["order"],
                    "target_side": old["target_side"],
                    "previous_request_key": document["key"],
                    "request": json.loads(canonical(request.document())),
                }
            )
    if len({t["request"]["key"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate revised judge request")
    manifest = {
        "version": VERSION,
        "status": "prepared_not_dispatched",
        "scope": "Existing mixed-family triadic E2 pairs only; no synthesis, debate, or other layer changes",
        "addendum": ADDENDUM,
        "source_directory": str(source),
        "previous_judge_directory": str(previous),
        "source_sha256": old_manifest["source_sha256"],
        "previous_files_sha256": {name: sha(previous / name) for name in ("manifest.json", "tasks.json")},
        "implementation_sha256": sha(__file__),
        "models": sorted({t["request"]["model"] for t in tasks}),
        "questions": len({t["question_id"] for t in tasks}),
        "answer_pairs": len({(t["question_id"], t["assignment"]) for t in tasks}),
        "planned_calls_if_executed": len(tasks),
        "tasks_sha256": digest(tasks),
        "unchanged": ["answer_texts", "answer_order", "models", "reasoning_settings", "schema", "forced_choice"],
        "interpretation": (
            "Exploratory rubric sensitivity revision prompted by human review. It neither rejects consensus "
            "as possible evidence nor ignores genuine evidential gaps. No human labels or previous judge "
            "preferences enter the model request. The already-inspected human sample is not held-out "
            "validation for this revision. Original ratings and reports remain unchanged."
        ),
    }
    freeze(output / "manifest.json", manifest)
    freeze(output / "tasks.json", tasks)
    return manifest, tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=PREVIOUS_JUDGING)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest, _ = prepare(args.previous, args.output)
    print(json.dumps({"status": manifest["status"], "directory": str(args.output.resolve()), "model_calls": 0}))
