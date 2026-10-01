"""Outcome-blind, 20-question human sample; answer origins stay out of the HTML."""

import argparse
import copy
import random
from pathlib import Path

from llm_committee.pivot.models import canonical, digest
from scripts.grok_quality_check import SOURCE
from scripts.scale_public_history import ROOT, freeze, read, sha

OUTPUT = ROOT / "docs/human-answer-review-20.html"
PRIVATE = ROOT / "runs/human-answer-review-20-20260928"
TEMPLATE = ROOT / "scripts/templates/human_answer_review.html"
SEED = 2026092801


def apply_translation_corrections(public, value, path):
    """Apply recorded fidelity corrections without overwriting the model output."""
    if not path.exists():
        return value["items"]
    corrections = read(path)
    if corrections["base_translation_id"] != value["translation_id"]:
        raise ValueError("Corrections belong to a different translation")
    items = copy.deepcopy(value["items"])
    source = {item["id"]: item for item in public["items"]}
    targets = {item["id"]: item for item in items}
    for edit in corrections["edits"]:
        if edit["item_id"] not in targets:
            raise ValueError("Unknown correction item")
        parent, original = targets[edit["item_id"]], source[edit["item_id"]]
        field = edit["field"]
        if field == ["question"]:
            key = "question"
        elif len(field) == 2 and field[0] == "answers" and field[1] in ("A", "B"):
            parent, original, key = parent["answers"], original["answers"], field[1]
        elif (
            len(field) == 2
            and field[0] == "options"
            and type(field[1]) is int
            and 0 <= field[1] < len(parent["options"])
        ):
            parent, original, key = parent["options"], original["options"], field[1]
        else:
            raise ValueError("Invalid correction field")
        if (
            not edit["before"]
            or not edit["after"].strip()
            or parent[key].count(edit["before"]) != 1
            or not edit["source_excerpt"]
            or edit["source_excerpt"] not in original[key]
        ):
            raise ValueError("Correction does not match the English source and translated text")
        parent[key] = parent[key].replace(edit["before"], edit["after"], 1)
    return items


def attach_translation(public, path):
    """Translations never change item IDs, English, ordering or saved-session identity."""
    path = Path(path)
    if not path.exists():
        return public
    value = read(path)
    if value["dataset_id"] != public["dataset_id"] or value["source_items_sha256"] != digest(public["items"]):
        raise ValueError("Translation belongs to a different English review set")
    if value["language"] != "zh" or value["translation_id"] != digest(value["items"]):
        raise ValueError("Invalid translation language or content hash")
    if len(value["items"]) != len(public["items"]):
        raise ValueError("Incomplete translation set")
    for original, translated in zip(public["items"], value["items"], strict=True):
        if (
            not isinstance(translated, dict)
            or set(translated) != set(original)
            or translated["id"] != original["id"]
            or not isinstance(translated["options"], list)
            or len(translated["options"]) != len(original["options"])
            or not isinstance(translated["answers"], dict)
            or set(translated["answers"]) != {"A", "B"}
        ):
            raise ValueError("Translation changes item identity, options or answer sides")
        strings = [translated["question"], *translated["options"], *translated["answers"].values()]
        if any(not isinstance(text, str) or not text.strip() for text in strings):
            raise ValueError("Blank or malformed translation")
    items = apply_translation_corrections(public, value, path.with_name("translation.zh-corrections.json"))
    return {**public, "translations": {"zh": items}, "translation_ids": {"zh": digest(items)}}


def build(source=SOURCE, output=OUTPUT, private=PRIVATE):
    source, output, private = map(Path, (source, output, private))
    manifest = read(source / "manifest.json")
    assert manifest["config"]["roster"] == "mixed_family"
    qids = sorted(p["question_id"] for p in manifest["plans"])
    assert len(qids) == len(set(qids)) == 50
    rng = random.Random(SEED)
    selected = rng.sample(qids, 20)
    assignments = [(arm, side) for arm in ("original", "alternate") for side in ("A", "B")] * 5
    rng.shuffle(assignments)
    items, key = [], []
    source_hashes = {"manifest.json": sha(source / "manifest.json")}
    for qid, (arm, debate_side) in zip(selected, assignments, strict=True):
        path = source / "questions" / f"{qid}.json"
        source_hashes[str(path.relative_to(source))] = sha(path)
        record = read(path)
        row = next(r for r in record["E2"] if r["assignment"] == arm)
        assert row["baseline"] and row["debated"]
        answers = {debate_side: row["debated"]["answer"], "B" if debate_side == "A" else "A": row["baseline"]["answer"]}
        item_id = digest({"seed": SEED, "question": qid, "assignment": arm})[:20]
        items.append(
            {
                "id": item_id,
                "question": record["question"]["text"],
                "options": record["question"]["options"],
                "answers": answers,
            }
        )
        key.append(
            {
                "item_id": item_id,
                "question_id": qid,
                "assignment": arm,
                "debate_side": debate_side,
                "baseline_sha256": digest(row["baseline"]["answer"]),
                "debate_sha256": digest(row["debated"]["answer"]),
            }
        )
    public = {"version": "blind-answer-review-v1", "items": items}
    public["dataset_id"] = digest(public)[:24]
    freeze(private / "researcher-key.json", {"dataset_id": public["dataset_id"], "items": key})
    freeze(
        private / "manifest.json",
        {
            "dataset_id": public["dataset_id"],
            "source_directory": str(source.resolve()),
            "source_sha256": source_hashes,
            "seed": SEED,
            "sampling_frame": qids,
            "selected_questions": selected,
            "method": "Uniform 20 of 50 questions, one pair per question, outcome-blind; ten per tone assignment and balanced A/B sides within each assignment. No judge outputs enter sampling.",
            "human_rubric": "Same substantive preference criteria as model judges; humans may additionally choose tie or unjudgeable, kept separate from forced-choice model scores.",
            "annotation_status": "No human labels collected by this build",
        },
    )
    public = attach_translation(public, private / "translation.zh.json")
    document = TEMPLATE.read_text().replace("__BLIND_DATA__", canonical(public).replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return output.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    args = parser.parse_args()
    print(build(output=args.output, private=args.private))
