"""Reuse archived eligibility decisions; do not re-screen or inspect debate outcomes."""

from __future__ import annotations

import ast
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from .models import MAIN_STUDY_REUSE_POLICY, SCREEN_JUDGES, Question, Screen


def load_questions(path: Path) -> tuple[Question, ...]:
    document = json.loads(path.read_text())
    if document.get("purpose") != "engineering_pilot":
        raise ValueError("This entry point only accepts an explicitly identified pilot manifest")
    questions = tuple(Question.from_dict(q) for q in document["questions"])
    # Read the old field for provenance compatibility, not as a main-study exclusion rule.
    fingerprints = document.get("question_fingerprints", document.get("reserved_pilot_fingerprints"))
    if fingerprints != [q.fingerprint for q in questions]:
        raise ValueError("Pilot question fingerprints are missing or inconsistent")
    if "reserved_pilot_fingerprints" in document and document["reserved_pilot_fingerprints"] != fingerprints:
        raise ValueError("Conflicting legacy/current question fingerprints")
    if len({q.fingerprint for q in questions}) != len(questions):
        raise ValueError("Duplicate pilot questions")
    return questions


def load_approved_bank(path: Path) -> tuple[tuple[Question, ...], dict]:
    """Import exactly the user-approved 60 questions without another selection or rewrite."""
    raw = path.read_bytes()
    bank = json.loads(raw)
    if bank.get("purpose") != "main_study_question_bank" or bank.get("status") != "user_approved_question_set":
        raise ValueError("Main study requires the approved question bank")
    rows = bank["questions"]
    if len(rows) != 60 or bank.get("question_count") != 60 or bank.get("candidate_count") != 60:
        raise ValueError("This launch is authorized for the approved 60-question bank only")
    if any(r.get("user_decision") != "保留" for r in rows):
        raise ValueError("Every main-study question must be approved")
    if len({r["review_number"] for r in rows}) != 60:
        raise ValueError("Duplicate review numbers")
    original = bank["original_opinion_screen"]
    screening_path = Path(original["screening_source"])
    screen_bytes = screening_path.read_bytes()
    if hashlib.sha256(screen_bytes).hexdigest() != original["screening_sha256"]:
        raise ValueError("Archived screening evidence changed")
    screened = {r["index"]: r for r in json.loads(screen_bytes)["per_question"]}
    questions, provenance = [], []
    checked_files = {}
    for row in rows:
        evidence = row.get("screening_evidence")
        if evidence:
            evidence_path = path.parent / evidence["results_file"]
            if evidence_path not in checked_files:
                checked_files[evidence_path] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if checked_files[evidence_path] != evidence["results_sha256"]:
                raise ValueError("Extension screening evidence changed")
            verdicts = evidence["verdicts"]
            if len(verdicts) != 3 or {r["model"] for r in verdicts} != set(SCREEN_JUDGES):
                raise ValueError("Extension must retain the three original screening votes")
            if sum(r["question_class"] == "debatable_opinion" for r in verdicts) != row["opinion_votes"]:
                raise ValueError("Extension screening vote count differs from evidence")
            screen = Screen(
                SCREEN_JUDGES,
                row["opinion_votes"],
                evidence_path.name,
                evidence["results_sha256"],
                "fresh_individual_three_judge_votes",
            )
            question_id = f"global-row-{row['dataset_row_zero_based']}"
        else:
            record = screened[row["archive_index"]]
            if not (
                record["keep"]
                and record["question_class"] == "debatable_opinion"
                and record["keep_votes"] == row["opinion_votes"]
                and record["n_judges"] == 3
                and row["question"].startswith(record["question"])
            ):
                raise ValueError("Question does not match its archived opinion screen")
            screen = Screen(SCREEN_JUDGES, row["opinion_votes"], screening_path.name, original["screening_sha256"])
            question_id = f"archived-global-{row['archive_index']}"
        question = Question(
            question_id, row["question"], tuple(row["options"]), screen, "Anthropic/llm_global_opinions"
        )
        question.validate()
        questions.append(question)
        provenance.append(
            {
                "question_id": question.id,
                "fingerprint": question.fingerprint,
                "review_number": row["review_number"],
                "input_review_note": row["input_review_note"],
                "user_decision": row["user_decision"],
                "pilot_question": row["pilot_question"],
            }
        )
    if len({q.fingerprint for q in questions}) != 60 or len({q.id for q in questions}) != 60:
        raise ValueError("Main question bank has duplicate questions or IDs")
    return tuple(questions), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "approval": bank["final_selection"]["approval"],
        "items": provenance,
        "note": "Original text/options retained, including user-accepted historical input limitations",
    }


def import_screened_archive(
    screening_path: Path,
    records_dir: Path,
    *,
    count: int,
    seed: int,
    excluded_fingerprints: set[str] | None = None,
    indices: tuple[int, ...] | None = None,
) -> dict:
    if not 1 <= count <= 10:
        raise ValueError("The engineering importer accepts 1–10 questions, not a main-study launch")
    raw = screening_path.read_bytes()
    screening = json.loads(raw)
    if set(screening["jury"]) != {"gpt-5.5", "opus-4.8", "gemini"}:
        raise ValueError("Unrecognized archived screening jury; verify the model IDs before reuse")
    source_sha = hashlib.sha256(raw).hexdigest()
    eligible = {
        item["index"]: item
        for item in screening["per_question"]
        if item["keep"]
        and item["question_class"] == "debatable_opinion"
        and item["keep_votes"] >= 2
        and item["n_judges"] == 3
    }
    candidates: dict[int, Question] = {}
    record_hashes: dict[int, str] = {}
    for path in sorted(records_dir.glob("*.json")):
        data = path.read_bytes()
        record = json.loads(data)
        if not isinstance(record, dict) or record.get("benchmark") != "global_opinions":
            continue
        index = record.get("index")
        if index not in eligible:
            continue
        item = eligible[index]
        if not record["question"].startswith(item["question"]):
            raise ValueError(f"Archived screening/record mismatch at index {index}")
        options = record["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        question = Question(
            f"archived-global-{index}",
            record["question"],
            tuple(options),
            Screen(SCREEN_JUDGES, item["keep_votes"], screening_path.name, source_sha),
            "Anthropic/llm_global_opinions",
        )
        question.validate()
        if index in candidates and question.fingerprint != candidates[index].fingerprint:
            raise ValueError("Different questions share an archive index")
        if question.fingerprint not in (excluded_fingerprints or set()):
            candidates[index] = question
            record_hashes[index] = hashlib.sha256(data).hexdigest()
    if len(candidates) < count:
        raise ValueError("Not enough matching screened questions with original options")
    if indices is not None:
        if len(indices) != count or len(set(indices)) != count or any(i not in candidates for i in indices):
            raise ValueError("Explicit indices must be distinct, eligible, available and match count")
        chosen = list(indices)
    else:
        chosen = random.Random(seed).sample(sorted(candidates), count)
    questions = [candidates[i] for i in chosen]
    if len({q.fingerprint for q in questions}) != count:
        raise ValueError("Sample contains duplicate question/option pairs")
    return {
        "purpose": "engineering_pilot",
        "selection_seed": seed,
        "selection_rule": (
            "explicit archived indices for engineering/input review; no outcomes used"
            if indices
            else "uniform sample from matching archived eligible question records; no outcomes used"
        ),
        "screening_evidence": "original three-judge aggregate, not reconstructed individual votes",
        "questions": [asdict(q) for q in questions],
        "question_fingerprints": [q.fingerprint for q in questions],
        "archive_record_sha256": {str(i): record_hashes[i] for i in chosen},
        "main_study_reuse": dict(MAIN_STUDY_REUSE_POLICY),
        "input_review": {"status": "pending", "items": []},
    }


def validate_input_review(path: Path, questions: tuple[Question, ...]) -> None:
    review = json.loads(path.read_text()).get("input_review", {})
    items = review.get("items", [])
    if review.get("status") != "complete" or len(items) != len(questions):
        raise ValueError("Resolve and record input-context review before live execution")
    if {item.get("fingerprint") for item in items} != {q.fingerprint for q in questions}:
        raise ValueError("Input review does not cover the exact question/option fingerprints")
    if any(item.get("status") != "usable" or not item.get("reason") for item in items):
        raise ValueError("Unresolved input-context issue in the proposed live question list")
