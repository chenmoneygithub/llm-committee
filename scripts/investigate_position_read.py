"""Bounded, explicitly authorized replay of one incomplete position read.

This is a diagnostic, not a main-study restart. Each attempt has its own durable
journal. Original requests/responses remain untouched; stop at the first valid
response, never select among successful responses based on content.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from llm_committee.pivot.databricks_provider import DatabricksProvider, databricks_payload
from llm_committee.pivot.models import Message, Question, Request, digest
from llm_committee.pivot.prompts import parse_position
from llm_committee.pivot.storage import Journal, RunBlocked
from llm_committee.pivot.study import atomic_json


def investigate(source: Path, key: str, output: Path, *, retries: int, provider_factory=None):
    if type(retries) is not int or not 1 <= retries <= 2:
        raise ValueError("Choose at most two explicitly authorized diagnostic retries")
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Keep diagnostic outputs outside the original run directory")
    old_manifest = json.loads((source / "manifest.json").read_text())
    with sqlite3.connect((source / "requests.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        found = db.execute("SELECT * FROM calls WHERE key=?", (key,)).fetchone()
        if found is None:
            raise ValueError("Source request does not exist")
        row = dict(found)
    document = json.loads(row["request"])
    request = Request(
        **{
            **document,
            "messages": tuple(Message(**m) for m in document["messages"]),
            "candidate_labels": tuple(document["candidate_labels"]),
        }
    )
    saved = json.loads(row["response"])
    if not (
        row["status"] == "invalid"
        and row["error"] == "Position must contain a first-line choice and full position text"
        and request.purpose == "position"
        and request.effort == "none"
        and not request.candidate_labels
        and old_manifest["config"]["closed_provider"] == "databricks"
        and saved["status"] == "completed"
    ):
        raise ValueError("This diagnostic is restricted to a saved incomplete Databricks position paragraph")
    question = next(Question.from_dict(q) for q in old_manifest["questions"] if key.startswith(q["id"] + "/"))
    payload = databricks_payload(request)
    if digest(request.document()) != row["request_hash"] or payload != saved["raw"]["request_payload"]:
        raise ValueError("Request or provider payload changed; no diagnostic call made")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "paid_diagnostic_retry",
        "source_directory": str(source),
        "source_manifest_sha256": digest(old_manifest),
        "source_record_sha256": digest(row),
        "source_request_key": key,
        "source_request_sha256": row["request_hash"],
        "source_payload_sha256": digest(payload),
        "maximum_additional_attempts": retries,
        "config": old_manifest["config"],
        "execution": {"budget_policy": "no_limit_user_requested"},
        "authorization": "User authorized at most two retries with logging; investigate this record before full resumption",
        "policy": "Identical request; first valid response only; preserve every attempt; do not modify or restart main study",
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "original-record.json", row)
    summary = {
        "status": "running",
        "attempts": [],
        "new_charged_or_reserved_usd": 0.0,
        "main_study_resumed": False,
        "source_unchanged": True,
    }
    atomic_json(output / "report.json", summary)
    provider = (
        provider_factory or (lambda: DatabricksProvider(profile=old_manifest["config"]["databricks_profile"]))
    )()
    try:
        for attempt in range(1, retries + 1):
            journal = Journal(
                output / f"attempt-{attempt}" / "requests.sqlite3", {**manifest, "attempt": attempt}, None
            )
            try:
                parsed, error = None, None
                try:
                    parsed = journal.call(request, provider, lambda text: parse_position(text, question))
                except RunBlocked as exc:
                    error = str(exc)
                audit = journal.audit()
                (call,) = audit["calls"]
                completion = call["response"] or {}
                raw = completion.get("raw") or {}
                response = raw.get("response") or {}
                if raw.get("request_payload") is not None and raw["request_payload"] != payload:
                    raise ValueError("Diagnostic payload provenance does not match the original")
                item = {
                    "attempt": attempt,
                    "status": call["status"],
                    "error": call["error"] or error,
                    "journal": f"attempt-{attempt}/requests.sqlite3",
                    "request_sha256": digest(call["request"]),
                    "payload_sha256": digest(raw["request_payload"]) if "request_payload" in raw else None,
                    "response_id": response.get("id"),
                    "response_id_differs_from_original": (
                        response["id"] != saved["raw"]["response"].get("id") if response.get("id") else None
                    ),
                    "text": completion.get("text"),
                    "parsed": parsed,
                    "output_tokens": completion.get("output_tokens"),
                    "reasoning_tokens": completion.get("reasoning_tokens"),
                    "charge_usd": call["charge_usd"],
                }
                summary["attempts"].append(item)
                summary["new_charged_or_reserved_usd"] += audit["charged_or_reserved_usd"]
                atomic_json(output / "report.json", summary)
                print(json.dumps(item, ensure_ascii=False), flush=True)
                if parsed is not None:
                    summary["status"] = "first_valid_retry_recorded"
                    summary["first_valid_attempt"] = attempt
                    break
                if call["status"] != "invalid" or call["error"] != row["error"]:
                    summary["status"] = "different_failure_requires_inspection"
                    break
            finally:
                journal.close()
        else:
            summary["status"] = "same_failure_on_both_retries"
    finally:
        try:
            provider.close()
        except Exception as exc:
            summary["cleanup_error_type"] = type(exc).__name__
        with sqlite3.connect((source / "requests.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            after = dict(db.execute("SELECT * FROM calls WHERE key=?", (key,)).fetchone())
            if after != row:
                summary["source_unchanged"] = False
                raise RuntimeError("Source record changed during diagnosis")
        atomic_json(output / "report.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorized-retries", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    result = investigate(args.source, args.key, args.output, retries=args.authorized_retries)
    print(json.dumps({k: v for k, v in result.items() if k != "attempts"}, ensure_ascii=False, indent=2))
