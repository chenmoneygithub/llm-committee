"""Read-only checkpoint import for engineering continuations, not a manifest bypass."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .failures import (
    FORMAT_RETRY_POLICY,
    attempt_number,
    local_transport_record,
    retry_key,
    retryable_failure_record,
    retryable_format_record,
)
from .models import Message, Request, canonical, digest
from .probabilities import LOGPROB_MISMATCH_MESSAGES, TOPK_ZERO_FILL_POLICY


def validate_attempts(rows, *, allow_retries, closed_provider="direct"):
    """Physical attempts may alias one logical request, never a different prompt or later success."""
    groups = {}
    for row in rows:
        request = json.loads(row["request"])
        if digest(request) != row["request_hash"]:
            raise ValueError("Checkpoint request hash/identity mismatch")
        number = attempt_number(row["key"], request["key"])
        if number and not allow_retries:
            raise ValueError("Checkpoint retries require the explicit failure policy")
        group = groups.setdefault(request["key"], {})
        if number in group:
            raise ValueError("Duplicate attempt")
        group[number] = row
    for group in groups.values():
        if sorted(group) != list(range(len(group))):
            raise ValueError("Checkpoint retry sequence has missing attempts")
        first = group[0]
        for number, row in group.items():
            if row["request_hash"] != first["request_hash"] or canonical(json.loads(row["request"])) != canonical(
                json.loads(first["request"])
            ):
                raise ValueError("Retry changed the logical request")
            if number and not retryable_failure_record(
                json.loads(first["request"]), group[number - 1], closed_provider
            ):
                raise ValueError("Retry after success or unsupported failure")


def reconcile_json_duplicates(row: dict) -> dict:
    """Accept only redundant JSON fields in a saved, otherwise valid structured response."""
    from .prompts import parse_json

    request = json.loads(row["request"])
    saved = json.loads(row["response"])
    if not (
        row["status"] == "invalid"
        and (row["error"] or "").startswith("Duplicate JSON field: ")
        and request.get("schema")
        and not request.get("candidate_labels")
        and not request.get("scoring")
        and saved["status"] == "completed"
        and not (request["effort"] == "none" and saved.get("reasoning_tokens"))
    ):
        raise ValueError("This checkpoint failure cannot be reconciled as identical JSON duplicates")
    # Full validation rejects conflicts, wrong types, invalid enums and missing/extra fields.
    parsed = parse_json(saved["text"], request["schema"])
    pairs = json.loads(saved["text"], object_pairs_hook=list)
    seen, duplicates = set(), set()
    for key, _ in pairs:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if row["error"].removeprefix("Duplicate JSON field: ") not in duplicates:
        raise ValueError("Saved response does not contain the reported duplicate field")
    correction = {
        "key": row["key"],
        "source_response_sha256": digest(saved),
        "source_status": row["status"],
        "source_error": row["error"],
        "source_charge_usd": row["charge"],
        "identical_duplicate_fields": sorted(duplicates),
        "note": "Reparsed saved text; raw response and charge unchanged; zero new API calls; source unchanged",
    }
    row.update(status="completed", parsed=canonical(parsed), error=None)
    return correction


def reconcile_gemini_usage(row: dict, closed_provider: str) -> dict:
    """Reparse a received response offline; never regenerate or modify the source."""
    from .databricks_provider import parse_databricks_response
    from .prompts import parse_json
    from .providers import cost_usd

    document = json.loads(row["request"])
    saved = json.loads(row["response"])
    raw = saved.get("raw") or {}
    if not (
        closed_provider == "databricks"
        and row["status"] == "billing_unknown"
        and row["error"] in ("Inconsistent billing token counts", "Missing or invalid billing token counts")
        and document["model"] == "gemini-3.8-flash"
        and document["purpose"] in ("judge_b", "judge_c", "judge_e")
        and raw.get("provider") == "databricks"
        and raw.get("http_status") == 200
    ):
        raise ValueError("This checkpoint failure cannot be reconciled by the Gemini usage fix")
    request = Request(**{**document, "messages": tuple(Message(**m) for m in document["messages"])})
    response = raw["response"]
    completion = parse_databricks_response(request, response, {k: v for k, v in raw.items() if k != "response"})
    charge = cost_usd(request.model, completion, closed_provider=closed_provider)
    if completion.status != "completed" or charge > row["charge"] or completion.text != saved["text"]:
        raise ValueError("Offline usage reconciliation failed validation")
    parsed = parse_json(completion.text, request.schema)
    correction = {
        "key": row["key"],
        "source_response_sha256": digest(saved),
        "source_status": row["status"],
        "source_reserved_usd": row["charge"],
        "normalized_cost_usd": charge,
        "usage_accounting": completion.raw["usage_accounting"],
        "note": "Reparsed saved HTTP response; same judgment text; zero new API calls; source unchanged",
    }
    row.update(
        status="completed", charge=charge, response=canonical(asdict(completion)), parsed=canonical(parsed), error=None
    )
    return correction


def load_checkpoint(
    directory: Path,
    manifest: dict,
    *,
    reconcile_databricks_usage: bool = False,
    reconcile_identical_json_duplicates: bool = False,
    reanalyze_topk_zero_fill: bool = False,
    allow_question_expansion: bool = False,
) -> tuple[dict, list[dict]]:
    directory = directory.resolve()
    source = json.loads((directory / "manifest.json").read_text())
    if allow_question_expansion:
        if manifest["kind"] not in ("paid_main_study", "offline_mock"):
            raise ValueError("Question expansion requires an explicit main-study manifest")
        if manifest.get("study", {}).get("scope") != "approved_bank_mixed_family":
            raise ValueError("Question expansion is limited to the approved mixed-family study")
        allowed_sources = ("paid_engineering_pilot", "paid_main_study")
        if source["kind"] not in (("offline_mock",) if manifest["kind"] == "offline_mock" else allowed_sources):
            raise ValueError("Cannot mix synthetic and real checkpoints")
        for field, key in (("questions", "id"), ("plans", "question_id")):
            before = {q[key]: q for q in source[field]}
            after = {q[key]: q for q in manifest[field]}
            if len(before) != len(source[field]) or len(after) != len(manifest[field]):
                raise ValueError("Duplicate checkpoint question identifiers")
            if not before or any(after.get(qid) != value for qid, value in before.items()):
                raise ValueError(f"Study expansion would change existing {field}")
        if source["question_fingerprints"] != [q["question_fingerprint"] for q in source["plans"]]:
            raise ValueError("Inconsistent checkpoint question fingerprints")
    policy_change = source.get("probability_readout") != manifest.get("probability_readout")
    if policy_change and not (
        reanalyze_topk_zero_fill
        and source.get("probability_readout") is None
        and manifest.get("probability_readout") == TOPK_ZERO_FILL_POLICY
    ):
        raise ValueError("Continuation would change probability_readout; explicit offline reanalysis is required")
    # Scheduling/failure reporting and implementation may change. Scientific inputs,
    # prompt version, runtime, costs, routes and samples may not change under this command.
    for field in (
        "schema_version",
        "kind",
        "prompt_version",
        "agreement_rubric",
        "runtime_packages",
        "cost_accounting",
        "config",
        "questions",
        "question_fingerprints",
        "main_study_reuse",
        "plans",
        "excluded",
    ):
        if allow_question_expansion and field in ("kind", "questions", "question_fingerprints", "plans"):
            continue
        if source.get(field) != manifest.get(field):
            raise ValueError(f"Continuation would change {field}; no checkpoint reuse")
    db = sqlite3.connect((directory / "requests.sqlite3").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        if identity["manifest"] != source:
            raise ValueError("Source journal and source manifest disagree")
        rows = [dict(row) for row in db.execute("SELECT * FROM calls ORDER BY rowid")]
    finally:
        db.close()
    source_rows_sha256 = digest(rows)
    allow_retries = manifest.get("execution", {}).get("failure_policy") == FORMAT_RETRY_POLICY
    validate_attempts(rows, allow_retries=allow_retries, closed_provider=manifest["config"]["closed_provider"])
    reconciliations = []
    json_reconciliations = []
    for row in rows:
        request = json.loads(row["request"])
        if row["status"] == "billing_unknown" and reconcile_databricks_usage:
            reconciliations.append(reconcile_gemini_usage(row, source["config"]["closed_provider"]))
        if (
            reconcile_identical_json_duplicates
            and row["status"] == "invalid"
            and (row["error"] or "").startswith("Duplicate JSON field: ")
        ):
            json_reconciliations.append(reconcile_json_duplicates(row))
        known_mismatch = (
            row["status"] == "invalid"
            and request["purpose"] == "candidate_scores"
            and request.get("scoring")
            and row["error"] in LOGPROB_MISMATCH_MESSAGES
        )
        known_format = allow_retries and retryable_format_record(request, row)
        known_transport = allow_retries and local_transport_record(request, row, manifest["config"]["closed_provider"])
        if row["status"] != "completed" and not known_mismatch and not known_format and not known_transport:
            raise ValueError(f"Cannot continue unresolved/unsupported checkpoint status: {row['status']}")
        if (not row["response"] and not known_transport) or (row["status"] == "completed" and not row["parsed"]):
            raise ValueError("Checkpoint is missing its recorded response/parsed result")
        if not math.isfinite(row["charge"]) or row["charge"] < 0:
            raise ValueError("Checkpoint charge is invalid")
    inherited_cost = source.get("continuation", {}).get("prior_charged_or_reserved_usd", 0.0)
    descriptor = {
        "source_directory": str(directory),
        "source_manifest_sha256": digest(source),
        "source_calls_sha256": source_rows_sha256,
        "import_rows_sha256": digest(rows),
        "offline_usage_reconciliations": reconciliations,
        "reused_calls": len(rows),
        "prior_charged_or_reserved_usd": inherited_cost + sum(row["charge"] for row in rows),
        "policy": "Exact request-hash reuse; original invalid records remain invalid; import makes no API calls",
    }
    if reconcile_identical_json_duplicates:
        descriptor["offline_json_reconciliations"] = json_reconciliations
    if policy_change:
        descriptor["analysis_policy_change"] = {
            "from": "exact_candidate_scores_legacy",
            "to": dict(TOPK_ZERO_FILL_POLICY),
            "note": "Recompute all D metrics from original sampled top-k; do not use or alter supplemental scores",
        }
    if allow_question_expansion:
        descriptor["study_extension"] = {
            "reused_question_ids": [
                q["id"] for q in source["questions"] if any(r["key"].startswith(q["id"] + "/") for r in rows)
            ],
            "source_kind": source["kind"],
            "target_question_count": len(manifest["questions"]),
            "note": (
                "Same prior questions, routes, samples, prompts and model settings; "
                "reused_question_ids have saved calls, not necessarily completed questions"
            ),
        }
    return descriptor, rows


def reuse_diagnostic_retry(directory: Path, inherited):
    """Import an already paid, identical diagnostic retry; verify sources read-only."""
    from .databricks_provider import databricks_payload
    from .models import Completion, Question
    from .prompts import parse_position
    from .providers import cost_usd

    descriptor, original_rows = inherited
    rows = [dict(r) for r in original_rows]
    directory = directory.resolve()
    diagnostic = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    source_row = json.loads((directory / "original-record.json").read_text())
    key = diagnostic["source_request_key"]
    base = next(r for r in rows if r["key"] == key)
    if not (
        diagnostic["kind"] == "paid_diagnostic_retry"
        and diagnostic["source_directory"] == descriptor["source_directory"]
        and diagnostic["source_manifest_sha256"] == descriptor["source_manifest_sha256"]
        and digest(base) == digest(source_row) == diagnostic["source_record_sha256"]
        and base["request_hash"] == diagnostic["source_request_sha256"]
        and report["status"] == "first_valid_retry_recorded"
        and 1 <= report["first_valid_attempt"] <= diagnostic["maximum_additional_attempts"] <= 2
        and len(report["attempts"]) == report["first_valid_attempt"]
    ):
        raise ValueError("Diagnostic provenance does not match the checkpoint")
    document = json.loads(base["request"])
    request = Request(**{**document, "messages": tuple(Message(**m) for m in document["messages"])})
    payload = databricks_payload(request)
    if (
        digest(payload) != diagnostic["source_payload_sha256"]
        or payload != json.loads(base["response"])["raw"]["request_payload"]
    ):
        raise ValueError("Diagnostic payload differs from original")
    source_manifest = json.loads((Path(descriptor["source_directory"]) / "manifest.json").read_text())
    question = next(Question.from_dict(q) for q in source_manifest["questions"] if key.startswith(q["id"] + "/"))
    imported, cost = [], 0.0
    for number, item in enumerate(report["attempts"], 1):
        path = directory / f"attempt-{number}" / "requests.sqlite3"
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
            records = [dict(r) for r in db.execute("SELECT * FROM calls")]
        if identity["manifest"] != {**diagnostic, "attempt": number} or len(records) != 1:
            raise ValueError("Diagnostic journal identity mismatch")
        row = records[0]
        saved = json.loads(row["response"])
        if not (
            row["key"] == key
            and row["request_hash"] == base["request_hash"]
            and digest(json.loads(row["request"])) == base["request_hash"]
            and saved["raw"]["request_payload"] == payload
            and item["attempt"] == number
            and item["status"] == row["status"]
            and item["charge_usd"] == row["charge"]
            and math.isclose(row["charge"], cost_usd(request.model, Completion(**saved), closed_provider="databricks"))
        ):
            raise ValueError("Diagnostic attempt does not match saved request, payload or cost")
        if number == report["first_valid_attempt"]:
            if (
                row["status"] != "completed"
                or saved["status"] != "completed"
                or saved.get("reasoning_tokens")
                or json.loads(row["parsed"]) != parse_position(saved["text"], question)
            ):
                raise ValueError("Diagnostic selected response is not format-valid")
        elif not retryable_format_record(document, row):
            raise ValueError("Diagnostic retried a non-format failure")
        imported.append({"attempt": number, "source_record_sha256": digest(row), "source_journal": str(path)})
        cost += row["charge"]
        row["key"] = retry_key(key, number)
        if any(r["key"] == row["key"] for r in rows):
            raise ValueError("Diagnostic attempt was already imported")
        rows.append(row)
    if not math.isclose(cost, report["new_charged_or_reserved_usd"]):
        raise ValueError("Diagnostic cost ledger mismatch")
    validate_attempts(rows, allow_retries=True, closed_provider=source_manifest["config"]["closed_provider"])
    return {
        **descriptor,
        "import_rows_sha256": digest(rows),
        "reused_calls": len(rows),
        "prior_charged_or_reserved_usd": descriptor["prior_charged_or_reserved_usd"] + cost,
        "diagnostic_retry_import": {
            "directory": str(directory),
            "manifest_sha256": digest(diagnostic),
            "report_sha256": digest(report),
            "attempts": imported,
            "cost_usd": cost,
        },
    }, rows
