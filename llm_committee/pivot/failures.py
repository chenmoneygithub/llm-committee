"""Bounded technical retries; never choose a response based on its substantive answer."""

from __future__ import annotations

import json

FORMAT_RETRY_POLICY = {
    "name": "logged_request_retries_v4_transport",
    "max_additional_attempts": 2,
    "selection": "first format-valid response; identical model, prompt and generation settings",
    "exhaustion": "mark task missing and block true dependents; continue independent work",
    "inkling_direct_reasoning": (
        "User-authorized identical retry for unexpected reasoning in Inkling effort=none position/d_text reads; "
        "same two-additional-attempt limit; never accept the noncompliant response or strip its reasoning"
    ),
    "transport_billing_integrity": (
        "known Databricks connection/timeout failures: retry within the same two-additional-attempt limit, "
        "retain unresolved original reservations; mark missing only after exhaustion; "
        "unknown provider, billing, configuration and integrity errors still stop for inspection"
    ),
    "transport_retry_backoff_seconds": [0.5, 1.0],
}
OUTPUT_LIMIT_RETRY_POLICY = {
    "name": "databricks_known_output_limit_v1",
    "max_additional_attempts": 2,
    "scope": "HTTP 400 BAD_REQUEST with the exact known max_tokens/model output limit error",
    "settings": "Keep the original model, prompt, reasoning and output-token limit unchanged",
    "billing": "Preserve missing usage as unknown with its original reservation; never assume zero cost",
    "exhaustion": "Mark only this task and its true dependents missing; continue independent work",
}
# Historical storage-key spelling retained for checkpoint compatibility; also used
# for connection retries. The provider receives the unchanged original request key.
RETRY_SEPARATOR = "::format-retry:"


def retry_key(key: str, attempt: int) -> str:
    return key if attempt == 0 else f"{key}{RETRY_SEPARATOR}{attempt}"


def attempt_number(storage_key: str, logical_key: str) -> int:
    for attempt in range(3):
        if storage_key == retry_key(logical_key, attempt):
            return attempt
    raise ValueError("Invalid retry storage key")


def retryable_format_record(request: dict, row: dict) -> bool:
    if row["status"] != "invalid" or not row.get("response") or request.get("scoring"):
        return False
    response = json.loads(row["response"])
    raw = response.get("raw") or {}
    if (
        request.get("model") == "thinkingmachines/Inkling"
        and request["effort"] == "none"
        and request["purpose"] in ("position", "d_text", "d_choice")
        and raw.get("provider") == "tinker"
        and raw.get("model") == request["model"]
        and raw.get("effort") == "none"
        and response["status"] == "unexpected_reasoning"
        and row.get("error") == "Response status: unexpected_reasoning"
    ):
        return True
    if request["effort"] == "none" and response.get("reasoning_tokens"):
        return False
    error = row.get("error") or ""
    if response["status"] == "incomplete":
        return error == "Response status: incomplete"
    if response["status"] == "invalid_native_output":
        # Multiple visible text blocks can make concatenated text impossible to align
        # before the ordinary output-contract parser runs. Retry only when that SAME
        # parser independently rejects the text. A well-formed answer with invalid
        # probability provenance remains an integrity failure, not a sampling retry.
        from types import SimpleNamespace

        from .prompts import parse_position, parse_rating

        raw = response.get("raw") or {}
        if not (
            error == "Response status: invalid_native_output"
            and raw.get("provider") == "tinker"
            and raw.get("stop_reason") == "stop"
            and raw.get("parsed_message", {}).get("role") == "assistant"
            and not raw.get("parsed_message", {}).get("tool_calls")
            and (
                raw.get("validation_error") == "Cannot uniquely align parsed content with raw sampled tokens"
                or (
                    request["purpose"] == "d_choice"
                    and raw.get("validation_error")
                    == "Direct output must begin with the requested letter, without whitespace"
                )
            )
        ):
            return False
        try:
            if request["purpose"] == "d_text":
                parse_rating(response["text"])
            elif request["purpose"] == "d_choice":
                from .stateful_prompts import parse_choice

                parse_choice(response["text"], SimpleNamespace(labels=tuple(request["candidate_labels"])))
            elif request["purpose"] == "position":
                parse_position(response["text"], SimpleNamespace(labels=tuple(request["candidate_labels"])))
            else:
                return False
        except ValueError:
            return True
        return False
    if response["status"] != "completed":
        return False
    if request.get("schema"):
        # These requests have no probability readout: their parse errors are output-format errors.
        return not request.get("candidate_labels")
    if request["purpose"] == "position":
        return error in (
            "Position must contain a first-line choice and full position text",
            "Not a direct choice-first position reading",
        )
    if request["purpose"] == "d_text":
        return error == "D-text must return exactly one rating letter, no prose or thinking"
    if request["purpose"] == "d_choice":
        return error == "D-choice must return exactly one original option letter, no prose or thinking"
    return False


def local_transport_record(request: dict, row: dict, closed_provider: str) -> bool:
    """Recognize a retryable connection/timeout failure without claiming zero usage.

    Scope is the Databricks/OpenRouter HTTP adapters. Native Tinker parsing/provenance
    exceptions must not be swallowed as network errors. Old rows retained the
    exception class only, so use a narrow list of HTTP connection/timeout classes.
    """
    return (
        closed_provider in ("databricks", "openrouter")
        and (request["model"].startswith("gpt-5.6-") or request["model"] == "gemini-3.8-flash")
        and row["status"] == "uncertain"
        and row.get("response") is None
        and row.get("error") in ("ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout")
    )


def retryable_failure_record(request: dict, row: dict, closed_provider: str) -> bool:
    return retryable_format_record(request, row) or local_transport_record(request, row, closed_provider)


def local_output_limit_record(request: dict, row: dict, closed_provider: str) -> bool:
    """A narrowly identified generation failure, not permission to ignore billing errors."""
    if not (
        closed_provider == "databricks"
        and row.get("status") == "billing_unknown"
        and row.get("response")
        and not request.get("scoring")
        and not request.get("candidate_labels")
    ):
        return False
    try:
        response = json.loads(row["response"])
        raw = response.get("raw") or {}
        body = raw.get("response") or {}
        # Use exactly the error observed, not a substring that could also match
        # a context overflow, unsupported parameter, authentication or quota error.
        detail = json.loads(body.get("message", ""))
        from .databricks_provider import ENDPOINTS, databricks_payload
        from .models import Message, Request

        saved_request = Request(**{**request, "messages": tuple(Message(**m) for m in request["messages"])})
        return (
            response.get("status") == "http_error"
            and not response.get("text")
            and raw.get("provider") == "databricks"
            and raw.get("http_status") == 400
            and raw.get("endpoint") == ENDPOINTS.get(request["model"])
            and raw.get("request_payload") == databricks_payload(saved_request)
            and body.get("error_code") == "BAD_REQUEST"
            and detail.get("error", {}).get("type") == "invalid_request_error"
            and detail.get("error", {}).get("message")
            == "Could not finish the message because max_tokens or model output limit was reached. Please try again with higher max_tokens."
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def retry_statistics(rows, failures):
    """Count physical retries separately from logical requests, grouped by prior failure.

    A request encountering both failure kinds belongs to both recovery subsets;
    request_retries is the unduplicated total.
    """
    groups = {}
    for key, request, status in rows:
        logical = json.loads(request)["key"]
        groups.setdefault(logical, {})[attempt_number(key, logical)] = status
    totals = {
        name: {"additional_attempts": 0, "recovered_logical_requests": 0, "exhausted_logical_requests": 0}
        for name in ("request_retries", "format_retries", "transport_retries", "output_limit_retries")
    }
    for key, attempts in groups.items():
        kinds = set()
        for number in attempts:
            if not number:
                continue
            kind = (
                "transport_retries"
                if attempts[number - 1] == "uncertain"
                else "output_limit_retries" if attempts[number - 1] == "billing_unknown" else "format_retries"
            )
            kinds.add(kind)
            totals[kind]["additional_attempts"] += 1
            totals["request_retries"]["additional_attempts"] += 1
        recovered = len(attempts) > 1 and "completed" in attempts.values()
        exhausted = key in failures and failures[key]["status"] in (
            "failed_after_format_retries",
            "failed_after_transport_retries",
            "failed_after_output_limit_retries",
        )
        for name in {"request_retries", *kinds}:
            totals[name]["recovered_logical_requests"] += recovered
            totals[name]["exhausted_logical_requests"] += exhausted
    return totals


class LocalTaskFailure(RuntimeError):
    """The authorized technical-retry budget is exhausted; not a global scheduler failure."""

    def __init__(self, key, attempts, reason, *, kind="failed_after_format_retries"):
        self.key, self.attempts, self.reason = key, attempts, reason
        self.kind = kind
        super().__init__(f"{key}: {kind} after {len(attempts)} attempts: {reason}")

    def document(self):
        return {
            "key": self.key,
            "attempt_keys": self.attempts,
            "reason": self.reason,
            "status": self.kind,
        }
