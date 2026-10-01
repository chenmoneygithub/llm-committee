"""Existing workspace OAuth, exact endpoints, raw chat responses, no generation retries.

Uses the same Databricks profile/workspace-header convention as databricks_lm.py,
without its DSPy parsing/retry/response-cache behavior. Docs checked 2026-09-25:
https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference
https://docs.databricks.com/aws/en/machine-learning/model-serving/query-reason-models
"""

from __future__ import annotations

import threading

from .models import Completion, Request

# SDK Config/CLI OAuth refresh is shared per profile, not initialized independently
# by every request worker. Serialize auth access only; model HTTP requests stay parallel.
_AUTH_LOCK = threading.RLock()
_AUTH_CONFIGS = {}


def _new_workspace_config(profile):
    from databricks.sdk.config import Config

    return Config(profile=profile, http_timeout_seconds=30, retry_timeout_seconds=1)


def _workspace_config(profile):
    with _AUTH_LOCK:
        if profile not in _AUTH_CONFIGS:
            config = _new_workspace_config(profile)
            config.authenticate()
            _AUTH_CONFIGS[profile] = config
        return _AUTH_CONFIGS[profile]


ENDPOINTS = {
    "gpt-5.6-luna": "databricks-gpt-5-6-luna",
    "gpt-5.6-terra": "databricks-gpt-5-6-terra",
    "gpt-5.6-sol": "databricks-gpt-5-6-sol",
    "gemini-3.8-flash": "databricks-gemini-3-8-flash",
}


def databricks_payload(request: Request) -> dict:
    if request.model not in ENDPOINTS:
        raise ValueError("No exact Databricks endpoint mapping; no model substitution")
    if request.cache:
        raise ValueError("Explicit cache breakpoints are not supported by this Databricks adapter")
    if request.candidate_labels or request.scoring is not None:
        raise ValueError("D reads must use the native Tinker adapter")
    if request.model.startswith("gemini-") and (request.effort != "low" or request.schema is None):
        raise ValueError("Databricks Gemini is restricted to the low-reasoning structured judge")
    messages = []
    for i, message in enumerate(request.messages):
        role = "system" if message.role == "developer" else message.role
        if role not in ("system", "user", "assistant") or (role == "system" and i != 0):
            raise ValueError("Expected one leading instruction and explicit branch messages")
        messages.append({"role": role, "content": message.text})
    payload = {
        "messages": messages,
        "max_tokens": request.max_output_tokens,
        "reasoning_effort": request.effort,
        "stream": False,
        "service_tier": "default",
    }
    if request.schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "pilot_output", "schema": request.schema, "strict": True},
        }
    return payload


def parse_databricks_response(request: Request, raw: dict, provenance: dict) -> Completion:
    """Return invalid responses too, so the journal retains evidence before stopping."""
    usage = raw.get("usage") or {}
    reasoning = usage.get("reasoning_tokens", (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    output = usage.get("completion_tokens")
    accounting = "completion_tokens_includes_reasoning"
    if request.model.startswith("gemini-"):
        # Databricks' Gemini chat response can report visible completion and reasoning
        # separately. Reconcile against its total; never omit or double-bill thinking.
        prompt, total = usage.get("prompt_tokens"), usage.get("total_tokens")
        if all(type(n) is int and n >= 0 for n in (prompt, output, total)) and reasoning is None:
            if total == prompt + output:
                accounting = "total_verified_output_reasoning_partition_unreported"
            else:
                output, accounting = None, "inconsistent_usage_totals"
        elif all(type(n) is int and n >= 0 for n in (prompt, output, reasoning, total)):
            if total == prompt + output + reasoning:
                output += reasoning
                accounting = "total_verified_completion_plus_separate_reasoning"
            elif total == prompt + output and reasoning <= output:
                accounting = "total_verified_completion_includes_reasoning"
            else:
                output, accounting = None, "inconsistent_usage_totals"
        else:
            output, accounting = None, "missing_usage_totals"
    cached = usage.get("cache_read_input_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    writes = usage.get(
        "cache_creation_input_tokens", (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens", 0)
    )
    choices = raw.get("choices") or []
    choice = choices[0] if len(choices) == 1 else {}
    message = choice.get("message") or {}
    content = message.get("content")
    status = "completed" if choice.get("finish_reason") == "stop" else "incomplete"
    text = ""
    thinking_present = bool(message.get("reasoning_content"))
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                text += part.get("text", "")
            elif part.get("type") in ("reasoning", "thinking"):
                thinking_present = True
            else:
                status = "unsupported_content"
    else:
        status = "missing_content"
    if message.get("tool_calls") or message.get("role") != "assistant":
        status = "unexpected_message"
    if request.effort == "none":
        if reasoning is None:
            status = "unverified_direct_read"
        elif reasoning != 0 or thinking_present:
            status = "unexpected_reasoning"
    return Completion(
        text,
        usage.get("prompt_tokens"),
        output,
        cached,
        writes,
        reasoning if reasoning is not None else 0,
        status,
        {
            **provenance,
            "response": raw,
            "usage_accounting": accounting,
            "reasoning_usage_reported": reasoning is not None,
        },
    )


class DatabricksProvider:
    def __init__(self, *, profile: str = "un", config=None, client=None):
        if config is None:
            # SDK auth only: it resolves the project's existing OAuth profile and refreshes
            # when needed. Never export/log the token or repurpose it as an OpenAI key.
            config = _workspace_config(profile)
        if not config.host or not config.host.startswith("https://"):
            raise ValueError("Databricks profile must have an HTTPS workspace host")
        if client is None:
            import httpx

            client = httpx.Client(timeout=180, follow_redirects=False, transport=httpx.HTTPTransport(retries=0))
        self.config = config
        self.client = client
        self.profile = profile

    def generate(self, request: Request) -> Completion:
        payload = databricks_payload(request)
        endpoint = ENDPOINTS[request.model]
        with _AUTH_LOCK:
            headers = {**self.config.authenticate(), "Content-Type": "application/json"}
        if self.config.workspace_id:
            headers["X-Databricks-Workspace-Id"] = str(self.config.workspace_id)
        response = self.client.post(
            self.config.host.rstrip("/") + f"/serving-endpoints/{endpoint}/invocations",
            headers=headers,
            json=payload,
        )
        # Headers (including OAuth) are never put into saved provenance.
        provenance = {
            "provider": "databricks",
            "profile": self.profile,
            "endpoint": endpoint,
            "request_payload": payload,
            "http_status": response.status_code,
            "served_model_name": response.headers.get("served-model-name"),
        }
        try:
            raw = response.json()
        except ValueError:
            raw = {"non_json_response": response.text}
        if not isinstance(raw, dict):
            raw = {"unexpected_response": raw}
        if response.status_code != 200:
            # Unknown usage keeps the full reservation; retain the provider's error body.
            return Completion("", None, None, status="http_error", raw={**provenance, "response": raw})
        return parse_databricks_response(request, raw, provenance)

    def close(self) -> None:
        self.client.close()
