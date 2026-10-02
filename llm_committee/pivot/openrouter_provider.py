"""OpenRouter-only transport; preserve normal responses without extra API calls.

No company credentials, direct OpenAI key, or Databricks profile is consulted.
"""

from __future__ import annotations

import os

import httpx

from .models import Completion, Request

BASE_URL = "https://openrouter.ai/api/v1"
MODELS = {
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gemini-3.8-flash": "google/gemini-3.8-flash",
}


def openrouter_payload(request: Request) -> dict:
    if request.model not in MODELS or request.candidate_labels or request.scoring or request.cache:
        raise ValueError("Unsupported OpenRouter model/probability/cache request; no substitution")
    if request.effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unknown reasoning effort")
    if any(m.role not in ("developer", "system", "user", "assistant") for m in request.messages):
        raise ValueError("Unsupported message role")
    payload = {
        "model": MODELS[request.model],
        # Match the prior Databricks adapter's developer-to-system conversion.
        "messages": [
            {"role": "system" if m.role == "developer" else m.role, "content": m.text}
            for m in request.messages
        ],
        "reasoning": {"effort": request.effort},
        "max_tokens": request.max_output_tokens,
        "stream": False,
        "provider": {"require_parameters": True, "data_collection": "deny"},
    }
    if request.schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "pilot_output", "strict": True, "schema": request.schema},
        }
    return payload


def parse_openrouter_response(request: Request, body: dict, payload: dict, http_status: int = 200) -> Completion:
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    choices = body.get("choices") or []
    choice = choices[0] if len(choices) == 1 else {}
    message = choice.get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    status = "completed" if choice.get("finish_reason") == "stop" else "incomplete"
    if http_status != 200 or body.get("error"):
        status = "http_error"
    elif (
        len(choices) != 1 or message.get("role") != "assistant"
        or message.get("tool_calls") or message.get("refusal")
        or (content is not None and not isinstance(content, str))
    ):
        status = "unexpected_message"
    elif body.get("model") != MODELS[request.model]:
        # Preserve any received output and charge, but do not accept a substituted model.
        status = "unexpected_model"
    elif request.effort == "none":
        if "reasoning_tokens" not in output_details:
            status = "unverified_direct_read"
        elif output_details["reasoning_tokens"] or message.get("reasoning") or message.get("reasoning_details"):
            status = "unexpected_reasoning"
    return Completion(
        text=text,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cached_tokens=details.get("cached_tokens", 0),
        cache_write_tokens=details.get("cache_write_tokens", 0),
        reasoning_tokens=output_details.get("reasoning_tokens", 0),
        status=status,
        raw={
            "provider": "openrouter",
            "endpoint": BASE_URL + "/chat/completions",
            "http_status": http_status,
            "request_payload": payload,
            "response": body,
        },
    )


class OpenRouterProvider:
    def __init__(self, client=None, *, api_key=None):
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required; no other credential/provider fallback")
        self._key = key
        self.client = client if client is not None else httpx.Client(timeout=180, follow_redirects=False)

    def generate(self, request: Request) -> Completion:
        payload = openrouter_payload(request)
        response = self.client.post(
            BASE_URL + "/chat/completions",
            headers={"Authorization": "Bearer " + self._key},
            json=payload,
        )
        # HTTP failures are preserved with any returned billing information; never
        # archive request headers or an httpx exception containing credentials.
        return parse_openrouter_response(request, response.json(), payload, response.status_code)

    def close(self, status="success"):
        self.client.close()
