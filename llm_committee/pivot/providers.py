"""Small provider boundary: explicit models/settings, no tools or automatic retries.

Sources checked 2026-09-25:
https://developers.openai.com/api/docs/models/gpt-5.6-terra
https://developers.openai.com/api/docs/guides/prompt-caching
https://ai.google.dev/api/generate-content
https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol

from .models import AGREEMENT, OPEN_MODELS, Completion, Request, canonical, digest


class Provider(Protocol):
    def generate(self, request: Request) -> Completion: ...


@dataclass(frozen=True)
class Price:
    input: float
    cached: float
    output: float
    max_input: int
    write_multiplier: float = 1.0
    long_input_multiplier: float = 1.0
    long_output_multiplier: float = 1.0


PRICES = {
    "gpt-5.6-luna": Price(0.2, 0.02, 1.2, 922000, 1.25, 2, 1.5),
    "gpt-5.6-terra": Price(2, 0.2, 12, 922000, 1.25, 2, 1.5),
    "gpt-5.6-sol": Price(4, 0.4, 20, 922000, 1.25, 2, 1.5),
    "gemini-3.8-flash": Price(0.75, 0.075, 3.75, 1048576),
    "Qwen/Qwen3.8-27B": Price(1.86, 0.372, 5.595, 65536),
    "thinkingmachines/Inkling": Price(1.87, 0.374, 4.68, 65536),
}

# Databricks standard listed rates match these provider prices at $0.07/DBU.
# Conservatively include the documented 10% regional-processing uplift, without
# assuming account discounts or claiming these estimates are provider invoices.
# https://www.databricks.com/product/pricing/proprietary-foundation-model-serving
DATABRICKS_COST_MULTIPLIER = 1.1


def transport_multiplier(model: str, closed_provider: str) -> float:
    return DATABRICKS_COST_MULTIPLIER if closed_provider == "databricks" and model not in OPEN_MODELS else 1.0


def reservation_usd(request: Request, *, closed_provider: str = "direct") -> float:
    """Conservative per-call hold: full endpoint input limit, not a word/token guess.

    This can stop a small-budget run well before its nominal cap. Prices remain a
    dated API-cost model, not a guarantee about taxes or future provider repricing.
    """
    p = PRICES[request.model]
    # Reserve Gemini's entire documented output window, including possible thinking.
    output_bound = 65536 if request.model.startswith("gemini-") else request.max_output_tokens
    # Inkling has a temporary 50% discount; reserve at the published regular rate.
    reserve_multiplier = 2 if request.model == "thinkingmachines/Inkling" else 1
    return (
        transport_multiplier(request.model, closed_provider)
        * reserve_multiplier
        * (
            p.max_input * p.input * p.write_multiplier * p.long_input_multiplier
            + output_bound * p.output * p.long_output_multiplier
        )
        / 1e6
    )


def cost_usd(model: str, completion: Completion, *, closed_provider: str = "direct") -> float:
    if closed_provider == "openrouter":
        raw = completion.raw or {}
        value = (raw.get("response", {}).get("usage") or {}).get("cost")
        # Reuse a cost already present in the response. Missing cost metadata is
        # not an endpoint failure: fall through to the existing token estimate.
        if raw.get("provider") == "openrouter" and type(value) in (int, float) and math.isfinite(value) and value >= 0:
            return float(value)
    p = PRICES[model]
    counts = (
        completion.input_tokens,
        completion.output_tokens,
        completion.cached_tokens,
        completion.cache_write_tokens,
        completion.reasoning_tokens,
    )
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Missing or invalid billing token counts")
    normal = completion.input_tokens - completion.cached_tokens - completion.cache_write_tokens
    if normal < 0 or completion.reasoning_tokens > completion.output_tokens:
        raise ValueError("Inconsistent billing token counts")
    long_context = completion.input_tokens > 272000
    input_multiplier = p.long_input_multiplier if long_context else 1
    output_multiplier = p.long_output_multiplier if long_context else 1
    return (
        transport_multiplier(model, closed_provider)
        * (
            (
                normal * p.input
                + completion.cached_tokens * p.cached
                + completion.cache_write_tokens * p.input * p.write_multiplier
            )
            * input_multiplier
            + completion.output_tokens * p.output * output_multiplier
        )
        / 1e6
    )


def openai_payload(request: Request) -> dict:
    messages = [{"role": m.role, "content": [{"type": "input_text", "text": m.text}]} for m in request.messages]
    if request.cache:
        # Preserve content boundaries; cache up to four reusable prefixes before the task suffix.
        for message in messages[max(0, len(messages) - 5) : -1]:
            message["content"][0]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    payload = {
        "model": request.model,
        "input": messages,
        "reasoning": {"effort": request.effort},
        "max_output_tokens": request.max_output_tokens,
        "store": False,
        "tools": [],
        "truncation": "disabled",
        "prompt_cache_options": {"mode": "explicit"},
    }
    if request.schema is not None:
        payload["text"] = {
            "format": {"type": "json_schema", "name": "pilot_output", "schema": request.schema, "strict": True}
        }
    # No previous_response_id: branch history is explicitly reconstructed, never server-global.
    return payload


class OpenAIProvider:
    def __init__(self, client=None):
        if client is None:
            from openai import OpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is required; no credential discovery or fallback")
            client = OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=180)
        self.client = client

    def generate(self, request: Request) -> Completion:
        payload = openai_payload(request)
        cache = payload.pop("prompt_cache_options")
        raw = self.client.responses.create(**payload, extra_body={"prompt_cache_options": cache}).model_dump()
        usage = raw["usage"]
        details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        if request.effort == "none" and "reasoning_tokens" not in output_details:
            raise ValueError("Cannot verify direct measurement: provider omitted reasoning-token usage")
        text = "".join(
            part.get("text", "")
            for item in raw.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        return Completion(
            text,
            usage["input_tokens"],
            usage["output_tokens"],
            details.get("cached_tokens", 0),
            details.get("cache_write_tokens", 0),
            output_details.get("reasoning_tokens", 0),
            raw.get("status", "unknown"),
            raw,
        )


class GeminiProvider:
    def __init__(self, client=None):
        import httpx

        self.key = os.environ.get("GEMINI_API_KEY")
        if client is None and not self.key:
            raise ValueError("GEMINI_API_KEY is required for the explicitly selected judge")
        self.client = client or httpx.Client(timeout=180)

    def generate(self, request: Request) -> Completion:
        if request.model != "gemini-3.8-flash" or request.schema is None:
            raise ValueError("Gemini adapter is restricted to the structured pilot judge")
        response = self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent",
            headers={"x-goog-api-key": self.key or "mock"},
            json={
                "systemInstruction": {
                    "parts": [{"text": "\n".join(m.text for m in request.messages if m.role == "developer")}]
                },
                "contents": [
                    {"role": "user", "parts": [{"text": m.text} for m in request.messages if m.role != "developer"]}
                ],
                "generationConfig": {
                    "maxOutputTokens": request.max_output_tokens,
                    "thinkingConfig": {"thinkingLevel": "LOW"},
                    "responseMimeType": "application/json",
                    "responseJsonSchema": request.schema,
                },
            },
        )
        response.raise_for_status()
        raw = response.json()
        usage = raw["usageMetadata"]
        candidate = next(iter(raw.get("candidates", [])), {})
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thoughts = usage.get("thoughtsTokenCount", 0)
        return Completion(
            text,
            usage["promptTokenCount"],
            usage.get("candidatesTokenCount", 0) + thoughts,
            usage.get("cachedContentTokenCount", 0),
            0,
            thoughts,
            "completed" if candidate.get("finishReason") == "STOP" else "incomplete",
            raw,
        )


class LiveProviders:
    def __init__(
        self,
        *,
        judge: bool,
        mixed: bool = False,
        debate_effort: str = "medium",
        closed_provider: str = "databricks",
        databricks_profile: str = "un",
    ):
        self.databricks = self.openai = self.gemini = self.openrouter = None
        if closed_provider == "databricks":
            from .databricks_provider import DatabricksProvider

            self.databricks = DatabricksProvider(profile=databricks_profile)
        elif closed_provider == "direct":
            self.openai = OpenAIProvider()
            self.gemini = GeminiProvider() if judge else None
        elif closed_provider == "openrouter":
            if mixed:
                raise ValueError("OpenRouter mixed-family native probability reads are not implemented; no Tinker fallback")
            from .openrouter_provider import OpenRouterProvider

            self.openrouter = OpenRouterProvider()
        else:
            raise ValueError("Unknown closed-model transport; no fallback")
        self.judge = judge
        self.tinker = None
        if mixed:
            from .tinker_provider import TinkerProvider

            self.tinker = TinkerProvider()
            self.tinker.prepare(debate_effort)

    def generate(self, request: Request) -> Completion:
        if self.openrouter:
            if request.model == "gemini-3.8-flash" and not self.judge:
                raise ValueError("Judge access was not enabled")
            return self.openrouter.generate(request)
        if self.databricks and (
            request.model.startswith("gpt-5.6-") or (request.model == "gemini-3.8-flash" and self.judge)
        ):
            return self.databricks.generate(request)
        if request.model.startswith("gpt-5.6-") and request.model in PRICES:
            return self.openai.generate(request)
        if request.model == "gemini-3.8-flash" and self.gemini:
            return self.gemini.generate(request)
        if request.model in OPEN_MODELS and self.tinker:
            return self.tinker.generate(request)
        raise ValueError(f"No validated pilot adapter for {request.model}; no model substitution")

    def token_count(self, model: str, text: str) -> int:
        if self.tinker is None or model not in OPEN_MODELS:
            raise ValueError("Filler matching requires the receiver's native tokenizer")
        return self.tinker.token_count(model, text)

    def close(self, status: str = "interrupted") -> None:
        if self.openrouter:
            self.openrouter.close(status)
        if self.databricks:
            self.databricks.close()
        if self.openai:
            self.openai.client.close()
        if self.gemini:
            self.gemini.client.close()
        if self.tinker:
            self.tinker.close(status)


class MockProvider:
    """Deterministic engineering fixture, NEVER scientific/model output."""

    def __init__(self):
        self.calls: list[Request] = []

    def token_count(self, model: str, text: str) -> int:
        return len(text.split())  # Synthetic fixtures only; live matching uses the native tokenizer.

    def synthetic_readout(self, request):
        import math

        labels = request.candidate_labels
        weights = [1 + int(digest([request.key, label])[:6], 16) % 9 for label in labels]
        prefix = [int(digest(request.key)[:6], 16)]
        return {
            "candidate_ids": {label: 65 + i for i, label in enumerate(labels)},
            "candidate_logprobs": {
                label: math.log(0.98 * w / sum(weights)) for label, w in zip(labels, weights, strict=True)
            },
            "prefix_token_ids": prefix,
            "prefix_sha256": digest(prefix),
            "sampled_token_id": 65,
            "sampled_token_logprob": math.log(0.98 * weights[0] / sum(weights)),
            "sampled_output_index": 0,
            "source": "synthetic_sample_topk",
            "sampling_temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "read_temperature": 1.0,
        }

    def generate(self, request: Request) -> Completion:
        self.calls.append(request)
        i = int(digest(request.key)[:8], 16)
        if request.purpose == "initial":
            value = {"position": f"SYNTHETIC initial {request.key}"}
        elif request.purpose == "debate":
            value = {"reply": f"SYNTHETIC reply {request.key}", "agreement": AGREEMENT[i % 4]}
        elif request.purpose == "position":
            return Completion(
                f"A\nSYNTHETIC position {request.key}",
                100,
                30,
                raw={"mock": True},
                readout=self.synthetic_readout(request) if request.candidate_labels else None,
            )
        elif request.purpose == "d_text":
            return Completion("A", 100, 1, raw={"mock": True}, readout=self.synthetic_readout(request))
        elif request.purpose == "candidate_scores":
            return Completion(
                canonical(request.scoring), len(request.scoring["prefix_token_ids"]) + 1, 1, raw={"mock": True}
            )
        elif request.purpose == "judge_b":
            value = {"label": AGREEMENT[i % 4], "evidence": "SYNTHETIC evidence"}
        elif request.purpose == "judge_c":
            value = {"label": "unchanged", "evidence": "SYNTHETIC evidence"}
        elif request.purpose == "judge_e":
            value = {"preference": "left" if i % 2 else "right", "evidence": "SYNTHETIC evidence"}
        elif request.purpose == "synthesis":
            value = {"answer": f"SYNTHETIC synthesis {request.key}"}
        else:
            raise ValueError("Unknown mock purpose")
        return Completion(canonical(value), 100, 100, reasoning_tokens=50, raw={"mock": True})
