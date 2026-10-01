"""Native Tinker sampling with exact-token probability provenance, no hidden scoring calls.

Sources: Tinker 0.30.3 SamplingClient/target_prompt_logprobs; Cookbook 0.5.7
Qwen3.8 and TMLv0 renderers. Importing this module does not load SDKs or credentials.
"""

from __future__ import annotations

import math
import os
from importlib.metadata import version

from .models import OPEN_MODELS, Completion, Request, canonical, digest
from .probabilities import capture_readout

CONTEXT_LIMIT = 65536
EFFORT = {"none": 0.0, "low": 0.2, "medium": 0.7, "xhigh": 0.99}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Retain invalid raw evidence without emitting non-standard JSON.
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    return value


def make_renderer(model: str, effort: str):
    """Local rendering/tokenizer loading only; no Tinker session or model call."""
    if model not in OPEN_MODELS or effort not in EFFORT:
        raise ValueError("Unsupported exact Tinker model/effort; no substitutions")
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model)
    if model.startswith("Qwen/"):
        from tinker_cookbook.renderers.qwen3_8 import Qwen3_8DisableThinkingRenderer, Qwen3_8Renderer

        renderer = (
            Qwen3_8DisableThinkingRenderer(tokenizer)
            if effort == "none"
            else Qwen3_8Renderer(tokenizer, reasoning_effort=effort)
        )
    else:
        from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

        renderer = TmlV0Renderer(tokenizer)
    return tokenizer, renderer


def message_content(message: dict) -> tuple[str, str]:
    if message.get("role") != "assistant" or message.get("tool_calls") or message.get("unparsed_tool_calls"):
        raise ValueError("Expected an assistant response without tools")
    content = message.get("content", "")
    thinking = message.get("reasoning_content") or ""
    if isinstance(content, str):
        return content, thinking
    texts = []
    for part in content:
        if part["type"] == "text":
            texts.append(part["text"])
        elif part["type"] == "thinking":
            thinking += part["thinking"]
        else:
            raise ValueError("Non-text model output is outside the pilot")
    return "".join(texts), thinking


class TinkerProvider:
    def __init__(self, service=None, *, sdk=None, renderer_factory=make_renderer):
        if sdk is None:
            import tinker as sdk
        self.sdk = sdk
        self.renderer_factory = renderer_factory
        self.renderers = {}
        self.clients = {}
        if service is None:
            key = os.environ.get("TINKER_API_KEY")
            if not key:
                raise ValueError("TINKER_API_KEY is required; no credential discovery")
            # Disable configurable HTTP/outer sampling retries. SDK-internal transport
            # retry/backpressure for the same request ID still exists in 0.30.3.
            service = sdk.ServiceClient(api_key=key, max_retries=0, timeout=180)
        self.service = service

    def renderer_for(self, model, effort):
        key = (model, effort)
        if key not in self.renderers:
            self.renderers[key] = self.renderer_factory(model, effort)
        return self.renderers[key]

    def token_count(self, model, text):
        tokenizer, _ = self.renderer_for(model, "none")
        return len(tokenizer.encode(text, add_special_tokens=False))

    def prepare(self, debate_effort):
        """Validate local renderers before any committee request; no sampling session."""
        from .probabilities import RATING_LABELS, candidate_ids

        for model in sorted(OPEN_MODELS):
            tokenizer, _ = self.renderer_for(model, "none")
            candidate_ids(tokenizer, RATING_LABELS)
            self.renderer_for(model, debate_effort)

    def client_for(self, model):
        if model not in OPEN_MODELS:
            raise ValueError("Unapproved Tinker model")
        if model not in self.clients:
            from tinker.lib.retry_handler import RetryConfig

            client = self.service.create_sampling_client(
                base_model=model,
                retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=180),
            )
            if client.get_base_model() != model:
                raise ValueError("Tinker returned a different base model")
            self.clients[model] = client
        return self.clients[model]

    def generate(self, request: Request) -> Completion:
        if request.scoring is not None:
            return self.score(request)
        tokenizer, renderer = self.renderer_for(request.model, request.effort)
        messages = [
            {"role": "system" if m.role == "developer" else m.role, "content": m.text} for m in request.messages
        ]
        kwargs = {"effort": EFFORT[request.effort]} if request.model == "thinkingmachines/Inkling" else {}
        prompt = renderer.build_generation_prompt(messages, **kwargs)
        prompt_ids = prompt.to_ints()
        if len(prompt_ids) + request.max_output_tokens > CONTEXT_LIMIT:
            raise ValueError("Tinker context limit exceeded; no truncation")
        result = (
            self.client_for(request.model)
            .sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=self.sdk.SamplingParams(
                    max_tokens=request.max_output_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    stop=renderer.get_stop_sequences(),
                ),
                topk_sample_logprobs=20 if request.candidate_labels else 0,
            )
            .result(timeout=180)
        )
        sequence = result.sequences[0]
        tokens = sequence.tokens
        raw = {
            "provider": "tinker",
            "model": request.model,
            "renderer": type(renderer).__name__,
            "effort": request.effort,
            "packages": {name: version(name) for name in ("tinker", "tinker-cookbook", "tml-renderers")},
            "prompt_token_ids": prompt_ids,
            "sampled_token_ids": tokens,
            "sampled_logprobs": sequence.logprobs,
            "topk_logprobs": sequence.topk_logprobs,
            "sequence_id": sequence.sequence_id,
            "stop_reason": sequence.stop_reason,
            "prompt_cache_hit_tokens": result.prompt_cache_hit_tokens,
            "sampling_temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "cache_policy": "provider-managed; actual cache-hit count recorded",
            "retry_policy": "No application resampling/outer retry; SDK may retry transport of the same request ID",
        }
        text, reasoning, status, readout = "", "", "completed", None
        try:
            message, termination = renderer.parse_response(tokens)
            raw["parsed_message"] = message
            raw["parse_termination"] = str(termination)
            text, reasoning = message_content(message)
            if not termination.is_clean or sequence.stop_reason != "stop":
                status = "incomplete"
            if request.effort == "none" and (reasoning.strip() or "<think>" in text.lower()):
                status = "unexpected_reasoning"
            if request.candidate_labels and status == "completed":
                readout = capture_readout(
                    tokenizer,
                    prompt_ids,
                    tokens,
                    text,
                    request.candidate_labels,
                    sequence.topk_logprobs,
                    sequence.logprobs,
                )
                readout["sequence_id"] = sequence.sequence_id
                raw["probability_readout"] = readout
        except (ValueError, KeyError, TypeError) as exc:
            status = "invalid_native_output"
            raw["validation_error"] = str(exc)
        # This diagnostic counts retokenized reasoning text, not a provider-reported partition.
        # Billing always uses the actual full sampled token sequence length.
        reasoning_count = len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0
        raw["reasoning_count_method"] = "retokenized parsed reasoning; billing uses full sampled sequence"
        return Completion(
            text,
            len(prompt_ids),
            len(tokens),
            result.prompt_cache_hit_tokens,
            reasoning_tokens=reasoning_count,
            status=status,
            raw=json_safe(raw),
            readout=json_safe(readout),
        )

    def score(self, request: Request) -> Completion:
        """One journaled sampling call scores all candidates at the exact saved prefix.

        Appending the previously sampled token makes its position scoreable as a prompt
        token. The one new token required by Tinker's API is billed but never used as data.
        """
        saved = request.scoring
        if request.effort != "none" or request.max_output_tokens != 1 or not saved:
            raise ValueError("Invalid scoring-only request")
        prefix = saved["prefix_token_ids"]
        if digest(prefix) != saved["prefix_sha256"] or set(saved["candidate_ids"]) != set(request.candidate_labels):
            raise ValueError("Scoring prefix/candidates changed")
        prompt_ids = [*prefix, saved["sampled_token_id"]]
        if len(prompt_ids) + 1 > CONTEXT_LIMIT:
            raise ValueError("Scoring prefix exceeds the endpoint context limit")
        # Row i predicts prompt token i+1. Only the final row is populated (CSR).
        rows, width = len(prompt_ids) - 1, len(request.candidate_labels)
        ids = [saved["candidate_ids"][label] for label in request.candidate_labels]
        target = self.sdk.TensorData(
            data=ids,
            dtype="int64",
            shape=[rows, width],
            sparse_crow_indices=[0] * rows + [width],
            sparse_col_indices=list(range(width)),
        )
        result = (
            self.client_for(request.model)
            .sample(
                prompt=self.sdk.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=self.sdk.SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1),
                include_prompt_logprobs=True,
                target_prompt_logprobs=target,
            )
            .result(timeout=180)
        )
        sequence = result.sequences[0]
        raw = {
            "provider": "tinker",
            "model": request.model,
            "prompt_token_ids": prompt_ids,
            "scored_position": rows,
            "target_ids": ids,
            "target_prompt_logprobs": (
                result.target_prompt_logprobs.to_numpy().tolist() if result.target_prompt_logprobs is not None else None
            ),
            "prompt_logprobs": result.prompt_logprobs,
            "ignored_sampled_token_ids": sequence.tokens,
            "prompt_cache_hit_tokens": result.prompt_cache_hit_tokens,
            "sequence_id": sequence.sequence_id,
        }
        try:
            values = raw["target_prompt_logprobs"]
            if len(values) != rows or len(values[-1]) != width:
                raise ValueError("Wrong target-score tensor shape")
            scored = {
                **saved,
                "candidate_logprobs": dict(zip(request.candidate_labels, values[-1], strict=True)),
                "sampled_token_logprob": result.prompt_logprobs[-1],
                "source": "target_prompt_logprobs",
            }
            text, status = canonical(scored), "completed"
        except (TypeError, ValueError, IndexError) as exc:
            raw["validation_error"] = str(exc)
            text, status = "", "invalid_scores"
        return Completion(
            text,
            len(prompt_ids),
            len(sequence.tokens),
            result.prompt_cache_hit_tokens,
            status=status,
            raw=json_safe(raw),
        )

    def close(self, status="interrupted"):
        self.service.close(status).result(timeout=30)
