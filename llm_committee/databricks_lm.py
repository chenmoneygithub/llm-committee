"""
Databricks model-serving integration for DSPy.

Connects DSPy/litellm to Databricks Foundation Model serving endpoints, handling
the two non-obvious quirks of this workspace:

1. The host is an account-level *vanity* host (e.g. ``https://myaccount.databricks.com``).
   Naive HTTP clients get a 303 redirect to a login page. The fix is to send the
   ``X-Databricks-Workspace-Id`` header (the SDK's native ``query()`` does this).

2. Some models (notably ``databricks-gpt-5-5-pro``) are **Responses-API-only** and
   reject chat-completions calls. Those need ``model_type="responses"``.

Auth uses the Databricks SDK's OAuth resolution (profile ``un`` by default), so no
secrets are hard-coded. The OAuth access token is short-lived and re-fetched per
process.

Usage::

    from llm_committee.databricks_lm import make_lm
    opus = make_lm("databricks-claude-opus-4-8")            # chat model
    pro  = make_lm("databricks-gpt-5-5-pro")                # auto-detected responses model
"""

from __future__ import annotations

import functools

import dspy

# Default per-request HTTP timeout (seconds) and retry count for EVERY Databricks transport here.
# Both litellm and the databricks SDK default to NO request timeout (SDK: http_timeout_seconds /
# retry_timeout_seconds both None), so a stalled socket hangs the process forever at ~0% CPU with no
# self-recovery (observed: a 30-min Layer-C hang on the SDK/gemini path). These bound every call.
# The SDK's retry model is a *time window* (not a count), so we size that window to fit the initial
# attempt plus ``num_retries`` retries of up to ``timeout`` each: retry_timeout = timeout*(n+1).
DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0
DEFAULT_NUM_RETRIES = 3

# Endpoints that only accept the Responses API (chat-completions returns BadRequest).
RESPONSES_ONLY_ENDPOINTS = {
    "databricks-gpt-5-5-pro",
}

# Endpoints that return chat content as a *structured block list* (e.g. reasoning models, or
# Gemini) rather than a plain string. litellm's OpenAI-compat parser rejects these, so we route
# them through the SDK-native transport and flatten the content ourselves. See StructuredContentLM.
STRUCTURED_CONTENT_ENDPOINTS = {
    "databricks-qwen35-122b-a10b",
    "databricks-gemini-3-5-flash",
}

# Endpoints that *intermittently* reject ``response_format={"type": "json_object"}``. DSPy defaults
# to ChatAdapter and auto-falls-back to JSONAdapter on any parse hiccup; that fallback sets
# json_object whenever ``lm.supports_response_schema`` is False (true here) AND response_format is in
# ``lm.supported_params`` (litellm wrongly reports True for this endpoint). The endpoint then flakily
# 400s ("Response format type json_object is not supported"), causing differential data loss on
# gpt-5.5 cells. Dropping the param via litellm forces the always-working text-based JSON prompting
# path (which the adapter already emits). NOTE: this is the *chat* gpt-5-5 endpoint ONLY — gpt-5-5-pro
# is a different (Responses-API) transport and Gemini is StructuredContentLM; neither belongs here.
JSON_OBJECT_UNSUPPORTED_ENDPOINTS = {
    "databricks-gpt-5-5",
}

# Friendly short aliases -> Databricks endpoint names, for convenience at call sites.
MODEL_ALIASES = {
    "opus-4.8": "databricks-claude-opus-4-8",
    "opus": "databricks-claude-opus-4-8",
    "gpt-5.5-pro": "databricks-gpt-5-5-pro",
    "gpt-5.5": "databricks-gpt-5-5",
    "gpt5.5pro": "databricks-gpt-5-5-pro",
    "qwen": "databricks-qwen35-122b-a10b",
    "qwen3.5": "databricks-qwen35-122b-a10b",
    "gemini": "databricks-gemini-3-5-flash",
    "gemini-flash": "databricks-gemini-3-5-flash",
    "sonnet": "databricks-claude-sonnet-4-6",
    "sonnet-4.6": "databricks-claude-sonnet-4-6",
}


@functools.lru_cache(maxsize=1)
def _resolve_connection(profile: str) -> tuple[str, str, str]:
    """Resolve (base_url, token, workspace_id) once per process.

    Cached because constructing the WorkspaceClient and authenticating is not free,
    and the values are stable for the life of the process.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

    # Bound the auth/token-resolution call with the same default timeouts as the query transport,
    # so a stalled metadata/token endpoint can't hang process startup forever. Timeouts live on
    # Config (WorkspaceClient(**) does NOT accept http_timeout_seconds directly in this SDK).
    client = WorkspaceClient(config=Config(
        profile=profile,
        http_timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS,
        retry_timeout_seconds=DEFAULT_HTTP_TIMEOUT_SECONDS * (DEFAULT_NUM_RETRIES + 1),
    ))
    cfg = client.config
    auth_header = cfg.authenticate()["Authorization"]
    token = auth_header.split(" ", 1)[1]
    base_url = cfg.host.rstrip("/") + "/serving-endpoints"
    workspace_id = str(cfg.workspace_id) if cfg.workspace_id else ""
    return base_url, token, workspace_id


def _flatten_content(content) -> str:
    """Flatten a structured chat-content value into a plain string.

    Some Databricks endpoints (reasoning models like qwen3.5, and Gemini) return
    ``message.content`` as a list of typed blocks, e.g.
    ``[{"type": "reasoning", ...}, {"type": "text", "text": "Four"}]``. DSPy/litellm expect a
    string. We concatenate the visible ``text`` blocks; if there are none (e.g. the model spent
    its whole budget on a ``reasoning`` block and was truncated), we fall back to the reasoning
    summary text so the caller at least sees something rather than an empty string.

    Args:
        content: Either a string (returned unchanged) or a list of content blocks.

    Returns:
        A plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            return "".join(texts)
        # No text block (likely truncated mid-reasoning) — surface the reasoning summary.
        summaries = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "reasoning":
                for s in b.get("summary", []) or []:
                    if isinstance(s, dict) and s.get("text"):
                        summaries.append(s["text"])
        return "".join(summaries)
    return str(content)


class StructuredContentLM(dspy.LM):
    """A ``dspy.LM`` for Databricks endpoints that return structured (list) chat content.

    Routes completions through the Databricks SDK's native ``serving_endpoints.query`` (which
    handles the OAuth federation + workspace-id header correctly) instead of litellm's
    OpenAI-compat path, then flattens any list-shaped ``message.content`` to a string before
    handing a normal litellm-style ``ModelResponse`` back to DSPy. This makes reasoning models
    (qwen3.5) and Gemini usable as ordinary committee members.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        profile: str = "un",
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        num_retries: int = DEFAULT_NUM_RETRIES,
        **kwargs,
    ):
        # Register with DSPy as a normal chat LM; we override forward() for transport. dspy's own
        # num_retries wrapper does NOT apply to an overridden forward(), so retries on THIS path come
        # from the SDK's time-boxed retry window below, not from dspy — hence we store both.
        super().__init__(f"openai/{endpoint}", model_type="chat", num_retries=num_retries, **kwargs)
        self._endpoint = endpoint
        self._profile = profile
        self._http_timeout_seconds = http_timeout_seconds
        self._num_retries = num_retries

    def forward(self, prompt: str | None = None, messages: list | None = None, **kwargs):
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.config import Config
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        messages = messages or [{"role": "user", "content": prompt}]
        role_map = {
            "system": ChatMessageRole.SYSTEM,
            "user": ChatMessageRole.USER,
            "assistant": ChatMessageRole.ASSISTANT,
        }
        sdk_messages = [
            ChatMessage(role=role_map.get(m.get("role", "user"), ChatMessageRole.USER), content=str(m.get("content", "")))
            for m in messages
        ]

        merged = {**self.kwargs, **kwargs}
        max_tokens = merged.get("max_tokens", 8192)
        temperature = merged.get("temperature", 1.0)

        # Bound this call: serving_endpoints.query() has no per-call timeout kwarg, so the limit must
        # live on the client's Config. http_timeout_seconds caps each attempt; retry_timeout_seconds
        # is the SDK's total retry *window* (it retries transient failures until this elapses), sized
        # to fit the initial attempt plus num_retries retries. Without these this call is unbounded —
        # this is the exact path that hung the Layer-C relabel for 30+ min at ~0% CPU.
        client = WorkspaceClient(config=Config(
            profile=self._profile,
            http_timeout_seconds=self._http_timeout_seconds,
            retry_timeout_seconds=self._http_timeout_seconds * (self._num_retries + 1),
        ))
        resp = client.serving_endpoints.query(
            name=self._endpoint,
            messages=sdk_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Flatten structured content on each choice and rebuild a litellm ModelResponse so the
        # rest of DSPy (adapters, parsing, usage tracking) sees a standard shape.
        import litellm

        choices = []
        for ch in resp.choices:
            raw = ch.message.content
            text = _flatten_content(raw)
            choices.append(
                {
                    "index": getattr(ch, "index", 0) or 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": getattr(ch, "finish_reason", "stop") or "stop",
                }
            )
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            }
        return litellm.ModelResponse(
            model=self._endpoint,
            choices=choices,
            usage=usage,
        )


def make_lm(
    endpoint: str,
    *,
    profile: str = "un",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    cache: bool = False,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    num_retries: int = DEFAULT_NUM_RETRIES,
    **kwargs,
) -> dspy.LM:
    """Build a ``dspy.LM`` pointed at a Databricks serving endpoint.

    Args:
        endpoint: Databricks endpoint name (e.g. ``databricks-claude-opus-4-8``) or a
            short alias from ``MODEL_ALIASES`` (e.g. ``opus-4.8``, ``gpt-5.5-pro``).
        profile: Databricks CLI profile to authenticate with (default ``un``).
        max_tokens: Max output tokens.
        temperature: Sampling temperature. Responses-only models (gpt-5-5-pro) require
            temperature=1.0; this is enforced automatically.
        cache: Whether DSPy should cache responses (default False for experiments).
        timeout: Per-request HTTP timeout in seconds. Bounds BOTH transports — the litellm path
            (forwarded to litellm as ``timeout``) and the SDK path (mapped to the client
            ``Config``'s ``http_timeout_seconds``) — so a stalled endpoint can't hang the process
            forever. This is an EXPLICIT named param (not ``**kwargs``) so callers can pass
            ``timeout=`` without a dict double-keyword collision; override for unusually long calls.
        num_retries: Retry count on transient failures. On the litellm path this is dspy's
            first-class ``num_retries``; on the SDK path it sizes the SDK's retry *time window*
            (``timeout * (num_retries + 1)``), since dspy's retry wrapper does not apply to that
            path's overridden ``forward()``.
        **kwargs: Extra args forwarded to ``dspy.LM`` (must NOT include ``timeout``/``num_retries``
            — those are named params above).

    Returns:
        A configured ``dspy.LM`` instance.
    """
    endpoint = MODEL_ALIASES.get(endpoint, endpoint)

    # Structured-content endpoints (reasoning models, Gemini) bypass litellm's OpenAI-compat
    # parser via the SDK-native transport. Reasoning models burn output budget on a hidden
    # reasoning block before emitting text, so give them generous headroom by default.
    if endpoint in STRUCTURED_CONTENT_ENDPOINTS:
        sc_max = max(max_tokens, 4096)
        return StructuredContentLM(
            endpoint, profile=profile, max_tokens=sc_max, temperature=temperature, cache=cache,
            http_timeout_seconds=timeout, num_retries=num_retries, **kwargs
        )

    base_url, token, workspace_id = _resolve_connection(profile)

    extra_headers = {"X-Databricks-Workspace-Id": workspace_id} if workspace_id else {}

    lm_kwargs = dict(
        api_base=base_url,
        api_key=token,
        extra_headers=extra_headers,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=cache,
        # Bound the litellm path: without an explicit timeout litellm falls back to ~600s/phase.
        # ``timeout`` flows through dspy.LM.kwargs into the litellm request; num_retries is
        # dspy.LM's first-class retry count (default 3, set explicitly for clarity). Both are
        # make_lm named params, so a caller's timeout=/num_retries= binds them here (no collision).
        timeout=timeout,
        num_retries=num_retries,
        **kwargs,
    )

    if endpoint in RESPONSES_ONLY_ENDPOINTS:
        lm_kwargs["model_type"] = "responses"
        # Responses-only frontier models require temperature=1.0.
        lm_kwargs["temperature"] = 1.0

    if endpoint in JSON_OBJECT_UNSUPPORTED_ENDPOINTS:
        # Strip response_format before the API call so DSPy's JSONAdapter fallback can't send the
        # json_object this endpoint flakily rejects; it degrades to text-based JSON prompting instead.
        lm_kwargs["additional_drop_params"] = ["response_format"]

    # litellm routes "openai/<name>" + api_base to the OpenAI-compatible path, which
    # Databricks serving implements. The workspace-id header avoids the 303 redirect.
    return dspy.LM(f"openai/{endpoint}", **lm_kwargs)
