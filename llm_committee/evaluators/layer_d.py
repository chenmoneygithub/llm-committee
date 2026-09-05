#!/usr/bin/env python3
"""Shared logprob-in-debate instrument (prototype).

Two headline functions (Deliverable 2 of the prototype brief):

  stance_probe(context, question, K=7)      -> fixed-Likert K-point stance distribution
                                               (the SHORT teacher-forced target = clean path)
  dose_nll(context, opponent_text)           -> mean per-token NLL of a LONG span
                                               (the one long-span score = #48271-exposed)

Both are teacher-forced scores of a FIXED target on the PREFILL path (P of a fixed
string), NOT logprobs of sampled tokens. See docs/logprob_study/search/tooling.md §1-2
and the LEAD RULING (vLLM primary, batch-1/temp-0, score SHORT targets).

Design notes baked in:
  * batch-1 isolated requests, temperature 0  (kills vLLM #42019 batch-order jitter)
  * every HTTP call has an explicit timeout + retry  (memory: judge calls had NO timeout
    and a stalled gateway hangs forever — [[judge-calls-no-timeout-hang-risk]])
  * short single-token scale targets avoid the >256-tok #48271 RMSNorm regime by
    construction; dose_nll is the ONE long span, flagged as the noisy quantity.
  * chat models: the target is scored INSIDE the assistant turn via
    continue_final_message=True + add_generation_prompt=False (tooling §6.5).

Self-contained: only stdlib + numpy. Import as a module or run --selftest (offline).
"""
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

# --------------------------------------------------------------------------- #
#  Endpoint discovery
# --------------------------------------------------------------------------- #
# Endpoint config: set LLM_COMMITTEE_ENDPOINTS, or place an endpoints.json next to this file
# (see endpoints.example.json for the expected shape).
ENDPOINTS_JSON = os.environ.get(
    "LLM_COMMITTEE_ENDPOINTS",
    str(__import__("pathlib").Path(__file__).resolve().parent / "endpoints.json"),
)
# Ports we have observed live during this study (endpoints.json can be stale).
CANDIDATE_PORTS = [8103, 8104, 8003, 8002, 8001, 8000]

# Per-model chat-template kwargs. THINKING models (Qwen3, DeepSeek-V3/R1) emit a <think>
# block first, so a teacher-forced immediate answer gets ~0 mass on the answer tokens
# (the probe-validity blocker). enable_thinking=False makes the assistant turn start at
# the answer → qwen3 stance-digit mass jumps 0.00 → 0.9995 (verified live). Matched by
# substring of the served model id. NOTE: must be a TOP-LEVEL request field
# (chat_template_kwargs), NOT nested in extra_body (extra_body form returned 0 mass).
MODEL_CHAT_KWARGS = {
    "qwen3": {"enable_thinking": False},
    "qwen": {"enable_thinking": False},
    "deepseek": {"enable_thinking": False},  # DS-V3/R1 also reason; verify mass when it's live
}


def _chat_template_kwargs_for(model):
    if not model:
        return None
    ml = model.lower()
    for key, kw in MODEL_CHAT_KWARGS.items():
        if key in ml:
            return kw
    return None


# --------------------------------------------------------------------------- #
#  Canonical per-model READ CONFIG for the cross-family money-run
# --------------------------------------------------------------------------- #
# Reconciles TWO T sources (do NOT blindly apply MC gold-T to the stance probe —
# Cycle-20 finding): search-calibration's fitter output gold_T.json (MC-fit T +
# reasoning/degenerate flags) AND the stance-geometry N2-validation
# gold_T_stance_validate.json (the T at which the N2 irrelevant-arg control PASSES
# on the STANCE probe). RULE for the scaled run:
#   - stance operating-T = smallest T where N2 passes (|ΔE|<0.1); if the MC gold-T
#     disagrees, the STANCE-validated T wins (it's the geometry we actually read).
#   - if N2 never passes (deepseek: real/N2 ratio stuck ~2.0 at every T) OR the MC
#     T is degenerate (qwen: 6.26 bracket-collapse), MAGNITUDE DVs are NOT trusted →
#     content_gated=False → lead with the T-FREE ORDERING / rank+sign DVs only.
# The baked table is the reconciled roster verdict (both JSONs, 2026-07-21); the
# loader prefers the on-disk JSONs when present so a re-fit propagates.
_STANCE_READ_FALLBACK = {
    # model            stance_T  content_gated  magnitude_trusted  note
    # (task #40+#42 CORRECTED verdict; on-disk gold_T_stance_validate.json supersedes this when present.
    #  content_gated = sign-aware signed-margin AUROC(real>irr) CI-lower>0.5; magnitude_trusted = GRADED
    #  within-model E (strong ΔE>weak ΔE dynamic range), NOT the saturated categorical content-gate.)
    "llama70b":     {"stance_T": 2.5,  "content_gated": True,  "magnitude_trusted": False,
                     "note": "CONTENT-GATED on margin (AUROC 0.71) but SATURATED (p_argmax~1, strong-weak flat -0.04) -> NOT magnitude-trusted; report RANK/MARGIN not graded E-nats (task #42 fix)"},
    "qwen3_235b":   {"stance_T": 8.0,  "content_gated": True,  "magnitude_trusted": True,
                     "note": "CONTENT-GATED strongest (signed AUROC 0.94; irrelevant args HARDEN=resistance) AND GRADED within-model (strong-weak +0.171 CI[0.08,0.26]); degenerate MC-T blocks CROSS-family absolute scale only, not within-model grading (task #40+#42 fix)"},
    "deepseek_v3":  {"stance_T": 2.0,  "content_gated": False, "magnitude_trusted": False,
                     "note": "content-gating + graded both INCONCLUSIVE at pilot n=6 (AUROC 0.71 CI[0.44,0.94]; strong-weak +0.42 CI[-0.69,1.45]); irrelevant arm genuinely CO-MOVES (real leak, not sign artifact) -> lead with DIRECTION (heat-free antisym, where deepseek IS robust); settles ~20 items/model (task #38)"},
}


def stance_read_config(model, proto_dir=None):
    """Canonical per-model stance read-config for the cross-family money-run.

    Returns dict: stance_T (operating T for the stance E/DV reads), content_gated
    (does N2 pass -> is a MAGNITUDE persuasion claim admissible), magnitude_trusted
    (is the fitted T non-degenerate -> are nat-magnitudes comparable), chat_kwargs
    (enable_thinking flag), note. Prefers on-disk gold_T.json +
    gold_T_stance_validate.json when present (so a re-fit propagates); else the
    baked reconciled roster verdict. Match by substring (served id may vary)."""
    proto_dir = proto_dir or os.path.dirname(os.path.abspath(__file__))
    ml = (model or "").lower()

    def _match(table):
        for k, v in table.items():
            if k.lower() in ml or ml in k.lower():
                return v
        return None

    cfg = dict(_match(_STANCE_READ_FALLBACK) or {})
    # Consume search-calibration's AUTHORITATIVE stance-validation fields (do NOT re-derive
    # T semantics — that's the fitter owner's lane). The owner's schema encodes the exact
    # Cycle-20 rulings directly:
    #   magnitude_admissible   -> content_gated + magnitude_trusted (N2 real/irrelevant
    #                             ratio clears the threshold; llama True, qwen/ds False)
    #   stance_T_desat_anchored-> the de-saturation-anchored operating T for MAGNITUDE reads
    #                             (llama 2.5; None when magnitude not admissible -> rank/sign only)
    # This supersedes the baked fallback whenever the JSON is present, so an owner re-fit
    # propagates. (The prior stance_T_n2pass field was renamed by the owner; keying on it
    # silently mis-read — verify-fields-still-exist lesson.)
    try:
        sv = _match(json.load(open(os.path.join(proto_dir, "results", "gold_T_stance_validate.json"))))
    except Exception:
        sv = None
    try:
        g = _match(json.load(open(os.path.join(proto_dir, "gold_T.json"))))
    except Exception:
        g = None
    if sv is not None:
        # task #40 fix (search-calibration 2026-07-21): the owner's schema now SEPARATES the two
        # constructs the old single `magnitude_admissible` conflated (the old sign-blind ratio
        # wrongly failed qwen, whose irrelevant-arm HARDENING is content-gated resistance):
        #   content_gated       = sign-aware AUROC(real>irr) CI-lower > 0.5  (DISCRIMINATION)
        #   magnitude_admissible = content_gated AND non-degenerate gold-T   (graded-nat SCALE)
        # Prefer the new fields; fall back to the legacy flag only if an old JSON is on disk.
        if "content_gated" in sv:
            cfg["content_gated"] = bool(sv.get("content_gated", False))
            cfg["magnitude_trusted"] = bool(sv.get("magnitude_admissible", False))
            cfg["content_gated_inconclusive"] = bool(sv.get("content_gated_inconclusive", False))
            cfg["auroc_signed_margin"] = sv.get("auroc_signed_margin")
            cfg["auroc_signed_margin_ci"] = sv.get("auroc_signed_margin_ci")
        else:  # legacy schema (pre-#40)
            mag = bool(sv.get("magnitude_admissible", False))
            cfg["content_gated"] = mag
            cfg["magnitude_trusted"] = mag
        desat = sv.get("stance_T_desat_anchored")
        if desat is not None:
            cfg["stance_T"] = float(desat)        # magnitude operating-T
        elif sv.get("mc_gold_T"):
            # magnitude NOT admissible -> reads are rank/sign only; keep a sensible de-sat T
            # for the (untrusted) magnitude companion. Prefer the baked value; else MC gold-T.
            cfg.setdefault("stance_T", float(sv["mc_gold_T"]))
        cfg["ratio_at_gold_T_diagnostic"] = sv.get("ratio_at_gold_T_DIAGNOSTIC_ONLY", sv.get("ratio_at_gold_T"))
    if g is not None:
        # task #42 fix (search-calibration 2026-07-21): a degenerate MC-T blocks the CROSS-FAMILY
        # absolute nat-scale, but NOT the WITHIN-model graded magnitude (strong>weak is a within-model
        # ordinal at a fixed read-T, independent of the MC calibration). magnitude_trusted here is the
        # WITHIN-model graded-E flag (what the runner reports per-receiver), so a degenerate MC-T must
        # NOT force it False — that was the old conflation that flipped qwen backwards. The owner's
        # magnitude_admissible field already encodes the correct within-model graded test (dynamic
        # range), so we DEFER to it. Cross-family absolute comparison is separately gated by
        # gold_T.json.magnitude_axes.crossfamily_belief_magnitude_ok (False for all models).
        ax = g.get("magnitude_axes", {})
        if ax:
            cfg["crossfamily_belief_magnitude_ok"] = bool(ax.get("crossfamily_belief_magnitude_ok", False))
            cfg["dynamic_range"] = bool(ax.get("dynamic_range", False))
        # QUALITY-DOSE routing (search-calibration 2026-07-21): the strong-vs-weak leg
        # must read on the right substrate per model — MARGIN for a saturated receiver
        # (llama's E-space strong-vs-weak is a spurious ~0; the grading lives in the
        # T-free margin), E-RANK for graded receivers. Lifted straight from gold_T.json
        # so a single-cell misroute can't happen (the owner's field, not re-derived).
        if g.get("quality_dose_substrate") is not None:
            cfg["quality_dose_substrate"] = g["quality_dose_substrate"]
        if g.get("saturated") is not None:
            cfg["saturated"] = bool(g["saturated"])
        if g.get("is_reasoning") is not None:
            cfg["is_reasoning"] = bool(g["is_reasoning"])
    cfg.setdefault("stance_T", 8.0)
    cfg.setdefault("content_gated", False)
    cfg.setdefault("magnitude_trusted", False)
    cfg.setdefault("quality_dose_substrate", "margin")  # conservative default (T-free)
    cfg["chat_kwargs"] = _chat_template_kwargs_for(model)
    # READ-MODE audit stamp (team-lead / search-uncertainty): make the probe-mode-match
    # auditable, not asserted. chat_kwargs {enable_thinking:False} => thinking_off; None =>
    # the model has no thinking template (thinking_off by construction). The money-run
    # emit stamps this per row so a reviewer can verify the probe READ the same computation
    # the debate turns GENERATED (both default enable_thinking=False -> uniformly thinking_off).
    ek = (cfg["chat_kwargs"] or {}).get("enable_thinking", None)
    cfg["read_mode"] = "thinking_on" if ek is True else "thinking_off"
    cfg["model"] = model
    return cfg


def _get(url, timeout=8):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def discover_endpoint(prefer=None):
    """Return (base_url, served_model_name) for the first live chat endpoint.

    Probes endpoints.json first, then CANDIDATE_PORTS. `prefer` = substring to
    prioritise (e.g. 'bf16', 'llama', 'qwen'). Returns (None, None) if nothing live.
    """
    tried = []
    # 1) endpoints.json (may lag reality)
    urls = []
    try:
        with open(ENDPOINTS_JSON) as f:
            spec = json.load(f)
        host = spec.get("host", "127.0.0.1")
        for name, m in spec.get("models", {}).items():
            urls.append((f"http://{host}:{m['port']}", name))
    except Exception:
        pass
    # 2) observed ports
    for p in CANDIDATE_PORTS:
        urls.append((f"http://127.0.0.1:{p}", None))

    live = []
    for base, _hint in urls:
        if base in tried:
            continue
        tried.append(base)
        try:
            models = _get(base + "/v1/models", timeout=4)
            ids = [m["id"] for m in models.get("data", [])]
            for mid in ids:
                live.append((base, mid))
        except Exception:
            continue

    if not live:
        return None, None
    if prefer:
        for base, mid in live:
            if prefer.lower() in mid.lower() or prefer.lower() in base:
                return base, mid
    return live[0]


# --------------------------------------------------------------------------- #
#  Low-level HTTP (timeout + retry — never hang forever)
# --------------------------------------------------------------------------- #
def _post(base, path, payload, timeout=120, retries=3, backoff=2.0):
    url = base + path
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
        except Exception as e:  # noqa: BLE001 - surface HTTP 4xx/5xx bodies
            last = e
            # Try to read the error body for diagnosis (e.g. json_object 400s)
            if hasattr(e, "read"):
                try:
                    last = RuntimeError(f"{e}: {e.read().decode()[:400]}")
                except Exception:
                    pass
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} tries: {last}")


# --------------------------------------------------------------------------- #
#  Server-side tokenization / chat-template rendering (no local transformers —
#  AutoTokenizer.from_pretrained hangs on this box; the server renders for us)
# --------------------------------------------------------------------------- #
def tokenize_text(base, model, text, timeout=60):
    """Token-ids for a raw string (no special tokens / no chat template)."""
    r = _post(base, "/tokenize",
              {"model": model, "prompt": text, "add_special_tokens": False},
              timeout=timeout)
    return r["tokens"]


def render_chat_ids(base, model, messages, add_generation_prompt=True,
                    continue_final_message=False, timeout=60):
    """Token-ids for chat `messages` with the model's chat template applied.

    Uses vLLM's /tokenize with `messages` (renders the template server-side).
    """
    payload = {"model": model, "messages": messages,
               "add_generation_prompt": add_generation_prompt}
    if continue_final_message:
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
    ctk = _chat_template_kwargs_for(model)
    if ctk:
        payload["chat_template_kwargs"] = ctk   # keep the rendered context consistent (no <think>)
    r = _post(base, "/tokenize", payload, timeout=timeout)
    return r["tokens"]


# --------------------------------------------------------------------------- #
#  Sentinel guard (vLLM fp8 clamp -9999, NaN/inf) — poison a delta silently
# --------------------------------------------------------------------------- #
def _bad_lp(x):
    if x is None or not isinstance(x, (int, float)):
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    return x <= -9990.0


# --------------------------------------------------------------------------- #
#  Scale-token matching — LETTERS "A".."G" are the target (single-token, verified all 3 roster
#  tokenizers). _norm_token still accepts stray digits/punctuation defensively, but the stance scale
#  is scored on letters; digit surface forms are multi-token and must not be used as the target.
# --------------------------------------------------------------------------- #
def _norm_token(tok):
    """Normalise a returned token to a bare scale symbol, or None.

    Matches leading-space / trailing punctuation variants: ' 3', '3', '3.', '(A)'.
    """
    if tok is None:
        return None
    s = tok.strip().strip("().:").strip()
    if re.fullmatch(r"[0-9]", s):
        return s
    if re.fullmatch(r"[A-Ga-g]", s):
        return s.upper()
    return None


def _logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


# --------------------------------------------------------------------------- #
#  Next-token distribution over a fixed candidate set (the SHORT clean path)
# --------------------------------------------------------------------------- #
def next_token_scores(base, model, messages=None, prompt=None, candidates=None,
                      top_logprobs=20, chat=True, timeout=120):
    """Teacher-forced logprob of each candidate symbol as the NEXT token.

    Primary: ONE call reading top_logprobs at the answer position (raw logprobs =
    teacher-forced-equivalent for a single position). Fallback: for any candidate
    absent from the returned top-K, teacher-force it explicitly (prefill) so every
    candidate gets a real logprob regardless of rank.

    `candidates` = list of bare symbols e.g. ['1',..'7'] or ['A',..'E'].
    Returns dict symbol -> logprob (marginalised via logsumexp over surface forms).
    """
    assert candidates
    scores = {}  # symbol -> list of surface-form logprobs (to logsumexp)

    if chat:
        payload = {
            "model": model, "messages": messages, "max_tokens": 1,
            "temperature": 0.0, "logprobs": True, "top_logprobs": top_logprobs,
        }
        ctk = _chat_template_kwargs_for(model)
        if ctk:
            payload["chat_template_kwargs"] = ctk   # e.g. enable_thinking=False for qwen3
        resp = _post(base, "/v1/chat/completions", payload, timeout=timeout)
        content = (resp["choices"][0].get("logprobs") or {}).get("content") or []
        tops = (content[0].get("top_logprobs") if content else []) or []
    else:
        payload = {
            "model": model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0.0, "logprobs": top_logprobs, "echo": False,
        }
        resp = _post(base, "/v1/completions", payload, timeout=timeout)
        lp = resp["choices"][0].get("logprobs") or {}
        tl = (lp.get("top_logprobs") or [{}])
        d0 = tl[0] if tl else {}
        tops = [{"token": k, "logprob": v} for k, v in d0.items()]

    for e in tops:
        sym = _norm_token(e.get("token"))
        if sym in candidates and not _bad_lp(e.get("logprob")):
            scores.setdefault(sym, []).append(float(e["logprob"]))

    present = {s: _logsumexp(v) for s, v in scores.items()}

    # Fallback: teacher-force any missing candidate as a prefilled next token.
    missing = [c for c in candidates if c not in present]
    for c in missing:
        try:
            present[c] = _teacher_force_next(base, model, messages, prompt, c,
                                             chat=chat, timeout=timeout)
        except Exception:
            present[c] = float("-inf")  # unreachable candidate
    return present


def _realized_lp(pos_dict):
    """Return the realized token's logprob from one prompt_logprobs position dict.

    With prompt_logprobs=0 the dict holds exactly the realized (echoed) token,
    keyed by token_id-str -> {logprob, rank, decoded_token}. The realized token is
    NOT necessarily rank 1 (live: ' France' came back rank 6), so we must take the
    single finite logprob present, never filter on rank==1. If prompt_logprobs>0
    the dict has neighbours too; the realized token is the one whose rank we don't
    know, so take the finite entry (rank filtering is the bug we fixed).
    """
    if not isinstance(pos_dict, dict):
        return None
    cand = [info.get("logprob") for info in pos_dict.values()
            if isinstance(info, dict) and not _bad_lp(info.get("logprob"))]
    return float(cand[0]) if cand else None


def _teacher_force_next(base, model, messages, prompt, symbol, chat=True,
                        surface=None, timeout=120):
    """logP(symbol | context) via the /v1/completions echo path (token-id prompt).

    Chat `prompt_logprobs` is unavailable on this vLLM build (verified live), so we
    render the chat context to token-ids server-side (/tokenize) and score
    context_ids + [symbol_id] through completions echo. Reads the LAST position's
    realized logprob. `chat` selects whether `messages` or `prompt` is the context.
    """
    surf = surface if surface is not None else " " + symbol
    if chat:
        ctx_ids = render_chat_ids(base, model, messages, add_generation_prompt=True,
                                  timeout=timeout)
    else:
        ctx_ids = tokenize_text(base, model, prompt, timeout=timeout)
    tgt_ids = tokenize_text(base, model, surf, timeout=timeout)
    full_ids = ctx_ids + tgt_ids
    resp = _post(base, "/v1/completions",
                 {"model": model, "prompt": full_ids, "max_tokens": 1,
                  "temperature": 0.0, "echo": True, "prompt_logprobs": 0},
                 timeout=timeout)
    pls = resp["choices"][0].get("prompt_logprobs")
    if not pls:
        raise RuntimeError("no prompt_logprobs returned")
    lp = _realized_lp(pls[-1])
    if lp is None:
        raise RuntimeError("no finite logprob at target position")
    return lp


# --------------------------------------------------------------------------- #
#  Long-span teacher-forced scoring (the DOSE — #48271-exposed quantity)
# --------------------------------------------------------------------------- #
def score_span_logprobs(base, model, messages=None, prompt=None, target_text=None,
                        chat=True, timeout=180):
    """Per-token logprobs of `target_text` teacher-forced given the context.

    Returns list[float] (one per target token). The chat endpoint does NOT expose
    prompt_logprobs on this vLLM build (verified live), so we tokenize both the
    (chat-rendered) context and the target server-side, concatenate the ids, and
    score them through /v1/completions echo. The target span is the last
    len(target_ids) positions — an exact id boundary, no length-diff guessing and
    no realized-token rank filtering (that was the bug).
    """
    assert target_text is not None
    if chat:
        ctx_ids = render_chat_ids(base, model, messages, add_generation_prompt=True,
                                  timeout=timeout)
    else:
        ctx_ids = tokenize_text(base, model, prompt or "", timeout=timeout)
    tgt_ids = tokenize_text(base, model, target_text, timeout=timeout)
    if not tgt_ids:
        raise RuntimeError("empty target tokenization")
    full_ids = ctx_ids + tgt_ids

    resp = _post(base, "/v1/completions",
                 {"model": model, "prompt": full_ids, "max_tokens": 1,
                  "temperature": 0.0, "echo": True, "prompt_logprobs": 0},
                 timeout=timeout)
    pls = resp["choices"][0].get("prompt_logprobs") or []
    if len(pls) != len(full_ids):
        # vLLM returns one entry per prompt token (first is None). Align from the end.
        pass
    span = pls[len(pls) - len(tgt_ids):]  # last len(tgt_ids) positions = the target
    out = []
    for pos in span:
        lp = _realized_lp(pos)
        if lp is not None:
            out.append(lp)
    return out


# --------------------------------------------------------------------------- #
#  Stats helpers
# --------------------------------------------------------------------------- #
def softmax(logits, T=1.0):
    a = np.asarray(logits, dtype=float) / T
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def three_bin_mass(p):
    """Return (low, mid, high) fractions for a K-point ordinal scale.

    low = first ~2/7, high = last ~2/7, mid = the rest.  For K=7: {1,2}/{3,4,5}/{6,7}.
    """
    p = np.asarray(p, dtype=float)
    K = len(p)
    nlo = max(1, round(K * 2.0 / 7.0))
    nhi = nlo
    low = float(p[:nlo].sum())
    high = float(p[K - nhi:].sum())
    mid = float(max(0.0, 1.0 - low - high))
    return low, mid, high


# --------------------------------------------------------------------------- #
#  CANONICAL causal DV triple (computed ONE way — no per-pilot re-derivation)
# --------------------------------------------------------------------------- #
# The study's persuasion signal is a TRIPLE of orthogonal axes, each on the
# opponent-vs-length-matched-filler CAUSAL contrast (treat vs c0), never on
# treat-vs-round-0 (that carries pole-heating + context-growth drift). Pilots
# MUST call these instead of re-deriving — inconsistent re-derivation is exactly
# what caused the argmax_flip-vs-E_prev bug (search-debate) and the I1/I6
# direction disagreement (both re-measured margin-erosion as "direction").
#   - antisym_move_heatfree : DIRECTION headline (pole-heating-de-confounded, signed)
#   - antisym_move          : DIRECTION diagnostic (RAW mirror-odd; contaminated on
#                             pole-anchored starts — persist alongside, do NOT headline)
#   - margin_erosion        : CONVICTION (top1-top2 gap change; direction-agnostic)
#   - argmax_flip           : BEHAVIORAL (argmax(treat) != argmax(c0), SAME round)
# Direction ⟂ conviction (corr ~0.22 on i1) → both needed; never collapse
# direction into pole-gap/margin magnitude. See metric_spec §4a-bis / §4c.

def antisym_move(raw_treat, raw_ctrl, sign, T=8.0):
    """RAW mirror-odd TOWARD-opponent projection on the causal (treat-ctrl) contrast.

    ⚠️ NOT the direction headline — this is the pole-heating-CONTAMINATED diagnostic.
    Use antisym_move_heatfree (or E0-stratify to mid-scale) for the direction DV.

    `sign` = +1 if the opponent argues the HIGH pole (G), -1 if the LOW pole (A),
    so a positive return = the receiver's distribution moved TOWARD the opponent.
    The antisymmetric (mirror-odd) projection zeroes out any *static* symmetric
    flatten/sharpen — but it is peakedness-invariant ONLY for a CENTERED start.
    On a POLE-ANCHORED start, the *dynamic* entropy change between treat and ctrl
    (pole-heating: conviction erosion drains mass off the occupied pole toward
    center) projects onto this mirror-odd axis and FAKES a toward-opponent move.
    search-debate's heat-matched null (lead-verified 2026-07-21): raw antisym
    inherits ~55% pole-heating on pole-started i1 items (+0.29 of +0.39; heat-free
    residual there is a coin-flip 7/16), clean only on mid-scale-E0 headroom priors.
    ΔE ≡ this projection is an EXACT identity, so this is the mean-shift WITH the
    leak. Persist it as a diagnostic; headline antisym_move_heatfree."""
    pt = softmax(np.asarray(raw_treat, dtype=float), T=T)
    pc = softmax(np.asarray(raw_ctrl, dtype=float), T=T)
    dp = pt - pc
    anti = 0.5 * (dp - dp[::-1])                        # mirror-odd = location move + pole-heating leak
    K = len(dp)
    phi_bar = np.arange(1, K + 1) - (K + 1) / 2.0        # centered ordinal
    return float(sign) * float(np.dot(anti, phi_bar))


def antisym_move_heatfree(raw_treat, raw_ctrl, sign, T=8.0):
    """DIRECTION headline: toward-opponent move with the POLE-HEATING leak subtracted.

    = antisym_move(treat, ctrl) - antisym_move(heat_null, ctrl), where heat_null is
    the CONTROL logits heated (pure temperature scale, argmax preserved, ZERO location
    move) to treat's EXACT read-entropy. That null is the antisym a pole-anchored dist
    produces from symmetric heating ALONE; subtracting it leaves only the location
    shift NOT explainable by peakedness change. Reuses antisym_move as the single
    source of truth (heat_null in raw-logit space = raw_ctrl / h, then antisym_move's
    own softmax(., T) makes the net temperature T*h).

    Verified (search-debate + brainstorm-metric, lead-verified 2026-07-21):
      i1 pooled residual +0.18 (11/20 toward = coin-flip) vs raw +0.39; mid-scale-E0
      residual +0.89-0.97 (12/12 toward, clean); DeepSeek<-qwen +1.04 t=3.85 survives;
      raw-llama pole-started collapses to +0.05 (7/16). On a CENTERED start heat_null
      antisym ~ 0 so heatfree ~ raw (no leak to remove)."""
    treat = np.asarray(raw_treat, dtype=float)
    ctrl = np.asarray(raw_ctrl, dtype=float)
    if treat.size < 2:
        return 0.0
    Ht = entropy_bits(softmax(treat, T=T))                # target entropy = treat's read-entropy
    # bisect h so entropy(softmax(ctrl, T*h)) == Ht. entropy is monotone-increasing in
    # temperature, so a two-sided bracket finds the root whether treat is flatter (h>1,
    # the erosion/leak case) or sharper (h<1) than ctrl.
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        h = math.sqrt(lo * hi)                            # geometric bisection (temp is multiplicative)
        if entropy_bits(softmax(ctrl, T=T * h)) < Ht:
            lo = h                                        # too peaked -> need more heat
        else:
            hi = h
    h = math.sqrt(lo * hi)
    raw = antisym_move(treat, ctrl, sign, T=T)
    null = antisym_move(ctrl / h, ctrl, sign, T=T)        # heat_null vs ctrl: pure pole-heating antisym
    return raw - null


def margin_causal(raw_treat, raw_ctrl):
    """CONVICTION axis: change in top1-top2 raw-logit gap, causal (treat-ctrl).

    T-FREE (margin@T = margin@1 / T — sign & rank exactly T-invariant; LSE cancels
    since raw_logits are log-probs). Negative = opponent eroded conviction (the
    distribution got less peaked). This is the ONLY conviction DV with dynamic range
    on a saturated model (llama). DIRECTION-AGNOSTIC — do NOT read a drop as
    'moved toward opponent' (search-uncertainty/brainstorm-metric; anti-corr -0.73
    with signed move on i6). Pair with antisym_move for direction."""
    def _gap(raw):
        s = np.sort(np.asarray(raw, dtype=float))[::-1]
        return float(s[0] - s[1]) if s.size >= 2 else float("nan")
    return _gap(raw_treat) - _gap(raw_ctrl)


def argmax_flip_causal(raw_treat, raw_ctrl):
    """BEHAVIORAL axis: did the opponent flip the argmax letter vs the length-matched
    filler, IN THE SAME ROUND? (search-debate's bug fix: NOT vs E_prev/round-0 — that
    counts pole-heating drift and inflated the I1 round-1 flip rate ~10x, 50%→5%.)
    Returns 1 if argmax(treat) != argmax(ctrl), else 0."""
    at = int(np.argmax(np.asarray(raw_treat, dtype=float)))
    ac = int(np.argmax(np.asarray(raw_ctrl, dtype=float)))
    return int(at != ac)


def argmax_shift_causal(raw_treat, raw_ctrl, sign):
    """DIRECTION axis, GENUINELY T-FREE + SIGNED — the companion argmax_flip_causal (unsigned
    1/0) lacks and antisym_heatfree (a prob-space MAGNITUDE, ÷read-T) can't be at a per-model
    gold-T. Returns sign*(argmax(treat) − argmax(ctrl)) in SCALE-POINTS (treat-vs-ctrl, so
    pole-heating drift cancels exactly as in the other causal DVs); +ve = argmax moved TOWARD
    the opponent's pole. argmax is invariant under softmax(·/T) (monotone), so this leg has ZERO
    read-T dependence — it needs NO common-T recompute and reads identically at any receiver's
    gold-T, unlike antisym_heatfree whose magnitude AND near-zero sign both flatten at qwen's
    gold-T=8 (#46: qwen strong heatfree +0.48@T1→+0.06@T8, C2 frac 4/6→3/6; this leg is a stable
    [0,1,2,0,1,0]=3/6 toward, mean +0.667 at every T). A discrete companion to the level-DV
    direction leg, per [[construct-tagging-discipline]]'s T-free-DV rule — NOT a magnitude
    replacement (it's coarse: only fires when a full argmax step crosses)."""
    at = int(np.argmax(np.asarray(raw_treat, dtype=float)))
    ac = int(np.argmax(np.asarray(raw_ctrl, dtype=float)))
    return float(sign) * (at - ac)


def causal_dv_row(p0, p_treat, p_ctrl, sign, T=8.0):
    """SINGLE SOURCE OF TRUTH for the canonical causal DV row — computes EVERY DV one
    way from three stance_probe dicts (round-0 prior, treat, length-matched filler ctrl)
    on the treat-vs-ctrl contrast. The money-run + any re-analysis MUST call this rather
    than re-derive — inconsistent re-derivation is exactly what produced the argmax_flip-
    vs-E_prev bug and the I1/I6 direction disagreement.

    `sign` = +1 if the opponent argues the HIGH pole (G), -1 if the LOW pole (A), so all
    signed DVs return positive = moved TOWARD the opponent. `T` = the DIRECTION read-T
    (magnitude of margin/E is separately per-model; direction is T-robust in SIGN).

    Persists BOTH direction DVs (heatfree headline + raw diagnostic), CONVICTION (margin,
    T-free), BEHAVIORAL (argmax flip), the E-at-T triple + E0, and the raw_logits +
    probe-validity fields for all three reads so nothing is unrecoverable after a large
    spend (brainstorm-metric's GAP-2: E0 scalar alone can't be re-stratified at another T)."""
    lt = p_treat["raw_logits"]; lc = p_ctrl["raw_logits"]; l0 = p0["raw_logits"]
    def _E(raw):
        return float(np.dot(softmax(np.asarray(raw, float), T=T), np.arange(1, len(raw) + 1)))
    E0, Et, Ec = _E(l0), _E(lt), _E(lc)
    return {
        # ---- DIRECTION (headline = heatfree; raw = diagnostic, do NOT headline) ----
        "antisym_heatfree": antisym_move_heatfree(lt, lc, sign, T=T),
        "antisym_raw": antisym_move(lt, lc, sign, T=T),
        # ---- CONVICTION (T-free margin) + BEHAVIORAL ----
        "margin_causal": margin_causal(lt, lc),
        "argmax_flip_causal": argmax_flip_causal(lt, lc),
        # ---- DIRECTION, T-free + SIGNED (the emitted leg the money-run lacked, #46): read-T
        # invariant so it does NOT inherit qwen gold-T=8's antisym flattening; coarse (argmax-step
        # granular) so it CORROBORATES the level direction leg (read at common sharp T), not replace it.
        "argmax_shift_causal": argmax_shift_causal(lt, lc, sign),
        # ---- E-space companions (signed toward-opponent), all at direction-T ----
        "E0": E0, "E_treat": Et, "E_ctrl": Ec,
        "delta_causal": float(sign) * (Et - Ec),        # signed E move, causal
        "response_treat": float(sign) * (Et - E0),      # signed E move, vs round-0 (drift-confounded)
        "sign": float(sign), "T_direction": float(T),
        # ---- full raw vectors + probe-validity for offline re-stratification (GAP-2) ----
        "raw_logits_0": list(l0), "raw_logits_treat": list(lt), "raw_logits_ctrl": list(lc),
        "letter_mass_0": p0.get("letter_mass"), "letter_mass_treat": p_treat.get("letter_mass"),
        "letter_mass_ctrl": p_ctrl.get("letter_mass"),
        "degenerate_any": bool(p0.get("degenerate") or p_treat.get("degenerate") or p_ctrl.get("degenerate")),
        # p_argmax_treat is at the DIRECTION read-T (not T=1). Fine for the I26 DV-space admissibility
        # gate (the read AS TAKEN), but NOT the baseline-commitment gate — for "is this item committed
        # at baseline" recompute p_argmax@T=1 from raw_logits_0 (brainstorm-broaden: qwen T=8 deflates it).
        "p_argmax_treat": p_treat.get("p_argmax"),
        "n_interior_cutpoints_treat": p_treat.get("n_interior_cutpoints"),
        # on_scale_mass BEFORE renormalization, per read (brainstorm-design req 5, task #43): the
        # §2.3-c(v) three-branch DV dispatcher fires from emitted data (letter_mass == this alias).
        "on_scale_mass_0": p0.get("on_scale_mass", p0.get("letter_mass")),
        "on_scale_mass_treat": p_treat.get("on_scale_mass", p_treat.get("letter_mass")),
        "on_scale_mass_ctrl": p_ctrl.get("on_scale_mass", p_ctrl.get("letter_mass")),
    }


# --------------------------------------------------------------------------- #
#  DELIVERABLE 2a — stance_probe
# --------------------------------------------------------------------------- #
# NOTE (brainstorm-metric 2026-07-20): target = LETTERS, not digits. Verified on all 3 roster
# tokenizers that " 1".." 7" are MULTI-TOKEN ([space,digit]) and SHARE their first sub-token
# (220 Llama/Qwen, 223 DeepSeek), so a single-position teacher-forced read can't score a digit scale
# cleanly (forces every candidate through the per-candidate fallback + risks mixed conditioning).
# " A".." G" ARE single-token on all 3 → one clean top_logprobs read. See metric_spec.md §2.
LIKERT_PROMPT = (
    "On a 7-point scale from A to G where A = strongly disagree and G = strongly agree, "
    "state the single letter that best reflects YOUR OWN current view on the "
    "statement. Reply with only the letter.\nMy rating (A-G):"
)


def stance_probe(base, model, context_messages, question, K=7, T=1.0, chat=True,
                 top_logprobs=20, timeout=120):
    """Fixed-Likert K-point stance distribution by teacher-forced scoring.

    context_messages : list of chat messages (the frozen debate context) OR None.
    question         : the opinion proposition (string) being rated.
    Returns dict: p (K floats), E (expected score in [1,K]), H (bits),
                  endpoint_mass {low,mid,high}, raw_logits, off (unused mass proxy).
    """
    symbols = [chr(ord("A") + i) for i in range(K)]   # A..G, single-token on all 3 roster tokenizers
    phi = np.arange(1, K + 1, dtype=float)              # ordinal rank 1..K unchanged (E stays in [1,K])

    user_probe = f"Statement: {question}\n\n{LIKERT_PROMPT}"
    if chat:
        messages = list(context_messages or []) + [{"role": "user", "content": user_probe}]
        scored_input = messages                          # WHITE-BOX §7.1: verbatim scored input
        raw = next_token_scores(base, model, messages=messages, candidates=symbols,
                                top_logprobs=top_logprobs, chat=True, timeout=timeout)
    else:
        prompt = (context_messages or "") + "\n" + user_probe + " "
        scored_input = prompt                            # WHITE-BOX §7.1: verbatim scored input
        raw = next_token_scores(base, model, prompt=prompt, candidates=symbols,
                                top_logprobs=top_logprobs, chat=False, timeout=timeout)

    logits = np.array([raw.get(s, float("-inf")) for s in symbols], dtype=float)
    finite = np.isfinite(logits)
    if finite.sum() == 0:
        raise RuntimeError("stance_probe: no finite scale-token logprobs")
    # ---- PROBE-VALIDITY GATE (degeneracy catch): total ABSOLUTE prob mass on the K letters ----
    # raw is full-vocab log-softmax, so exp(logprob) is absolute mass. If the model puts most of
    # its mass OFF the letters (e.g. qwen3 answers with prose → 0.09 mass on the digits), then E
    # is computed from a degenerate low-mass TAIL and is NOISE, not the model's stance. Callers
    # MUST check letter_mass > ~0.5 before trusting E (see [[uncertainty-dv-degeneracy-gate]]).
    letter_mass = float(sum(math.exp(x) for x in logits[finite]))
    # renormalise over the on-scale symbols (softmax of raw logprobs == teacher-forced dist)
    p = softmax(logits[np.isfinite(logits)] if finite.all() else np.where(finite, logits, -1e9), T=T)
    E = float(np.dot(p, phi))
    low, mid, high = three_bin_mass(p)
    # ---- I26 dispatcher selectors (search-calibration-2): logged not post-hoc-inferred ----
    # p_argmax = saturation gate (>~0.9 → defer to raw-logit margin, interior-beta undefined);
    # n_interior_cutpoints = count of ordinal cutpoints F_r(k)=cumsum(p)[:-1] in (0.05,0.95) =
    # "is interior-beta even defined here" (0 → undefined-not-zero, saturated CDF jumps 0→1).
    # Both are on the returned p (at this read's T); reflects the read AS TAKEN.
    cdf = np.cumsum(p)[:-1] if K > 1 else np.array([])
    p_argmax = float(p.max())
    n_interior_cutpoints = int(np.sum((cdf > 0.05) & (cdf < 0.95)))
    return {
        "p": p.tolist(),
        "E": E,
        "H": entropy_bits(p),
        "endpoint_mass": {"low": low, "mid": mid, "high": high},
        "raw_logits": logits.tolist(),
        "n_finite": int(finite.sum()),
        "letter_mass": letter_mass,                 # PROBE-VALIDITY: assert > ~0.5 before trusting E
        "on_scale_mass": letter_mass,               # alias (search-calibration-2 naming); == letter_mass
        "degenerate": bool(letter_mass < 0.5),      # True = probe failed to read this model's stance
        "p_argmax": p_argmax,                       # I26 saturation gate: >~0.9 → margin-defer branch
        "n_interior_cutpoints": n_interior_cutpoints,  # I26: 0 → interior-beta UNDEFINED (not zero)
        "K": K,
        # WHITE-BOX §7.1 (brainstorm-broaden): the EXACT input teacher-forced here. This is the ONE
        # money-run field NOT recoverable offline from raw_logits — the temp-0.7 debate turns that
        # produced each probed context can't be re-forwarded for I12/I18/I23 without the verbatim
        # scored input. Persist it before any temp-0.7 debate generation or the white-box arm dies.
        "scored_messages": scored_input,
    }


# --------------------------------------------------------------------------- #
#  DELIVERABLE 2b — dose_nll
# --------------------------------------------------------------------------- #
def dose_nll(base, model, context_messages, opponent_text, chat=True, timeout=180):
    """Length-normalised (mean per-token) NLL of opponent_text under the model.

    NLL = -mean(logP(token)). This is the ONE long-span teacher-forced score, so it
    is the quantity most exposed to the vLLM #48271 >256-tok RMSNorm bug — the
    noise-floor probe (Deliverable 3) must characterise it before it is trusted.
    Returns dict: nll (mean per-token NLL), n_tokens, sum_logprob.
    """
    if chat:
        lps = score_span_logprobs(base, model, messages=context_messages,
                                  target_text=opponent_text, chat=True, timeout=timeout)
    else:
        lps = score_span_logprobs(base, model, prompt=(context_messages or ""),
                                  target_text=opponent_text, chat=False, timeout=timeout)
    if not lps:
        raise RuntimeError("dose_nll: no target-span logprobs")
    arr = np.array(lps, dtype=float)
    return {
        "nll": float(-arr.mean()),
        "sum_logprob": float(arr.sum()),
        "n_tokens": int(arr.size),
    }


# --------------------------------------------------------------------------- #
#  Generation helper (opponent turns for the pilot)
# --------------------------------------------------------------------------- #
def generate(base, model, messages, max_tokens=220, temperature=0.7, timeout=180,
             allow_thinking=False):
    """Generate a turn. For thinking models we default enable_thinking=False so opponent
    turns are answer-only (keeps DOSE = NLL of the ARGUMENT, not a reasoning trace). Pass
    allow_thinking=True to let the model reason (e.g. if studying reasoning explicitly)."""
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature,
    }
    ctk = _chat_template_kwargs_for(model)
    if ctk and not allow_thinking:
        payload["chat_template_kwargs"] = ctk
    resp = _post(base, "/v1/chat/completions", payload, timeout=timeout)
    return resp["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
#  Offline self-test (no endpoint needed) — exercises the pure-python paths
# --------------------------------------------------------------------------- #
def _selftest():
    print("== instrument.py offline self-test ==")
    # scale-token normalisation
    assert _norm_token(" 3") == "3"
    assert _norm_token("A.") == "A"
    assert _norm_token("(B)") == "B"
    assert _norm_token("hello") is None
    print("  _norm_token OK")
    # softmax / entropy / 3-bin
    logits = [0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.5]
    p = softmax(logits)
    assert abs(p.sum() - 1.0) < 1e-9
    lo, mid, hi = three_bin_mass(p)
    assert abs(lo + mid + hi - 1.0) < 1e-9
    E = float(np.dot(p, np.arange(1, 8)))
    print(f"  softmax/E OK  E={E:.3f}  3bin=({lo:.3f},{mid:.3f},{hi:.3f})  H={entropy_bits(p):.3f}")
    # sentinel guard
    assert _bad_lp(-9999.0) and _bad_lp(None) and _bad_lp(float("nan"))
    assert not _bad_lp(-2.3)
    print("  _bad_lp OK")
    # logsumexp
    assert abs(_logsumexp([0.0, 0.0]) - math.log(2)) < 1e-9
    print("  _logsumexp OK")
    # ---- canonical DV triple ----
    K = 7
    base_logits = [2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]   # peaked toward LOW pole (A)
    # DIRECTION: opponent argues HIGH pole (sign=+1); treat shifts mass toward G
    treat = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0]
    assert antisym_move(treat, base_logits, sign=+1, T=8.0) > 0, "antisym: toward-high should be +"
    assert antisym_move(treat, base_logits, sign=-1, T=8.0) < 0, "antisym: sign flips with opponent side"
    # STATIC PEAKEDNESS-INVARIANCE: a symmetric sharpen/flatten about the CENTER
    # (no location move) => raw antisym ~ 0. This is the case raw antisym DOES handle.
    sym_ctrl = [0.0] * K
    sym_treat = [1.0, 0.5, 0.2, 0.0, 0.2, 0.5, 1.0]          # symmetric about center, pure peakedness
    assert abs(antisym_move(sym_treat, sym_ctrl, sign=+1, T=8.0)) < 1e-9, "antisym must zero STATIC symmetric change"
    # DYNAMIC POLE-HEATING LEAK (search-debate's catch, lead-verified) — the case raw
    # antisym FAILS: a POLE-anchored ctrl (argmax G, ~10-nat gap) with treat = PURELY
    # heated ctrl (ctrl/3 => same argmax G, higher entropy, ZERO location move). Pure
    # heating of a pole-anchored dist drains mass toward center => a spurious antisym
    # toward the opposite pole. raw MUST be large (leak); heatfree MUST be ~0 (fix,
    # since treat is exactly a heated ctrl the heat-null reconstructs it).
    pole_ctrl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]         # entrenched at G (high pole)
    pole_treat = [x / 3.0 for x in pole_ctrl]                # pure heat, argmax fixed, zero location
    raw_leak = antisym_move(pole_treat, pole_ctrl, sign=-1, T=8.0)
    hf_leak = antisym_move_heatfree(pole_treat, pole_ctrl, sign=-1, T=8.0)
    assert abs(raw_leak) > 0.3, f"pole-heating leak: raw antisym must be LARGE, got {raw_leak:.4f}"
    assert abs(hf_leak) < 1e-3, f"heatfree must remove pure pole-heating (~0), got {hf_leak:.4f}"
    # AND heatfree must NOT over-subtract a GENUINE location move on a centered start
    # (matched-peakedness shift => heat-null ~0 => heatfree ~ raw, both toward-high +).
    cen_ctrl = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]           # centered, peak at D
    cen_treat = [0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0]          # same shape shifted toward high (F)
    raw_loc = antisym_move(cen_treat, cen_ctrl, sign=+1, T=8.0)
    hf_loc = antisym_move_heatfree(cen_treat, cen_ctrl, sign=+1, T=8.0)
    assert raw_loc > 0 and hf_loc > 0, "centered location move: both raw & heatfree toward-high +"
    assert abs(hf_loc - raw_loc) < 0.05, f"heatfree must preserve genuine location move, |hf-raw|={abs(hf_loc-raw_loc):.4f}"
    # CONVICTION: margin_causal negative when treat is less peaked than ctrl
    peaked = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    flat = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert margin_causal(flat, peaked) < 0, "margin: less-peaked treat => negative erosion"
    # T-FREENESS of margin sign: gap scales by 1/T but sign is invariant (raw-logit gap, no softmax)
    m1 = margin_causal(flat, peaked)
    assert m1 == margin_causal(flat, peaked), "margin deterministic"
    # BEHAVIORAL: flip iff argmax differs, same-round (treat vs ctrl, NOT vs prev)
    assert argmax_flip_causal([0, 0, 0, 0, 0, 0, 5], peaked) == 1, "flip: argmax moved A->G"
    assert argmax_flip_causal(peaked, peaked) == 0, "flip: same argmax => 0"
    # DIRECTION T-free signed argmax-shift: argmax A(0)->G(6), sign +1 => +6 toward high pole;
    # sign -1 flips it; and it is literally T-INVARIANT (argmax ignores the /T scaling).
    assert argmax_shift_causal([0, 0, 0, 0, 0, 0, 5], peaked, sign=+1) == +6.0, "argmax-shift: A->G, sign+ => +6"
    assert argmax_shift_causal([0, 0, 0, 0, 0, 0, 5], peaked, sign=-1) == -6.0, "argmax-shift: sign flips orientation"
    assert argmax_shift_causal(peaked, peaked, sign=+1) == 0.0, "argmax-shift: no argmax move => 0"
    print("  DV triple (antisym/margin/flip) + T-free argmax-shift OK")
    # ---- causal_dv_row assembler: single-source-of-truth wiring matches the helpers ----
    def _pd(raw):  # minimal stance_probe-shaped dict
        return {"raw_logits": raw, "letter_mass": 0.99, "degenerate": False,
                "p_argmax": float(softmax(np.asarray(raw, float), T=8.0).max()),
                "n_interior_cutpoints": 3}
    row = causal_dv_row(_pd(base_logits), _pd(treat), _pd(base_logits), sign=+1, T=8.0)
    assert row["antisym_heatfree"] == antisym_move_heatfree(treat, base_logits, +1, T=8.0), "row wires heatfree"
    assert row["antisym_raw"] == antisym_move(treat, base_logits, +1, T=8.0), "row wires raw antisym"
    assert row["margin_causal"] == margin_causal(treat, base_logits), "row wires margin"
    assert "raw_logits_0" in row and "letter_mass_treat" in row and "T_direction" in row, "row persists recovery fields"
    assert row["antisym_heatfree"] > 0, "toward-high heatfree +"
    # on_scale_mass per read (task #43): dispatcher-from-data completeness; == letter_mass alias
    assert "on_scale_mass_treat" in row and row["on_scale_mass_treat"] == 0.99, "row persists on_scale_mass per read"
    assert "on_scale_mass_0" in row and "on_scale_mass_ctrl" in row, "row persists on_scale_mass for all 3 reads"
    print("  causal_dv_row assembler OK")
    # ---- I26 dispatcher selector fields shape ----
    p_sat = softmax([20.0, 0, 0, 0, 0, 0, 0])                # saturated
    cdf = np.cumsum(p_sat)[:-1]
    assert float(p_sat.max()) > 0.9 and int(np.sum((cdf > 0.05) & (cdf < 0.95))) == 0, \
        "saturated read: p_argmax>0.9, n_interior_cutpoints==0"
    print("  I26 selectors OK")
    # ---- canonical stance read-config (reconciled MC gold-T vs stance-N2) ----
    cl = stance_read_config("llama70b")
    cq = stance_read_config("qwen3_235b")
    cd = stance_read_config("deepseek_v3")
    # Consumes search-calibration's authoritative verdict (task #40+#42 fix: sign-aware AUROC + graded
    # dynamic-range, NOT the retired sign-blind saturation-frozen ratio).
    #   CONTENT-GATED (signed margin AUROC CI-lower>0.5): llama (0.71) + qwen (0.94 strongest) YES;
    #     deepseek INCONCLUSIVE at pilot n (CI crosses 0.5).
    #   MAGNITUDE-TRUSTED = GRADED within-model E (strong ΔE>weak ΔE, dynamic range): qwen ONLY
    #     (+0.171 CI[0.08,0.26]); llama FAILS (saturated, strong-weak flat -0.04) even though it
    #     content-gates; deepseek graded-inconclusive at n=6. This is the CORRECTED (previously
    #     BACKWARDS) mapping: the runner reports graded E-nats gated by this, so it must follow
    #     dynamic-range, NOT the saturated-llama categorical content-gate.
    assert cl["content_gated"] and not cl["magnitude_trusted"], \
        "llama: content-gated (categorical) but NOT magnitude-trusted (SATURATED, no graded dynamic range)"
    assert cq["content_gated"] and cq["magnitude_trusted"], \
        "qwen: content-gated (signed AUROC 0.94) AND magnitude-trusted (graded within-model E, cleanest)"
    assert not cd["magnitude_trusted"], "deepseek: NOT magnitude-trusted (graded-inconclusive at pilot n)"
    # cross-family ABSOLUTE magnitude is separately blocked for ALL (no model has 2 admissible peers)
    assert not cq.get("crossfamily_belief_magnitude_ok") and not cl.get("crossfamily_belief_magnitude_ok"), \
        "cross-family absolute magnitude blocked for all (degenerate/saturated/single-admissible)"
    assert cq["chat_kwargs"] == {"enable_thinking": False}, "qwen needs enable_thinking=False"
    assert cl["chat_kwargs"] is None, "llama needs no chat_kwargs"
    # stance_T = owner's de-saturation-anchored T for the admissible model (llama 2.5)
    assert abs(cl["stance_T"] - 2.5) < 0.1, f"llama stance_T should be desat-anchored 2.5, got {cl['stance_T']}"
    # READ-MODE audit stamp (team-lead / search-uncertainty): uniformly thinking_off on the roster
    assert cl["read_mode"] == "thinking_off" and cq["read_mode"] == "thinking_off" and cd["read_mode"] == "thinking_off", \
        "read_mode must stamp thinking_off (all roster probes+turns generate enable_thinking=False)"
    # QUALITY-DOSE substrate routing (search-calibration): llama saturated -> margin; graded -> e_rank
    assert cl["quality_dose_substrate"] == "margin", f"llama quality-dose = margin (saturated), got {cl['quality_dose_substrate']}"
    assert cl["saturated"] is True and cq["saturated"] is False and cd["saturated"] is False, \
        "saturated flag: llama True, qwen/deepseek False"
    def _mtag(c):
        return "MAG" if c.get("magnitude_trusted") else ("gate" if c.get("content_gated") else "rank")
    print(f"  stance_read_config OK  llama(T={cl['stance_T']:.2f},{_mtag(cl)},{cl['quality_dose_substrate']}) "
          f"qwen(T={cq['stance_T']:.2f},{_mtag(cq)},{cq['quality_dose_substrate']}) "
          f"ds(T={cd['stance_T']:.2f},{_mtag(cd)},{cd['quality_dose_substrate']}) read_mode={cl['read_mode']}")
    print("ALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        base, model = discover_endpoint(prefer="bf16")
        print(f"discover_endpoint -> base={base} model={model}")
