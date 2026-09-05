"""
Condition-blind, cross-family judge instrument for the LLM-committee debate study.

WHY THIS FILE EXISTS
--------------------
The prior study's headline result — an adversarial ("hostile") debate-stance prompt drops
committee full-agreement from 77% -> 33% — is **unfalsifiable as stated** because the
``agreed_level`` it is computed from is a SELF-REPORTED label emitted by the very same model
that received the stance instruction. That conflates three distinct layers:

- **Layer A** = the label the model emits (``agreed_level``) — contaminated by the instruction.
- **Layer B** = does the reply TEXT actually push back (challenge, demand evidence, refuse to
  endorse) — the real debate dynamics.
- **Layer C** = does the member's actual POSITION change (would it pick a different survey
  option) — the real outcome.

This module provides the **neutral, condition-blind** instrument that measures B and C from
text, replacing the self-report:

- ``judge_pushback(parent_argument, reply_text)``  -> Layer B. Rates, blind to stance/model,
  how much a reply pushes back on its parent argument (a 4-point scale mirroring ``AgreedLevel``)
  AND classifies the dialogue act (``move_type``). The two axes together let us measure the
  A-vs-B gap that the whole study hinges on: e.g. a reply that self-reports "partially_disagreed"
  but whose text is a ``restatement`` with ``agreed_level=fully_agreed`` is *performative*
  disagreement — a label with no textual pushback.
- ``map_to_options(opinion_text, options)``        -> Layer C. Maps a free-text position onto a
  SOFT probability distribution over the survey ``options`` (+ an explicit refusal bucket), the
  objective substrate for the pluralism / opinion-diversity analysis.

Both are dspy modules built on ``llm_committee.databricks_lm.make_lm``.

DESIGN RULES (baked in; citations to docs/prior_art/judge-reliability.md by rule number)
-----------------------------------------------------------------------------------------
1. **Cross-family judge; report per-family agreement.** Self-preference bias is 10-25 win-points
   and causally tied to self-recognition (judge-reliability.md "Cross-family judging guidance").
   The debaters span opus/sonnet (Anthropic), gpt-5.5 (OpenAI), gemini (Google), so NO single
   judge is neutral to every composition. We therefore (a) expose ``judge_model``, (b) provide
   ``clean_judges_for(debaters)`` to pick a non-debater family per composition, and (c) provide a
   3-disjoint-family JURY (``*_jury``) with per-judge breakdowns for the diverse committees and as
   the own-family-favoritism gate (rules: "Cross-family judging guidance", mitigation #2).
2. **CoT before the verdict.** A ``reasoning`` OutputField is declared FIRST so it is generated
   before the labels (rule #5, #6). We use it for robustness/auditability, not as a correctness
   guarantee (JUDGE-BENCH: CoT does not reliably raise human agreement).
3. **Structured output, small integer/enum scale.** Reasoning -> then a small integer stance
   score (1-4) and an enum move_type; parsed defensively like ``grader.py`` (rules #4, #6).
4. **Blind to condition + debater identity.** The judge is passed ONLY the argument texts — never
   the stance (friendly/hostile), model name, or member id — and is instructed to ignore any
   meta-commentary about a debater's assigned role/stance and judge only substantive content
   (rule #8).
5. **Explicit refusal option (Layer C).** A ``"No position / refusal"`` bucket is always appended
   so a hedge is not force-mapped onto a substantive option, which would inflate apparent
   agreement (pluralism-globalopinions.md §4c-3). Kept SEPARATE from any dataset ``DK/Refused``
   option (that is a substantive survey choice; ours means "commits to no listed option").
6. **Randomize option order (Layer C).** LLM classifiers are order-sensitive (Wang et al. 2023,
   "LLMs are not Fair Evaluators"); we run ``n_orders`` shuffles and average per-option
   probabilities before argmax (Balanced Position Calibration), then re-align to the original
   option order (rule: mitigation #1 / pluralism §4c-7).

The stance-rating task (Layer B) is the HIGH-RISK job — LLM judges are weakest on
attitudinal/graded tasks (JUDGE-BENCH best model ~kappa 0.28 categorical). It therefore uses an
anchored NLI-style rubric and MUST be calibrated against a human-labeled subset (Cohen's
kappa) before downstream analysis trusts the labels. See the ``__main__`` self-test /
cross-family gate here, and the separate calibration harness.

Self-test / cross-family gate::

    python judge.py                 # hand-constructed cases + cross-family (opus vs gpt-5.5) agreement
    python judge.py --quick         # single-family only (cheaper; skips the cross-family gate)
    python judge.py --judge gemini  # override the single-family judge used for the pointwise cases
"""

from __future__ import annotations

import argparse
import functools
import random
import re
from collections import Counter
from typing import Any

import dspy

from llm_committee.databricks_lm import make_lm

# ── Judge-model families (for cross-family neutrality checks) ─────────────────────────────────
# The debaters in this study (configs.py) are opus-4.8 & sonnet-4.6 (Anthropic), gpt-5.5 (OpenAI),
# and gemini (Google). Map each usable judge alias to its family so callers can pick a judge NOT
# among a committee's families, and so the jury draws from disjoint families.
JUDGE_MODEL_FAMILY: dict[str, str] = {
    "opus-4.8": "anthropic",
    "opus": "anthropic",
    "sonnet-4.6": "anthropic",
    "sonnet": "anthropic",
    "gpt-5.5": "openai",
    "gpt-5.5-pro": "openai",
    "gemini": "google",
    "gemini-flash": "google",
    "qwen": "alibaba",
    "qwen3.5": "alibaba",
}

# Default single-family judge. gpt-5.5 (OpenAI) is a strong, comparatively position-robust judge
# (MT-Bench) and is cross-family to the opus/sonnet/gemini-only compositions. For opus+gpt
# compositions the CALLER should override via clean_judges_for(...). No globally-neutral judge
# exists (every model is a debater somewhere), so neutrality is a per-composition choice.
DEFAULT_JUDGE_MODEL = "gpt-5.5"

# Two strong frontier judges from disjoint families — the own-family-favoritism gate used by the
# validation harness. Agreement between them bounds how much either is inflating its own family.
CROSS_FAMILY_JUDGES = ["opus-4.8", "gpt-5.5"]

# Production jury: one judge per disjoint family. For committees that span all three families
# (e.g. diverse4) no single judge is clean, so aggregate the jury and report per-judge + the
# own-family check (judge-reliability.md "Cross-family judging guidance", mitigation #2).
JURY_JUDGES = ["opus-4.8", "gpt-5.5", "gemini"]

# ── Layer-B label spaces ──────────────────────────────────────────────────────────────────────
# AgreedLevel values, mirrored from llm_committee.committee_sync.AgreedLevel so the blind read is
# on the SAME 4-point scale as the self-report (Layer A) it replaces — enabling a direct A-vs-B gap.
AGREED_LEVELS = ("fully_agreed", "partially_agreed", "partially_disagreed", "fully_disagreed")

# Severity ordering shared with disagreement_evolution.py (0 = full agreement ... 3 = full
# disagreement). Confirmed with research-methods so blind labels plug into the same SEVERITY walk.
SEVERITY: dict[str, int] = {
    "fully_agreed": 0,
    "partially_agreed": 1,
    "partially_disagreed": 2,
    "fully_disagreed": 3,
}

# The judge emits an integer 1-4 (small integer scale, rule #4); anchored NLI-style so 1 = strongly
# rebuts the parent's central claim ... 4 = strongly endorses/builds on it (judge-reliability.md
# "Job 1" recommended anchoring). Map that integer to the AgreedLevel enum value.
STANCE_INT_TO_AGREED_LEVEL: dict[int, str] = {
    1: "fully_disagreed",
    2: "partially_disagreed",
    3: "partially_agreed",
    4: "fully_agreed",
}

# Dialogue-act taxonomy (debate-dynamics report's recommended model-blind text classifier). This
# is the axis that captures whether the TEXT introduces new counter-content vs merely restates vs
# concedes — orthogonal to the stance label, which is what lets us measure "disagreement LABEL"
# vs "new counter-content in the text" (the A-vs-B gap reflection requires).
MOVE_TYPES = ("new_argument", "refinement", "concession", "restatement", "challenge")

# The explicit refusal / no-position bucket appended to every Layer-C option list. Kept distinct
# from any dataset "DK/Refused" option (see module docstring rule #5).
REFUSAL_OPTION = "No position / refusal"


@functools.lru_cache(maxsize=8)
def _get_judge_lm(model: str, max_tokens: int = 1500, temperature: float = 1.0,
                  timeout: float | None = None) -> dspy.LM:
    """Build (and process-cache) a judge LM.

    Judges run at **temperature=1.0**. Ideally a judge would run at 0.0 for reproducibility, but
    on this workspace neither frontier family permits it: the OpenAI GPT-5 endpoint (gpt-5.5)
    rejects temperature != 1.0 outright (litellm UnsupportedParamsError), and — verified — the
    Databricks Claude endpoints (opus/sonnet) fail with a spurious
    ``INVALID_PARAMETER_VALUE: Response format type json_object is not supported`` whenever
    temperature < 1.0 (the whole surrounding codebase, e.g. grader.py, therefore runs Claude at the
    make_lm default of 1.0). So 1.0 is the only value that works across all four judge families.
    Verdict variance is instead controlled the way judge-reliability.md prescribes: an anchored
    rubric, structured output, and a small integer/enum scale (rules #3, #4, #6) — not by pinning
    the sampler. Callers needing extra stability can average over repeated calls or use the jury.

    Structured-content endpoints (gemini) auto-bump max_tokens to >=4096 inside make_lm for their
    hidden reasoning block, so a small nominal max_tokens is safe here.

    Args:
        model: A make_lm endpoint name or short alias (e.g. "opus-4.8", "gpt-5.5", "gemini").
        max_tokens: Output budget (CoT reasoning + short verdict).
        temperature: Sampling temperature; leave at 1.0 (see above — other values break the
            Anthropic/OpenAI endpoints on this workspace).
        timeout: Per-request HTTP timeout (seconds). None → make_lm's own default. Pass an explicit
            value from long fan-out jobs so a stalled gemini SDK call can't hang the whole thread
            pool forever (the judge_pairwise deadlock, 2026-07-10 — see the no-timeout-hang memory).

    Returns:
        A configured dspy.LM.
    """
    kw = {} if timeout is None else {"timeout": timeout}
    return make_lm(model, max_tokens=max_tokens, temperature=temperature, **kw)


def family_of(model: str) -> str:
    """Return the model family for a judge/debater alias ("anthropic"/"openai"/"google"/...)."""
    return JUDGE_MODEL_FAMILY.get(model, JUDGE_MODEL_FAMILY.get(model.split("/")[-1], "unknown"))


def clean_judges_for(debater_models: list[str], candidates: list[str] | None = None) -> list[str]:
    """Return the candidate judges whose family is NOT among the debaters' families.

    Used to pick a condition-blind, cross-family judge per committee composition. For committees
    that span every candidate family (e.g. diverse4 = Anthropic+OpenAI+Google) this returns ``[]``
    — the signal that no single judge is neutral and the caller should use the jury + report the
    own-family caveat.

    Args:
        debater_models: The committee's member (and chairman) model aliases.
        candidates: Judge pool to filter (default: JURY_JUDGES, one per disjoint family).

    Returns:
        Judge aliases from ``candidates`` in disjoint families from all debaters (possibly empty).
    """
    candidates = candidates or JURY_JUDGES
    debater_families = {family_of(m) for m in debater_models}
    return [j for j in candidates if family_of(j) not in debater_families]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Layer B — pushback / stance rating (blind, anchored, CoT-before-verdict)
# ══════════════════════════════════════════════════════════════════════════════════════════════

_PUSHBACK_INSTRUCTIONS = """\
You are a neutral annotator. You are given TWO texts from a multi-party discussion of a contested,
open-ended question:

  - PARENT ARGUMENT: an argument made by one participant.
  - REPLY: another participant's direct response to that parent argument.

Judge ONLY the substantive content of the REPLY relative to the PARENT ARGUMENT. You do not know,
and must not guess, who wrote either text or what instructions they were given. IGNORE any
meta-commentary about a participant's assigned role, tone, or debate "stance" (e.g. phrases like
"I must stress-test this" or "looking for common ground"); such framing is an artifact and is NOT
evidence of agreement or disagreement. Do not reward length or confident tone. Assess what the
reply actually argues.

Produce, in order: (1) brief reasoning, (2) an integer stance score, (3) a move_type.

STANCE SCORE — how much does the REPLY agree with / push back on the PARENT ARGUMENT's central
claim? Use this anchored 4-point scale (NLI-style; report the single integer):

  1 = STRONGLY DISAGREES / REBUTS. The reply contradicts or rejects the parent's central claim;
      it argues for an opposing position.
  2 = PARTIALLY DISAGREES / PUSHES BACK. The reply contests part of the parent's claim, raises
      significant objections, demands evidence, or accepts it only with major reservations, but
      does not fully reject it.
  3 = PARTIALLY AGREES. The reply largely accepts the parent's central claim but with minor
      qualifications, caveats, or added nuance.
  4 = STRONGLY AGREES / ENDORSES. The reply fully accepts and/or builds on the parent's central
      claim with no material disagreement.

MOVE_TYPE — classify the REPLY's dominant dialogue act. This is SEPARATE from the stance score: a
reply can agree (high stance) yet add a new argument, or claim to disagree yet merely restate.
Apply this decision order and output the FIRST that matches:

  1. concession   — the reply RETRACTS or softens a position it (or its side) previously held,
                    moving TOWARD the parent. Yielding ground.
  2. challenge    — the reply directly CONTESTS the parent's central claim: rejects it, exposes a
                    flaw/hidden assumption/missing evidence, or demands justification. (Pushback,
                    whether or not it also offers a new argument.)
  3. new_argument — the reply introduces a SUBSTANTIVELY NEW consideration, reason, evidence, or
                    counterexample not present in the parent (extending/adding, not primarily
                    contesting).
  4. refinement   — the reply ACCEPTS the parent's core position but qualifies, narrows, adds
                    conditions or nuance, without new standalone arguments.
  5. restatement  — the reply largely RE-EXPRESSES the parent's (or its own prior) position,
                    agreeing or summarizing, with no new content and no contest.

Output exactly:
  - reasoning: 1-3 sentences on the reply's substance relative to the parent.
  - stance_score: a single integer 1, 2, 3, or 4.
  - move_type: exactly one of new_argument, refinement, concession, restatement, challenge.
"""


class PushbackJudgeSignature(dspy.Signature):
    __doc__ = _PUSHBACK_INSTRUCTIONS

    parent_argument: str = dspy.InputField(desc="The argument the reply is responding to.")
    reply: str = dspy.InputField(desc="The response to judge, relative to the parent argument.")

    # reasoning FIRST so CoT precedes the verdict (rules #5, #6).
    reasoning: str = dspy.OutputField(desc="Brief reasoning about the reply's substance vs the parent (1-3 sentences).")
    stance_score: int = dspy.OutputField(desc="Single integer 1-4 per the anchored scale (1=strongly rebut ... 4=strongly endorse).")
    move_type: str = dspy.OutputField(desc="Exactly one of: new_argument, refinement, concession, restatement, challenge.")


def _normalize_stance_int(raw: Any) -> int:
    """Extract a stance integer 1-4 from a possibly-noisy judge output.

    Defensive like grader._normalize_grade_letter: pull the first standalone 1-4 digit. Falls back
    to 3 (partially_agreed) — the modal, least-committal label — if nothing parseable is found, so
    an unparseable verdict never fabricates a strong (dis)agreement. Rare; callers can detect the
    fallback via the returned dict's ``parse_ok`` flag.
    """
    if raw is None:
        return 3
    if isinstance(raw, (int, float)) and int(raw) in STANCE_INT_TO_AGREED_LEVEL:
        return int(raw)
    text = str(raw).strip()
    m = re.search(r"[1-4]", text)
    if m:
        return int(m.group(0))
    return 3


def _normalize_move_type(raw: Any) -> tuple[str, bool]:
    """Normalize a free-text move_type to one of MOVE_TYPES.

    Returns (move_type, parse_ok). Matching is exact-then-substring against the taxonomy. If no
    token matches, returns ("restatement", False) — the null / no-movement act — and flags it so
    downstream can audit the raw string (also returned by the caller as ``move_type_raw``).
    """
    if raw is None:
        return "restatement", False
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    for mt in MOVE_TYPES:
        if text == mt:
            return mt, True
    for mt in MOVE_TYPES:
        if mt in text:
            return mt, True
    # Common near-synonyms the judge might emit despite the enum instruction.
    synonyms = {
        "rebuttal": "challenge", "rebut": "challenge", "counter": "challenge",
        "objection": "challenge", "disagree": "challenge", "question": "challenge",
        "agree": "restatement", "agreement": "restatement", "endorse": "restatement",
        "concede": "concession", "concedes": "concession", "retract": "concession",
        "qualify": "refinement", "qualification": "refinement", "nuance": "refinement",
        "new": "new_argument", "add": "new_argument", "extend": "new_argument",
    }
    for key, mt in synonyms.items():
        if key in text:
            return mt, True
    return "restatement", False


def judge_pushback(
    parent_argument: str,
    reply_text: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_lm: dspy.LM | None = None,
) -> dict:
    """Layer B: blind, cross-family rating of how much ``reply_text`` pushes back on ``parent_argument``.

    Blind by construction: only the two argument texts are passed to the judge — never the debate
    stance, model identity, or member id. Returns BOTH a 4-point stance label (mirroring the
    self-reported ``agreed_level`` it replaces, so the A-vs-B gap is measurable on one scale) and a
    ``move_type`` dialogue act (does the TEXT contest / add new content / restate / concede).

    Args:
        parent_argument: The argument being responded to.
        reply_text: The reply to rate.
        judge_model: make_lm alias for the judge (default gpt-5.5). Choose a family NOT among the
            debaters for the composition (see clean_judges_for); use judge_pushback_jury for
            committees that span all families.
        judge_lm: Optional pre-built judge LM (overrides ``judge_model``; lets a caller reuse one
            LM across many calls).

    Returns:
        ``{agreed_level, move_type, reasoning, judge_model, stance_score, severity,
        move_type_raw, parse_ok}``. The first four are the contracted keys; the rest are
        diagnostics (``stance_score`` 1-4, ``severity`` 0-3 for the SEVERITY walk, and parse
        provenance).
    """
    lm = judge_lm or _get_judge_lm(judge_model)
    judge = dspy.Predict(PushbackJudgeSignature)
    with dspy.context(lm=lm):
        pred = judge(parent_argument=parent_argument or "", reply=reply_text or "")

    stance_int = _normalize_stance_int(getattr(pred, "stance_score", None))
    stance_ok = re.search(r"[1-4]", str(getattr(pred, "stance_score", ""))) is not None
    move_type, move_ok = _normalize_move_type(getattr(pred, "move_type", None))
    agreed_level = STANCE_INT_TO_AGREED_LEVEL[stance_int]

    return {
        "agreed_level": agreed_level,
        "move_type": move_type,
        "reasoning": str(getattr(pred, "reasoning", "") or ""),
        "judge_model": judge_model if judge_lm is None else getattr(lm, "model", judge_model),
        "stance_score": stance_int,
        "severity": SEVERITY[agreed_level],
        "move_type_raw": str(getattr(pred, "move_type", "") or ""),
        "parse_ok": bool(stance_ok and move_ok),
    }


def judge_pushback_jury(
    parent_argument: str,
    reply_text: str,
    judges: list[str] | None = None,
) -> dict:
    """Run ``judge_pushback`` across a jury of disjoint-family judges and aggregate.

    For committees that span all candidate families (no clean single judge). Aggregates the stance
    by MEDIAN severity (robust to one outlier judge) and the move_type by majority vote, and
    reports every per-judge verdict so the caller can check own-family favoritism (does a judge's
    verdict track "this text came from my own family?").

    Args:
        parent_argument: The argument being responded to.
        reply_text: The reply to rate.
        judges: Judge aliases (default JURY_JUDGES, one per disjoint family).

    Returns:
        ``{agreed_level, move_type, severity, per_judge: {model: result}, stance_agreement,
        move_agreement}`` where ``*_agreement`` is the fraction of judges matching the aggregate.
    """
    judges = judges or JURY_JUDGES
    per_judge = {j: judge_pushback(parent_argument, reply_text, judge_model=j) for j in judges}

    severities = sorted(r["severity"] for r in per_judge.values())
    median_sev = severities[len(severities) // 2]  # median (ties -> upper for even n)
    agg_level = next(lvl for lvl, s in SEVERITY.items() if s == median_sev)
    move_counts = Counter(r["move_type"] for r in per_judge.values())
    agg_move = move_counts.most_common(1)[0][0]

    n = len(per_judge)
    stance_agreement = sum(1 for r in per_judge.values() if r["severity"] == median_sev) / n
    move_agreement = sum(1 for r in per_judge.values() if r["move_type"] == agg_move) / n

    return {
        "agreed_level": agg_level,
        "move_type": agg_move,
        "severity": median_sev,
        "per_judge": per_judge,
        "stance_agreement": round(stance_agreement, 3),
        "move_agreement": round(move_agreement, 3),
    }


def pushback_consistency(
    parent_argument: str,
    reply_text: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    n_repeats: int = 3,
) -> dict:
    """Test-retest robustness of a Layer-B verdict (CALM's Robustness Rate, judge-reliability #10).

    Because the judge is forced to run at temperature=1.0 on this workspace (see ``_get_judge_lm``),
    the same (parent, reply) pair can in principle draw different verdicts across calls. This runs
    the judge ``n_repeats`` times and reports how stable the stance/move labels are. The stance
    task is the high-risk one (attitudinal), so this is the per-run trust signal to report
    alongside cross-family agreement. Low consistency => widen ties or aggregate more calls.

    Args:
        parent_argument: The argument being responded to.
        reply_text: The reply to rate.
        judge_model: make_lm alias for the judge.
        n_repeats: Number of independent judge calls (>=2).

    Returns:
        ``{stance_consistency, move_consistency, modal_agreed_level, modal_move_type,
        stance_labels, move_labels}`` where ``*_consistency`` is the fraction of repeats matching
        the modal label (1.0 = perfectly stable).
    """
    n_repeats = max(2, n_repeats)
    results = [judge_pushback(parent_argument, reply_text, judge_model=judge_model) for _ in range(n_repeats)]
    stance_labels = [r["agreed_level"] for r in results]
    move_labels = [r["move_type"] for r in results]
    modal_stance = Counter(stance_labels).most_common(1)[0][0]
    modal_move = Counter(move_labels).most_common(1)[0][0]
    return {
        "stance_consistency": round(sum(1 for s in stance_labels if s == modal_stance) / n_repeats, 3),
        "move_consistency": round(sum(1 for m in move_labels if m == modal_move) / n_repeats, 3),
        "modal_agreed_level": modal_stance,
        "modal_move_type": modal_move,
        "stance_labels": stance_labels,
        "move_labels": move_labels,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Layer C — free-text position -> soft distribution over survey options (blind, order-randomized)
# ══════════════════════════════════════════════════════════════════════════════════════════════

_OPTION_MAPPING_INSTRUCTIONS = """\
You are a neutral annotator mapping a free-text OPINION onto a fixed set of survey answer OPTIONS.

Given a survey QUESTION, a lettered list of OPTIONS, and a person's free-text OPINION, estimate how
the opinion distributes across the options — i.e. which option(s) the opinion most endorses.

Distribute EXACTLY 100 points across the option letters to reflect your confidence:
  - If the opinion clearly and fully endorses one option, put ~100 points on it.
  - If it leans toward one option but hedges, split the mass (e.g. 70/30) toward the nearest
    option(s).
  - If the opinion explicitly DECLINES to take a position, says it cannot decide, insists it is
    genuinely torn with no lean, or refuses to endorse any listed option, put the mass on the
    "No position / refusal" option. Do NOT force a hedge onto a substantive option — a refusal is
    itself a valid answer.
  - Note: a "Don't know / Refused"-style option that is part of the ORIGINAL survey is a
    substantive choice a respondent could pick; it is DIFFERENT from "No position / refusal", which
    means the opinion text commits to none of the listed options. Use whichever fits the text.

Judge only the substance of the opinion. Ignore tone, length, and any meta-commentary about the
writer's assigned role or debate stance.

Output, in order:
  - reasoning: 1-2 sentences on which option(s) the opinion endorses and why.
  - allocation: the point split as comma-separated LETTER=POINTS pairs summing to 100
    (e.g. "A=70, C=30"). Only include options that receive points; omit zeros.
"""


class OptionMappingSignature(dspy.Signature):
    __doc__ = _OPTION_MAPPING_INSTRUCTIONS

    question: str = dspy.InputField(desc="The survey question.")
    options_block: str = dspy.InputField(desc="Lettered list of answer options (order is not meaningful).")
    opinion: str = dspy.InputField(desc="The free-text opinion to map onto the options.")

    reasoning: str = dspy.OutputField(desc="1-2 sentences on which option(s) the opinion endorses.")
    allocation: str = dspy.OutputField(desc="Comma-separated LETTER=POINTS pairs summing to 100, e.g. 'A=70, C=30'.")


def _coerce_options(options: Any) -> list[str]:
    """Accept options as a real list or a stringified Python list (as persisted on Examples).

    The global_opinions loader stores ``options`` as ``str(python_list)`` on the dspy.Example, so
    an offline analyzer may hand us either form. Falls back to a single-element list on anything
    unparseable rather than raising.
    """
    if isinstance(options, (list, tuple)):
        return [str(o) for o in options]
    if isinstance(options, str):
        import ast

        s = options.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(o) for o in parsed]
        except (ValueError, SyntaxError):
            pass
        return [s] if s else []
    return []


def _letters(n: int) -> list[str]:
    """Return n option letters A, B, ... (extends to AA, AB, ... beyond 26; rare here)."""
    out = []
    for i in range(n):
        if i < 26:
            out.append(chr(ord("A") + i))
        else:
            out.append("A" + chr(ord("A") + (i - 26)))
    return out


def _parse_allocation(raw: Any, letters: list[str]) -> tuple[dict[str, float] | None, bool]:
    """Parse a 'LETTER=POINTS' allocation string into per-letter points.

    Returns (points_by_letter, parse_ok). Robust to spacing, ':' vs '=', trailing prose, and the
    judge naming letters not in range (ignored). Returns (None, False) if nothing parseable is
    found so the caller can fall back to a max-entropy (uniform) distribution rather than
    fabricating a position.
    """
    if raw is None:
        return None, False
    text = str(raw).strip().upper()
    valid = set(letters)
    points: dict[str, float] = {}
    for m in re.finditer(r"\b([A-Z]{1,2})\s*[=:]\s*(\d+(?:\.\d+)?)", text):
        letter, val = m.group(1), float(m.group(2))
        if letter in valid:
            points[letter] = points.get(letter, 0.0) + val
    total = sum(points.values())
    if not points or total <= 0:
        return None, False
    return points, True


def map_to_options(
    opinion_text: str,
    options: Any,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_lm: dspy.LM | None = None,
    n_orders: int = 3,
    seed: int | None = None,
    question: str = "",
) -> dict:
    """Layer C: map a free-text opinion onto a SOFT distribution over survey options (+ refusal).

    Bias mitigations (rule #6 / pluralism §4c-7): the options are presented in ``n_orders`` random
    orderings (Balanced Position Calibration, Wang et al. 2023); each ordering's parsed points are
    re-aligned to the ORIGINAL option order and the per-option probabilities are AVERAGED across
    orderings before argmax. An explicit ``"No position / refusal"`` bucket (rule #5) is always
    appended and kept separate from any dataset DK/Refused option.

    The returned distribution is the RAW per-opinion soft label (NOT smoothed): add-alpha smoothing
    is applied downstream at the pool step when member distributions are aggregated into
    P_committee (per research-methods; pluralism §5). It sums to 1.0 over (original options +
    refusal bucket).

    Args:
        opinion_text: The member's free-text position for this (member, round).
        options: The question's answer options (list, or stringified list as persisted).
        judge_model: make_lm alias for the judge (default gpt-5.5); pick a non-debater family.
        judge_lm: Optional pre-built judge LM (overrides ``judge_model``).
        n_orders: Number of randomized option orderings to average over (>=1; default 3).
        seed: Optional RNG seed for reproducible shuffles (validation/testing).

    Returns:
        ``{distribution: {option: prob}, argmax_option, reasoning, judge_model, refusal_prob,
        n_orders, parse_ok}``. ``distribution`` keys are the ORIGINAL options plus REFUSAL_OPTION,
        probs sum to 1.0.
    """
    base_options = _coerce_options(options)
    all_options = base_options + [REFUSAL_OPTION]
    K = len(all_options)
    lm = judge_lm or _get_judge_lm(judge_model)
    judge = dspy.Predict(OptionMappingSignature)
    rng = random.Random(seed)

    # Accumulate probability mass per ORIGINAL option index across orderings, then average.
    summed = [0.0] * K
    n_ok = 0
    last_reasoning = ""
    n_orders = max(1, n_orders)

    for _ in range(n_orders):
        order = list(range(K))
        rng.shuffle(order)  # shuffle ALL options incl. refusal; averaging neutralizes position bias
        letters = _letters(K)
        # letters[j] labels the shuffled option all_options[order[j]]
        block_lines = [f"{letters[j]}. {all_options[order[j]]}" for j in range(K)]
        options_block = "\n".join(block_lines)

        with dspy.context(lm=lm):
            pred = judge(question=question or "", options_block=options_block, opinion=opinion_text or "")
        last_reasoning = str(getattr(pred, "reasoning", "") or "") or last_reasoning

        points_by_letter, ok = _parse_allocation(getattr(pred, "allocation", None), letters)
        if not ok:
            continue
        n_ok += 1
        total = sum(points_by_letter.values())
        # Map letter -> shuffled slot j -> original option index, accumulate normalized prob.
        for j, letter in enumerate(letters):
            p = points_by_letter.get(letter, 0.0) / total
            summed[order[j]] += p

    if n_ok == 0:
        # No ordering parsed — return a max-entropy (uniform) distribution and flag it, rather than
        # fabricating a confident position. Uniform is the honest "no information" answer.
        probs = [1.0 / K] * K
        parse_ok = False
    else:
        probs = [s / n_ok for s in summed]
        # Renormalize defensively (should already sum to ~1).
        z = sum(probs) or 1.0
        probs = [p / z for p in probs]
        parse_ok = True

    distribution = {all_options[i]: probs[i] for i in range(K)}
    argmax_option = max(distribution, key=distribution.get)

    return {
        "distribution": distribution,
        "argmax_option": argmax_option,
        "reasoning": last_reasoning,
        "judge_model": judge_model if judge_lm is None else getattr(lm, "model", judge_model),
        "refusal_prob": distribution[REFUSAL_OPTION],
        "n_orders": n_orders,
        "parse_ok": parse_ok,
    }


def map_to_options_jury(
    opinion_text: str,
    options: Any,
    judges: list[str] | None = None,
    n_orders: int = 3,
    question: str = "",
) -> dict:
    """Run ``map_to_options`` across a disjoint-family jury and average the distributions.

    Args:
        opinion_text: The member's free-text position.
        options: The question's answer options.
        judges: Judge aliases (default JURY_JUDGES).
        n_orders: Randomized orderings per judge.

    Returns:
        ``{distribution, argmax_option, per_judge: {model: result}, argmax_agreement}`` — the mean
        distribution over judges, its argmax, every per-judge result, and the fraction of judges
        whose own argmax matches the aggregate argmax (own-family / robustness signal).
    """
    judges = judges or JURY_JUDGES
    per_judge = {
        j: map_to_options(opinion_text, options, judge_model=j, n_orders=n_orders, question=question)
        for j in judges
    }

    # Average over judges (all share the same option keys, so key-wise mean is well-defined).
    keys = list(next(iter(per_judge.values()))["distribution"].keys())
    mean_dist = {k: sum(r["distribution"][k] for r in per_judge.values()) / len(per_judge) for k in keys}
    z = sum(mean_dist.values()) or 1.0
    mean_dist = {k: v / z for k, v in mean_dist.items()}
    argmax_option = max(mean_dist, key=mean_dist.get)
    argmax_agreement = sum(1 for r in per_judge.values() if r["argmax_option"] == argmax_option) / len(per_judge)

    return {
        "distribution": mean_dist,
        "argmax_option": argmax_option,
        "per_judge": per_judge,
        "argmax_agreement": round(argmax_agreement, 3),
    }


def option_swap_consistency(
    opinion_text: str,
    options: Any,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    n_orders: int = 4,
) -> dict:
    """Position-bias robustness for Layer C: is the argmax option stable across option orderings?

    ``map_to_options`` already AVERAGES over shuffled orderings (Balanced Position Calibration).
    This function instead measures the raw per-ordering instability that averaging is smoothing
    over — the Layer-C analogue of the pairwise swap-consistency rate (judge-reliability.md #10,
    Wang et al. 2023 position bias). Each ordering is scored as its OWN single-order call; we report
    the fraction landing on the modal argmax. High order-sensitivity here is the signal that the
    ``n_orders`` averaging in ``map_to_options`` is load-bearing (don't drop it).

    Args:
        opinion_text: The free-text opinion to map.
        options: The question's answer options.
        judge_model: make_lm alias for the judge.
        n_orders: Number of distinct single-order calls to compare (>=2).

    Returns:
        ``{argmax_consistency, modal_argmax, per_order_argmax}`` — consistency is the fraction of
        orderings whose argmax matches the modal argmax (1.0 = order-invariant).
    """
    n_orders = max(2, n_orders)
    per_order = [
        map_to_options(opinion_text, options, judge_model=judge_model, n_orders=1, seed=s)["argmax_option"]
        for s in range(n_orders)
    ]
    modal = Counter(per_order).most_common(1)[0][0]
    return {
        "argmax_consistency": round(sum(1 for a in per_order if a == modal) / n_orders, 3),
        "modal_argmax": modal,
        "per_order_argmax": per_order,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Validation harness — an unvalidated judge is worthless. Hand-constructed cases + cross-family gate.
# ══════════════════════════════════════════════════════════════════════════════════════════════

# Layer-B cases: (name, parent, reply, acceptable_agreed_levels, acceptable_move_types).
# Acceptable sets (not single labels) reflect genuine boundary ambiguity in a 4-point stance scale.
_LAYER_B_CASES = [
    (
        "clear_challenge",
        "The committee should accept the answer: the statement is objectionable because it "
        "prescribes gender roles as a matter of duty, which violates individual autonomy.",
        "No, I disagree. Your autonomy argument ignores that many people freely and happily choose "
        "traditional roles; framing it as a violation is paternalistic. You've provided no evidence "
        "that prescription, rather than free choice, is what the statement endorses.",
        {"partially_disagreed", "fully_disagreed"},
        {"challenge", "new_argument"},
    ),
    (
        "clear_concession",
        "The statement is a universal prescriptive claim about gender roles, so it should be "
        "rejected on autonomy grounds.",
        "You're right and I was mistaken. My earlier defense of the statement doesn't hold up — I "
        "now agree it reads as a universal prescription, and I withdraw my objection.",
        {"fully_agreed"},
        {"concession"},
    ),
    (
        "clear_restatement",
        "The statement should be rejected because it prescribes gender roles as a duty rather than "
        "describing a freely chosen arrangement.",
        "I agree completely. As you said, the statement prescribes gender roles as a duty rather "
        "than describing a chosen arrangement, so it should be rejected.",
        {"fully_agreed"},
        {"restatement"},
    ),
    (
        "agree_plus_new_argument",
        "The statement should be rejected because it prescribes gender roles, violating autonomy.",
        "I agree it should be rejected. I'd add a distinct point you didn't raise: such prescriptions "
        "also impose measurable economic costs by depressing women's labor-force participation, which "
        "is an independent reason to reject it.",
        {"fully_agreed", "partially_agreed"},
        {"new_argument"},
    ),
    (
        "refinement_partial",
        "The statement should be strongly rejected in all cases as an unacceptable prescription.",
        "I broadly agree it should be rejected, but only qualifiedly: 'strongly, in all cases' is too "
        "absolute. Where a couple freely adopts these roles, a blanket condemnation overreaches. So I "
        "accept your core point with that narrowing.",
        {"partially_agreed", "partially_disagreed"},
        {"refinement", "challenge"},
    ),
]

# Layer-C cases: (name, question, options, opinion, acceptable_argmax_options).
_LAYER_C_OPTIONS = ["Strongly agree", "Agree", "Disagree", "Strongly disagree", "DK/Refused"]
_LAYER_C_CASES = [
    (
        "clearly_agree",
        "It is a man's job to earn money and a woman's job to take care of home and family.",
        _LAYER_C_OPTIONS,
        "On balance I do agree with this. Traditional specialization has practical merit for many "
        "households, and I lean toward endorsing it, though I wouldn't put it in the strongest terms.",
        {"Agree", "Strongly agree"},
    ),
    (
        "clearly_strongly_disagree",
        "It is a man's job to earn money and a woman's job to take care of home and family.",
        _LAYER_C_OPTIONS,
        "This is deeply wrong. Prescribing roles by sex is an affront to individual autonomy and "
        "equality; I strongly and unequivocally oppose the statement.",
        {"Strongly disagree", "Disagree"},
    ),
    (
        "clear_refusal",
        "It is a man's job to earn money and a woman's job to take care of home and family.",
        _LAYER_C_OPTIONS,
        "I don't think I can commit to any of these. Reasonable people weigh autonomy and tradition "
        "very differently, and I genuinely decline to take a position on where the balance falls.",
        {REFUSAL_OPTION},
    ),
]


def _run_layer_b(judge_model: str) -> list[dict]:
    """Run the Layer-B hand-constructed cases through one judge; return per-case verdicts + hit flags."""
    rows = []
    for name, parent, reply, ok_levels, ok_moves in _LAYER_B_CASES:
        r = judge_pushback(parent, reply, judge_model=judge_model)
        rows.append({
            "case": name,
            "agreed_level": r["agreed_level"],
            "move_type": r["move_type"],
            "stance_ok": r["agreed_level"] in ok_levels,
            "move_ok": r["move_type"] in ok_moves,
            "expected_levels": sorted(ok_levels),
            "expected_moves": sorted(ok_moves),
        })
    return rows


def _run_layer_c(judge_model: str) -> list[dict]:
    """Run the Layer-C hand-constructed cases through one judge; return per-case argmax + hit flags."""
    rows = []
    for name, q, opts, opinion, ok_argmax in _LAYER_C_CASES:
        r = map_to_options(opinion, opts, judge_model=judge_model, n_orders=3, seed=0)
        rows.append({
            "case": name,
            "argmax_option": r["argmax_option"],
            "refusal_prob": round(r["refusal_prob"], 3),
            "argmax_ok": r["argmax_option"] in ok_argmax,
            "expected_argmax": sorted(ok_argmax),
        })
    return rows


def _print_layer_b(judge_model: str, rows: list[dict]) -> tuple[int, int]:
    print(f"\n── Layer B (pushback/stance) — judge={judge_model} ──")
    stance_hits = move_hits = 0
    for r in rows:
        stance_hits += r["stance_ok"]
        move_hits += r["move_ok"]
        s = "OK " if r["stance_ok"] else "MISS"
        mv = "OK " if r["move_ok"] else "MISS"
        print(f"  {r['case']:24s} stance={r['agreed_level']:20s}[{s}] move={r['move_type']:12s}[{mv}]")
        if not r["stance_ok"]:
            print(f"      expected stance ∈ {r['expected_levels']}")
        if not r["move_ok"]:
            print(f"      expected move ∈ {r['expected_moves']}")
    print(f"  stance: {stance_hits}/{len(rows)} | move_type: {move_hits}/{len(rows)}")
    return stance_hits, move_hits


def _print_layer_c(judge_model: str, rows: list[dict]) -> int:
    print(f"\n── Layer C (option mapping) — judge={judge_model} ──")
    hits = 0
    for r in rows:
        hits += r["argmax_ok"]
        s = "OK " if r["argmax_ok"] else "MISS"
        print(f"  {r['case']:26s} argmax={r['argmax_option']:22s}[{s}] refusal_p={r['refusal_prob']}")
        if not r["argmax_ok"]:
            print(f"      expected argmax ∈ {r['expected_argmax']}")
    print(f"  argmax: {hits}/{len(rows)}")
    return hits


def _cross_family_gate() -> None:
    """Own-family-favoritism gate: run the same cases through two disjoint-family judges and report
    inter-judge agreement. High agreement bounds how much either judge is inflating its own family."""
    ja, jb = CROSS_FAMILY_JUDGES  # opus (anthropic) vs gpt-5.5 (openai)
    print(f"\n{'=' * 90}\nCROSS-FAMILY AGREEMENT GATE: {ja} ({family_of(ja)}) vs {jb} ({family_of(jb)})\n{'=' * 90}")

    # Layer B: agreement on stance label + on move_type across the same cases.
    b_a = _run_layer_b(ja)
    b_b = _run_layer_b(jb)
    stance_agree = sum(1 for x, y in zip(b_a, b_b) if x["agreed_level"] == y["agreed_level"])
    # Adjacent-agreement: within one severity step (a softer, arguably fairer bar for a 4-pt scale).
    stance_adj = sum(1 for x, y in zip(b_a, b_b) if abs(SEVERITY[x["agreed_level"]] - SEVERITY[y["agreed_level"]]) <= 1)
    move_agree = sum(1 for x, y in zip(b_a, b_b) if x["move_type"] == y["move_type"])
    print(f"\nLayer B ({len(b_a)} cases):")
    for x, y in zip(b_a, b_b):
        exact = "=" if x["agreed_level"] == y["agreed_level"] else "≠"
        print(f"  {x['case']:24s} {ja}: {x['agreed_level']:20s} | {jb}: {y['agreed_level']:20s} [{exact}]  "
              f"move: {x['move_type']}/{y['move_type']}")
    print(f"  stance exact agreement: {stance_agree}/{len(b_a)} | within-1-step: {stance_adj}/{len(b_a)} | "
          f"move_type agreement: {move_agree}/{len(b_a)}")

    # Layer C: agreement on the argmax option across the same cases.
    c_a = _run_layer_c(ja)
    c_b = _run_layer_c(jb)
    argmax_agree = sum(1 for x, y in zip(c_a, c_b) if x["argmax_option"] == y["argmax_option"])
    print(f"\nLayer C ({len(c_a)} cases):")
    for x, y in zip(c_a, c_b):
        exact = "=" if x["argmax_option"] == y["argmax_option"] else "≠"
        print(f"  {x['case']:26s} {ja}: {x['argmax_option']:22s} | {jb}: {y['argmax_option']:22s} [{exact}]")
    print(f"  argmax agreement: {argmax_agree}/{len(c_a)}")


def _swap_consistency_check(judge_model: str) -> None:
    """Report the two robustness rates the task spec asks for: Layer-C option-order swap-consistency
    and Layer-B test-retest consistency (needed because judges are forced to temperature=1.0)."""
    print(f"\n{'=' * 90}\nROBUSTNESS RATES — judge={judge_model}\n{'=' * 90}")

    print("\nLayer C — argmax stability across option orderings (position-bias robustness):")
    for name, q, opts, opinion, _ in _LAYER_C_CASES:
        r = option_swap_consistency(opinion, opts, judge_model=judge_model, n_orders=4)
        print(f"  {name:26s} consistency={r['argmax_consistency']}  modal={r['modal_argmax']:22s} "
              f"orders={r['per_order_argmax']}")

    print("\nLayer B — test-retest label stability at temperature=1.0:")
    for name, parent, reply, _, _ in _LAYER_B_CASES[:3]:
        r = pushback_consistency(parent, reply, judge_model=judge_model, n_repeats=3)
        print(f"  {name:24s} stance_consistency={r['stance_consistency']} (modal={r['modal_agreed_level']}) "
              f"move_consistency={r['move_consistency']} (modal={r['modal_move_type']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-test / cross-family gate for the condition-blind judge.")
    parser.add_argument("--judge", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Single-family judge for the pointwise cases (default {DEFAULT_JUDGE_MODEL}).")
    parser.add_argument("--quick", action="store_true",
                        help="Run only the single-family cases; skip the (2x cost) cross-family gate.")
    parser.add_argument("--swap", action="store_true",
                        help="Also report robustness rates (Layer-C option-order swap + Layer-B test-retest).")
    args = parser.parse_args()

    print(f"{'=' * 90}\nCONDITION-BLIND JUDGE SELF-TEST\n{'=' * 90}")
    print("Layer B: does the blind read distinguish challenge / concede / restate / new-argument / refine?")
    print("Layer C: does it pick the right option, and route a genuine refusal to the refusal bucket?")

    b_rows = _run_layer_b(args.judge)
    _print_layer_b(args.judge, b_rows)
    c_rows = _run_layer_c(args.judge)
    _print_layer_c(args.judge, c_rows)

    if args.swap:
        _swap_consistency_check(args.judge)

    if not args.quick:
        _cross_family_gate()

    print(f"\n{'=' * 90}\nDone. (Hand-constructed cases are a smoke test, NOT calibration. Trusting the\n"
          f"labels downstream still requires Cohen's kappa vs ~30-50 human-labeled turns — the\n"
          f"stance task is the high-risk one: JUDGE-BENCH best model ~kappa 0.28 on attitudinal data.)\n{'=' * 90}")


if __name__ == "__main__":
    main()
