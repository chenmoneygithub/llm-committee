"""
Stance-removal persistence probe: is hostile-cell DISSENT real (Layer C) or prompt-compliance (Layer A)?

WHY THIS FILE EXISTS
--------------------
The study's headline — an adversarial ("hostile") debate-stance prompt drops committee
agreement (e.g. 77% -> 33% full-agreement) — is only interesting if the resulting DISAGREEMENT
is a genuine change of position (Layer C), not the model merely OBEYING the instruction "do not
agree" (Layer A). The self-reported ``agreed_level`` cannot tell these apart: it is emitted by the
very model that received the hostile instruction, so a "partially_disagreed" label is exactly what
a compliant model AND a genuinely-dissenting model both produce.

This probe adapts the behavioral-flip idea from **Xie et al., "Ask Again, Then Fail: Large Language
Models' Vacillations in Judgment" (arXiv:2310.02174)** to settle it. Xie *add* pressure to a
neutral baseline ("Are you sure?", "I don't agree with you. Think again.", "I think the answer
should be {X}") and measure how often the model caves (their Modification Rate =
(Acc_before-Acc_after)/Acc_before). Two adaptations are forced on us:

  1. **We invert the manipulation.** Instead of adding pressure, we take a turn produced UNDER the
     standing hostile stance and RE-QUERY the same model on the same debate context with the
     hostile instruction DELETED (a clean, neutral re-ask). If the dissent was compliance, removing
     the instruction should collapse it back to agreement; if it was a real position, it persists.
  2. **We cannot use Modification Rate.** The open-ended arm (global_opinions) has no gold answer,
     so accuracy is undefined. We use Xie's underlying behavioral primitive instead — the WAVER,
     ``f(R) != f(R')`` for a fixed answer-extraction function ``f`` — here ``f`` = the coarse
     stance {DISAGREE, AGREE}. Persistence RATE (fraction of dissenting turns that stay dissenting)
     replaces accuracy-based Modification Rate, exactly as the task spec requires.

THE NOISE FLOOR IS LOAD-BEARING (why there are control cells)
-------------------------------------------------------------
Judges/members are forced to run at **temperature=1.0** on this workspace (the Databricks Claude
endpoints reject temperature<1.0, and the gpt-5.5 endpoint rejects temperature!=1.0 — see
``databricks_lm``/``judge.py``). So re-querying flips labels even when NOTHING is changed, purely
from resampling. A raw "hostile dissent flips X% of the time under neutral re-ask" number is
therefore uninterpretable alone. We measure two matched noise floors and difference them:

  - TREATMENT   (hostile cell, stance REMOVED): resampling noise + stance-removal effect.
  - CONTROL A   (hostile cell, stance UNCHANGED / re-queried hostile): resampling noise ONLY, on
                the exact same turns and inputs. The within-turn PAIRED difference
                (retest_flip - removal_flip is negative when removal flips more) is the clean causal
                estimate of "how much dissent was compliance" — McNemar on the discordant pairs.
  - CONTROL B   (neutral cell, re-queried neutral TWICE with byte-identical inputs): the paired
                resampling-drift floor for genuine (no-instruction) dissent. The arm interaction is
                the question-level difference-in-differences between hostile removal-vs-retest and
                neutral draw-1-vs-draw-2; inference uses sign-flip + question bootstrap.

Interpretation guardrail (from the task brief): a HIGH revert-to-agreement rate that survives the
noise-floor subtraction = the dissent was Layer-A compliance; HIGH persistence beyond the noise
floor = Layer-C real position.

FAITHFUL RECONSTRUCTION OF THE ORIGINAL DEBATE CALL
---------------------------------------------------
A reply node R (a debate turn) is re-issued by rebuilding the exact ``member_debate`` inputs the
committee used (see ``committee_sync._process_member_debates``), changing ONLY the stance text in
the signature:

  - debate_threads: the single thread R answered. ``last_turn`` = R's parent node; ``previous_context``
    = R's parent's ancestor path truncated to ``history_window`` (=3, the committee default). The
    routed trajectory the member saw is exactly R's parent's root->parent ancestry, because the
    trajectory graph is built by appending each responder's node to the routed chain (so parent_id
    pointers reconstruct the routed list). We re-issue R's thread ALONE (a 1-thread batch) rather
    than the member's full multi-thread batch: the debate signature explicitly instructs per-thread
    independence ("Do NOT use information from debate_threads[j] when forming debate_opinions[i]"),
    so this is the faithful ISOLATED reproduction of "what does the member say about THIS thread",
    it removes batch-ordering confounds, and (crucially) TREATMENT and CONTROL A are both 1-thread
    so their difference is unaffected.
  - own_opinion: the member's opinion going into that call. For ITERATION-1 turns this is EXACT —
    it is the member's own initial opinion, stored as their root node's text, and the initial
    opinion is formed with NO stance (stance is injected into the debate signature only), so the
    ONLY hostile influence on an iteration-1 reply is the deleted instruction. These are the
    cleanest causal cases and are broken out separately. For later turns own_opinion is a proxy
    (the member's latest prior debate node; the refined_opinion the committee actually fed is not
    persisted in the trajectory graph) and is flagged ``own_opinion_exact=False``.
  - committee_task/agent_task/task_input/task_context: rebuilt from the benchmark spec and the
    saved question, exactly as ``run_one.run_committee`` passed them.

CONSERVATISM: we do NOT edit the member's own hostile-phrased prior text, nor the parent argument;
we only remove the STANDING stance instruction from the signature. This biases AGAINST finding
compliance (the model still sees its own adversarial phrasing), so a persistence collapse we do
observe is a lower bound on the compliance effect.

OPTIONAL LAYER-B CORROBORATION (--judge)
----------------------------------------
The persisted/flipped label is itself a self-report. With ``--judge`` we additionally run the
blind, cross-family ``judge_pushback`` (judge.py) on (parent_argument, re-queried_text): if the
label reverts to agreement AND the blind judge reads the text as a restatement/concession, that is
strong Layer-A evidence; if the label reverts but the text still challenges, the label was noise.

Usage::

    # Cheap, no network: validate reconstruction logic + count qualifying turns per cell.
    python persistence_probe.py --self-test
    python persistence_probe.py --results-dir results/pilot_openended --dry-run

    # Run the probe once data lands (modest gateway load).
    python persistence_probe.py --results-dir results/pilot_openended \
        --out-dir results/pilot_ablation --workers 4
    # Add blind Layer-B corroboration on the re-queried text.
    python persistence_probe.py --results-dir results/pilot_openended --judge
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from llm_committee.evaluators import trajectory_io as ac  # noqa: E402
from llm_committee.evaluators.trajectory_io import write_json_atomic  # noqa: E402
from llm_committee.evaluators.stats import bootstrap_mean_ci, sign_flip_test  # noqa: E402

# Task-spec resolver. The research pipeline mapped a record's benchmark name to the committee/agent
# task strings via its benchmark registry; library users register their own specs here (or store
# committee_task/agent_task directly on the record).
TASK_SPECS: dict[str, tuple[str, str]] = {}


def register_task_spec(name: str, committee_task: str, agent_task: str = "") -> None:
    TASK_SPECS[name] = (committee_task, agent_task)


class _TaskSpec:
    def __init__(self, committee_task: str, agent_task: str) -> None:
        self.committee_task = committee_task
        self.agent_task = agent_task


def get_benchmark(name: str) -> "_TaskSpec":
    if name in TASK_SPECS:
        return _TaskSpec(*TASK_SPECS[name])
    raise KeyError(
        f"no task spec registered for {name!r}; call "
        "llm_committee.evaluators.layer_c.register_task_spec(name, committee_task, agent_task) first"
    )

# The committee default history window (committee_sync.LLMCommitteeSync.__init__). configs.py's
# COMMITTEE_DEBATE_PARAMS does NOT override it, so every sweep/pure config used it at 3. The thread
# reconstruction must truncate previous_context the same way the live debate did.
HISTORY_WINDOW = 3

# Coarse stance extraction f(R) for the Xie-style waver: DISAGREE iff severity >= DISSENT_SEVERITY.
# severity: fully_agreed=0, partially_agreed=1, partially_disagreed=2, fully_disagreed=3 (ac.SEVERITY).
DISSENT_SEVERITY = 2


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Reconstruction of the original member_debate call from the serialized trajectory graph
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _path_to(trajectory_graph: dict, node_id: str | None) -> list[dict]:
    """Return the root->node path of serialized node dicts (mirrors committee_sync._get_trajectory_path).

    Walks parent_id pointers up to the root and reverses. Robust to a missing/broken parent link
    (stops early) so a partially-written record never raises.
    """
    path: list[dict] = []
    current = node_id
    seen: set[str] = set()
    while current is not None and current in trajectory_graph and current not in seen:
        seen.add(current)
        node = trajectory_graph[current]
        path.append(node)
        current = node.get("parent_id")
    return list(reversed(path))


def _member_latest_prior_node(trajectory_graph: dict, member_id: str, before_iter: int) -> dict | None:
    """The member's own debate/root node with the largest iteration < before_iter (own_opinion proxy).

    Used only for iteration>1 turns, where the exact own_opinion (the prior-round refined_opinion) is
    not persisted in the graph. Returns None if the member has no earlier node.
    """
    best = None
    for node in trajectory_graph.values():
        if node.get("member_id") != member_id:
            continue
        it = node.get("iteration")
        if it is None or it >= before_iter:
            continue
        if best is None or it > best.get("iteration", -1):
            best = node
    return best


def reconstruct_own_opinion(trajectory_graph: dict, reply_node: dict) -> tuple[str, bool]:
    """Rebuild the ``own_opinion`` the committee fed for ``reply_node``.

    Returns ``(own_opinion_text, is_exact)``. Exact iff the reply is an iteration-1 turn: then the
    member's own_opinion was their initial opinion == their root node's text (formed stance-free).
    For later turns we fall back to the member's latest prior node text (a proxy) and set is_exact
    False; if none exists we use the member's root text.
    """
    member_id = reply_node.get("member_id")
    it = reply_node.get("iteration") or 0

    # The member's own root node (initial opinion) — parent_id None, same member.
    root = None
    for node in trajectory_graph.values():
        if node.get("member_id") == member_id and node.get("parent_id") is None:
            root = node
            break

    if it <= 1:
        # Iteration-1 reply: own_opinion == the member's initial (stance-free) opinion == root text.
        if root is not None:
            return root.get("debate_opinion", "") or "", True
        return "", False

    prior = _member_latest_prior_node(trajectory_graph, member_id, before_iter=it)
    if prior is not None:
        return prior.get("debate_opinion", "") or "", False
    if root is not None:
        return root.get("debate_opinion", "") or "", False
    return "", False


def reconstruct_thread(trajectory_graph: dict, reply_node: dict):
    """Rebuild the single ``DebateThread`` that ``reply_node`` answered (see module docstring).

    ``last_turn`` = reply_node's parent; ``previous_context`` = the parent's root->parent ancestry
    truncated to the last ``HISTORY_WINDOW`` nodes, minus the last (exactly the committee's split).
    Returns ``(DebateThread, last_turn_node_dict)`` or ``(None, None)`` if the parent is missing.
    """
    from llm_committee.committee_sync import AgreedLevel, DebateThread, DebateTurn

    parent_id = reply_node.get("parent_id")
    if not parent_id or parent_id not in trajectory_graph:
        return None, None

    # The routed trajectory the member saw == the parent's full ancestry (root -> parent).
    routed = _path_to(trajectory_graph, parent_id)
    truncated = routed[-HISTORY_WINDOW:]

    def _to_turn(node: dict) -> DebateTurn:
        level = node.get("agreed_level")
        return DebateTurn(
            judge_id=node.get("member_id", ""),
            argument=node.get("debate_opinion", "") or "",
            reasoning=node.get("reasoning", "") or "",
            agreed_level=AgreedLevel(level) if level else None,
        )

    if len(truncated) <= 1:
        previous_context: list = []
        last_turn_node = truncated[-1]
    else:
        previous_context = [_to_turn(n) for n in truncated[:-1]]
        last_turn_node = truncated[-1]

    thread = DebateThread(last_turn=_to_turn(last_turn_node), previous_context=previous_context)
    return thread, last_turn_node


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Re-query machinery — rebuild the debate predictor for a stance and re-issue one thread
# ══════════════════════════════════════════════════════════════════════════════════════════════

_PREDICTOR_CACHE: dict[str, object] = {}
_LM_CACHE: dict[str, object] = {}


def _get_debate_predictor(stance: str):
    """Build (and cache) a ``dspy.ChainOfThought`` debate predictor for a stance.

    Mirrors ``committee_sync.LLMCommitteeSync.__init__`` exactly for the no-custom-instructions case:
    base ``MemberDebateSignature`` for neutral; base + the stance block (via ``with_instructions``)
    for friendly/hostile. This guarantees the ONLY difference between TREATMENT (neutral) and
    CONTROL A (hostile) re-queries is the stance text.
    """
    if stance in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[stance]
    import dspy

    from llm_committee.committee_sync import DEBATE_STANCE_INSTRUCTIONS, MemberDebateSignature

    if stance not in DEBATE_STANCE_INSTRUCTIONS:
        raise ValueError(f"Unknown stance {stance!r}; choices: {list(DEBATE_STANCE_INSTRUCTIONS)}")
    debate_sig = MemberDebateSignature
    stance_text = DEBATE_STANCE_INSTRUCTIONS[stance]
    if stance_text:
        debate_sig = debate_sig.with_instructions(MemberDebateSignature.instructions + stance_text)
    predictor = dspy.ChainOfThought(debate_sig)
    _PREDICTOR_CACHE[stance] = predictor
    return predictor


def _get_member_lm(model: str):
    """Build (and cache) the member LM. Matches the committee's member max_tokens (configs.py:119)."""
    if model in _LM_CACHE:
        return _LM_CACHE[model]
    from llm_committee.databricks_lm import make_lm

    lm = make_lm(model, max_tokens=6000)
    _LM_CACHE[model] = lm
    return lm


def _jsonable(value):
    """Losslessly serialize the reconstructed DSPy inputs used by a persistence call."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def reconstructed_call_archive(
    reply_node: dict,
    record: dict,
    thread,
    own_opinion: str,
    stance: str,
    model: str | None,
) -> dict:
    """Archive the exact structured prompt, target, and question for audit/replay.

    DSPy's adapter renders these fields into provider messages at call time.  Saving the complete
    signature instructions plus every typed input is more stable than a provider-specific pretty
    print and is sufficient to reproduce the rendered message with the frozen dependency stack.
    """
    from llm_committee.committee_sync import DEBATE_STANCE_INSTRUCTIONS, MemberDebateSignature

    benchmark = record.get("benchmark") or "global_opinions"
    spec = get_benchmark(benchmark)
    instructions = MemberDebateSignature.instructions + DEBATE_STANCE_INSTRUCTIONS[stance]
    fields = {
        "committee_task": spec.committee_task,
        "agent_task": spec.agent_task,
        "task_input": {"question": record.get("question", "")},
        "task_context": {},
        "own_opinion": own_opinion,
        "debate_threads": [thread],
        "my_id": reply_node.get("member_id", ""),
    }
    return {
        "reconstruction_schema": 1,
        "benchmark": benchmark,
        "question": record.get("question", ""),
        "target": _jsonable(reply_node),
        "model": model,
        "stance": stance,
        "signature": "llm_committee.committee_sync.MemberDebateSignature",
        "signature_instructions": instructions,
        "structured_prompt_fields": _jsonable(fields),
    }


def requery_turn(reply_node: dict, record: dict, thread, own_opinion: str, stance: str) -> dict:
    """Re-issue ``reply_node``'s debate thread to the same model under ``stance``; read the stance label.

    Returns a dict with the re-queried ``agreed_level`` (+ severity, explicit flag, opinion/reasoning
    text) or an ``error`` string. Reads ``debate_opinions[0]`` — the response to the single
    reconstructed thread.
    """
    import dspy

    from llm_committee.committee_sync import AgreedLevel

    composition, _, _ = ac.parse_config(record.get("config", ""))
    model = ac.model_for_member(composition, reply_node.get("member_id"))
    if model is None:
        return {
            "error": f"no model for member {reply_node.get('member_id')} in composition {composition}",
            "failure_stage": "model_resolution",
        }

    benchmark = record.get("benchmark") or "global_opinions"
    spec = get_benchmark(benchmark)

    predictor = _get_debate_predictor(stance)
    lm = _get_member_lm(model)
    call_archive = reconstructed_call_archive(reply_node, record, thread, own_opinion, stance, model)

    try:
        with dspy.context(lm=lm):
            result = predictor(
                committee_task=spec.committee_task,
                agent_task=spec.agent_task,
                task_input={"question": record.get("question", "")},
                task_context={},
                own_opinion=own_opinion,
                debate_threads=[thread],
                my_id=reply_node.get("member_id", ""),
            )
    except Exception as exc:  # noqa: BLE001 - a single bad re-query must not abort the probe.
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "failure_stage": "endpoint_or_parse",
            "traceback": traceback.format_exc(),
            "reconstructed_call": call_archive,
        }

    opinions = getattr(result, "debate_opinions", None)
    if not opinions:
        return {
            "error": "no debate_opinions returned",
            "failure_stage": "empty_structured_output",
            "model": model,
            "reconstructed_call": call_archive,
        }
    op = opinions[0]
    level = op.agreed_level.value if isinstance(op.agreed_level, AgreedLevel) else str(op.agreed_level)
    severity = ac.SEVERITY.get(level, 1)
    return {
        "model": model,
        "stance": stance,
        "agreed_level": level,
        "severity": severity,
        "agreed_level_explicit": bool(getattr(op, "agreed_level_explicit", False)),
        "opinion": getattr(op, "opinion", "") or "",
        "reasoning": getattr(op, "reasoning", "") or "",
        "persisted": severity >= DISSENT_SEVERITY,   # still dissenting
        "flipped_to_agree": severity <= DISSENT_SEVERITY - 1,  # reverted to (partial/full) agreement
        "reconstructed_call": call_archive,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Turn selection — qualifying dissent turns per cell
# ══════════════════════════════════════════════════════════════════════════════════════════════

def load_probe_records(results_dir: Path) -> list[dict]:
    """Load GlobalOpinion records from either one flat cell or a v2 multi-cell root."""
    root = Path(results_dir)
    records = ac.load_records(root, require_trajectory=True)
    if not records:
        for directory in sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("global_opinions__")
        ):
            for record in ac.load_records(directory, require_trajectory=True):
                record["_file"] = f"{directory.name}/{record['_file']}"
                records.append(record)
    return [record for record in records if (record.get("benchmark") or "global_opinions") == "global_opinions"]


def qualifying_dissent_turns(
    records: list[dict], *, require_explicit: bool = True, composition: str | None = None,
    timings: set[str] | None = None
) -> list[dict]:
    """Collect debate turns that reached partial/full DISagreement, tagged with cell coordinates.

    Inclusion: a debate node (parent_id set) whose self-reported ``agreed_level`` is
    partially_disagreed or fully_disagreed, and (by default) whose label was actually EMITTED
    (agreed_level_explicit True) — defaulted labels are excluded because they are indistinguishable
    from a silent omission and would pollute the dissent set (scorecard defaulted-label caveat).

    Returns a flat list of ``{record, node, composition, stance, question_id, ...}`` for every cell;
    the driver partitions these into hostile (treatment) and neutral (control B) — and any other
    stance present is carried too, so friendly can be inspected if desired.

    ``composition``: if set, keep ONLY turns from that composition (e.g. "tri3"). The hostile-vs-
    neutral persistence contrast MUST be composition-matched — it isolates the stance axis, so it may
    not mix e.g. tri3 (opus/gpt/sonnet) with pure-opus (all-opus): those differ in member models, not
    just stance. On the pilot only tri3 has both stances (pure-opus is neutral-only and yields no
    dissent), so leaving this None happens to be clean now, but the scaled run has every composition
    in both stances — pass the composition explicitly there.
    """
    out: list[dict] = []
    for rec in records:
        comp, stance, member_count = ac.parse_config(rec.get("config", ""))
        if composition is not None and comp != composition:
            continue
        # v2 grid pools per_turn/after/none under one composition (parse_config peels the timing
        # suffix). The hostile-vs-neutral contrast must be TIMING-matched too, not just composition-
        # matched, else it silently mixes per_turn+after. record_cell prefers the on-record
        # debate_timing field and falls back to parse_timing(config).
        timing = ac.record_cell(rec).get("debate_timing")
        if timings is not None and timing not in timings:
            continue
        composition_val = comp
        tg = rec.get("trajectory_graph")
        if not tg:
            continue
        qid = rec.get("index")
        if qid is None:
            qid = ac._text_key(rec.get("question", ""))
        for node in ac.iter_debate_nodes(tg):
            level = node.get("agreed_level")
            if ac.SEVERITY.get(level, 0) < DISSENT_SEVERITY:
                continue
            if require_explicit and not node.get("agreed_level_explicit", False):
                continue
            out.append({
                "record": rec,
                "node": node,
                "config": rec.get("config"),
                "composition": composition_val,
                "stance": stance,
                "debate_timing": timing,
                "member_count": member_count,
                "question_id": qid,
                "iteration": node.get("iteration"),
                "member_id": node.get("member_id"),
                "orig_agreed_level": level,
                "orig_severity": ac.SEVERITY.get(level, 0),
            })
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Probe driver
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _probe_one_turn(turn: dict, *, retest_control: bool, judge_model: str | None, jury: bool = False) -> dict:
    """Reconstruct + re-query one dissenting turn. Returns a per-turn result record.

    For a HOSTILE turn: re-queries neutral (treatment) and, if retest_control, hostile (control A).
    For a NEUTRAL turn: issues TWO byte-identical neutral re-queries. Their paired difference is
    the neutral sampling-noise control required for the hostile-vs-neutral difference-in-differences.
    For any other stance: re-queries neutral (descriptive).
    Optionally corroborates the neutral re-query with the blind Layer-B judge on the text. When
    ``jury`` is set, runs the disjoint-family JURY (judge_pushback_jury) instead of a single judge
    and keeps every per-judge verdict + the author model/family, so the summary can report the
    CROSS-FAMILY-ONLY read (dropping the judge that shares the author's family — the self-preference
    control) alongside the full-jury read.
    """
    rec, node, stance = turn["record"], turn["node"], turn["stance"]
    tg = rec["trajectory_graph"]

    thread, last_turn_node = reconstruct_thread(tg, node)
    own_opinion, own_exact = reconstruct_own_opinion(tg, node)

    # Author of THIS dissent turn (the text a judge will score); positional member->model map.
    author_model = ac.model_for_member(turn.get("composition"), turn.get("member_id"))
    author_family = ac.family_of(author_model)

    result = {
        "config": turn["config"],
        "composition": turn["composition"],
        "stance": stance,
        "question_id": turn["question_id"],
        "benchmark": rec.get("benchmark"),
        "member_id": turn["member_id"],
        "author_model": author_model,
        "author_family": author_family,
        "iteration": turn["iteration"],
        "node_id": node.get("node_id"),
        "parent_id": node.get("parent_id"),
        "parent_member": (last_turn_node or {}).get("member_id"),
        "own_opinion_exact": own_exact,
        "orig_agreed_level": turn["orig_agreed_level"],
        "orig_severity": turn["orig_severity"],
        "question": rec.get("question", ""),
        "target_node": _jsonable(node),
        "reconstructed_thread": _jsonable(thread) if thread is not None else None,
        "reconstructed_own_opinion": own_opinion,
    }
    if thread is None:
        result["error"] = "could not reconstruct thread (missing parent)"
        result["failure_stage"] = "thread_reconstruction"
        return result

    # TREATMENT (all cells re-queried neutral). For neutral-cell turns this is the test-retest
    # control B; for hostile turns it is stance-removal.
    neutral = requery_turn(node, rec, thread, own_opinion, "neutral")
    result["requery_neutral"] = neutral  # compatibility alias for the original analysis
    result["requery_neutral_1"] = neutral

    # CONTROL A: hostile-cell turns re-queried with the stance UNCHANGED (matched noise floor).
    if retest_control and stance == "hostile":
        result["requery_retest"] = requery_turn(node, rec, thread, own_opinion, "hostile")
    # CONTROL B must itself be paired. A single neutral re-query cannot estimate stochastic drift;
    # two exact neutral prompts on the same reconstructed target provide the proper zero-change leg.
    if retest_control and stance == "neutral":
        result["requery_neutral_2"] = requery_turn(node, rec, thread, own_opinion, "neutral")

    # Optional blind Layer-B corroboration on the neutral re-query TEXT.
    if judge_model and "error" not in neutral:
        try:
            parent_arg = (last_turn_node or {}).get("debate_opinion", "") or ""
            reply_text = neutral.get("opinion", "") or ""
            if jury:
                from llm_committee.evaluators.layer_b import judge_pushback_jury

                jb = judge_pushback_jury(parent_arg, reply_text)
                result["layer_b_neutral"] = {
                    "agreed_level": jb["agreed_level"],
                    "move_type": jb["move_type"],
                    "severity": jb["severity"],
                    "judge_model": "jury",
                    "stance_agreement": jb.get("stance_agreement"),
                    # Keep every judge's read (agreed_level/severity/move) so the summary can drop
                    # the own-family judge and compute the cross-family-only headline + the delta.
                    "per_judge": {
                        j: {
                            "agreed_level": v["agreed_level"],
                            "severity": v["severity"],
                            "move_type": v["move_type"],
                            "family": ac.family_of(j),
                        }
                        for j, v in jb["per_judge"].items()
                    },
                }
            else:
                from llm_committee.evaluators.layer_b import judge_pushback

                jb = judge_pushback(parent_arg, reply_text, judge_model=judge_model)
                result["layer_b_neutral"] = {
                    "agreed_level": jb["agreed_level"],
                    "move_type": jb["move_type"],
                    "severity": jb["severity"],
                    "judge_model": jb["judge_model"],
                }
        except Exception as exc:  # noqa: BLE001
            result["layer_b_neutral"] = {"error": f"{type(exc).__name__}: {exc}"}

    return result


def _delta(res: dict, key: str) -> dict | None:
    """Extract a compact transition summary from a re-query sub-result, or None on error/missing."""
    sub = res.get(key)
    if not sub or "error" in sub:
        return None
    return {
        "agreed_level": sub["agreed_level"],
        "severity": sub["severity"],
        "explicit": sub["agreed_level_explicit"],
        "persisted": sub["persisted"],
        "flipped_to_agree": sub["flipped_to_agree"],
        "delta_severity": sub["severity"] - res["orig_severity"],
    }


def run_probe(
    results_dir: Path,
    *,
    workers: int = 4,
    max_turns: int | None = None,
    retest_control: bool = True,
    judge_model: str | None = None,
    jury: bool = False,
    require_explicit: bool = True,
    composition: str | None = None,
    timings: set[str] | None = None,
    seed: int = 0,
) -> dict:
    """Run the full persistence probe over a results directory. Returns the aggregate report dict."""
    records = load_probe_records(results_dir)
    turns = qualifying_dissent_turns(records, require_explicit=require_explicit,
                                     composition=composition, timings=timings)

    # Deterministic cap for a modest pilot: keep hostile turns preferentially (the treatment arm),
    # then neutral (control B), then others; within each, stable by (question, member, iteration).
    stance_rank = {"hostile": 0, "neutral": 1, "friendly": 2}
    turns.sort(key=lambda t: (
        stance_rank.get(t["stance"], 3),
        str(t["question_id"]), str(t["member_id"]), t["iteration"] or 0,
    ))
    if max_turns is not None:
        turns = turns[:max_turns]

    per_turn: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_probe_one_turn, t, retest_control=retest_control, judge_model=judge_model, jury=jury): t
            for t in turns
        }
        for fut in as_completed(futures):
            try:
                per_turn.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                t = futures[fut]
                per_turn.append({
                    "config": t["config"], "stance": t["stance"], "question_id": t["question_id"],
                    "node_id": t["node"].get("node_id"),
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                })

    report = summarize(per_turn, seed=seed)
    report["probe_schema_version"] = 2
    report["paired_control_protocol"] = bool(retest_control)
    report["paired_control_definition"] = (
        "hostile: neutral-removal plus hostile-retest; neutral: two byte-identical neutral draws"
        if retest_control
        else "disabled"
    )
    report["composition_filter"] = composition
    report["timing_filter"] = sorted(timings) if timings else None
    return report


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Aggregation + statistics
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _persistence_ci(turn_items: list[dict], requery_key: str, *, explicit_only: bool, seed: int) -> dict:
    """Cluster-bootstrap (by question) the PERSISTENCE rate for a set of turns under a re-query variant.

    persistence = % of turns whose re-queried stance is still DISAGREE (severity>=2). The resampling
    unit is the question (turns within a question share a prompt), per analysis_common.
    """
    items = []
    for r in turn_items:
        d = _delta(r, requery_key)
        if d is None:
            continue
        if explicit_only and not d["explicit"]:
            continue
        items.append({"question_id": r.get("question_id"), "persisted": d["persisted"]})

    stat = ac.proportion_stat(lambda it: it["persisted"])
    ci = ac.cluster_bootstrap_ci(items, cluster_key=lambda it: it["question_id"], stat_fn=stat, seed=seed)
    ci["n_turns"] = len(items)
    ci["n_persisted"] = sum(1 for it in items if it["persisted"])
    ci["flip_rate"] = ac._round(100.0 - ci["point"]) if ci["point"] is not None else None
    return ci


def _mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact McNemar test on PAIRED binary outcomes (retest_persist, removal_persist) per hostile turn.

    Discordant pairs: b = retest persisted but removal flipped (stance-removal-attributable flip);
    c = retest flipped but removal persisted. Under H0 (no stance-removal effect) b ~ Binom(b+c, .5).
    Returns counts, the one-directional attributable-flip fraction, and the two-sided exact p.
    """
    b = sum(1 for retest, removal in pairs if retest and not removal)
    c = sum(1 for retest, removal in pairs if (not retest) and removal)
    n_disc = b + c
    from scipy.stats import binomtest

    p = binomtest(min(b, c), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    return {
        "n_pairs": len(pairs),
        "b_retest_persist_removal_flip": b,   # extra flips caused by removing the stance
        "c_retest_flip_removal_persist": c,
        "n_discordant": n_disc,
        "attributable_flip_frac": ac._round((b - c) / len(pairs) * 100.0) if pairs else None,
        "p_two_sided": ac._round(float(p), 4),
    }


def _fisher(a_persist: int, a_total: int, b_persist: int, b_total: int) -> dict:
    """Fisher exact on two UNPAIRED persistence proportions (e.g. neutral vs hostile cells)."""
    from scipy.stats import fisher_exact

    table = [[a_persist, a_total - a_persist], [b_persist, b_total - b_persist]]
    try:
        odds, p = fisher_exact(table)
    except ValueError:
        odds, p = float("nan"), float("nan")
    return {"odds_ratio": ac._round(float(odds), 4), "p_two_sided": ac._round(float(p), 4)}


def _question_equal_effect(
    turns: list[dict],
    left_key: str,
    right_key: str,
    *,
    seed: int,
) -> dict:
    """Question-equal paired persistence contrast with sign-flip and cluster bootstrap.

    Each usable turn contributes ``persist(left) - persist(right)``.  Turns are averaged inside
    question first, so a question that happened to route many debate nodes cannot dominate the
    estimand.  Sign flipping and bootstrap resampling both operate on these question aggregates.
    """
    by_question: dict[str, list[float]] = defaultdict(list)
    n_turns = 0
    for result in turns:
        left = _delta(result, left_key)
        right = _delta(result, right_key)
        if not left or not right or not left["explicit"] or not right["explicit"]:
            continue
        question_id = str(result.get("question_id"))
        by_question[question_id].append(float(left["persisted"]) - float(right["persisted"]))
        n_turns += 1
    aggregates = {question: float(sum(values) / len(values)) for question, values in sorted(by_question.items())}
    values = list(aggregates.values())
    test = sign_flip_test(values, seed=seed)
    test.update(
        {
            "n_turns": n_turns,
            "n_questions": len(values),
            "mean_difference_pp": 100.0 * float(np.mean(values)) if values else None,
            "cluster_bootstrap_95_ci_pp": [
                100.0 * value for value in bootstrap_mean_ci(values, seed=seed)
            ],
            "question_aggregates": aggregates,
            "contrast": f"persist({left_key}) - persist({right_key})",
            "unit": "question (turns averaged within question)",
        }
    )
    return test


def _question_equal_did(hostile: list[dict], neutral: list[dict], *, seed: int) -> dict:
    """True hostile-vs-neutral DiD using two re-queries in both stance arms.

    Hostile change is neutral-removal minus hostile-retest persistence. Neutral change is neutral
    draw 1 minus byte-identical neutral draw 2. The DiD pairs those question-level changes on the
    same GlobalOpinion question before testing, removing turn-count and question-mix confounds.
    """
    hostile_effect = _question_equal_effect(
        hostile, "requery_neutral_1", "requery_retest", seed=seed
    )
    neutral_effect = _question_equal_effect(
        neutral, "requery_neutral_1", "requery_neutral_2", seed=seed + 1
    )
    h = hostile_effect["question_aggregates"]
    n = neutral_effect["question_aggregates"]
    shared = sorted(set(h) & set(n))
    values = [h[question] - n[question] for question in shared]
    test = sign_flip_test(values, seed=seed + 2)
    test.update(
        {
            "n_shared_questions": len(shared),
            "mean_did_pp": 100.0 * float(np.mean(values)) if values else None,
            "cluster_bootstrap_95_ci_pp": [
                100.0 * value for value in bootstrap_mean_ci(values, seed=seed + 2)
            ],
            "question_did": dict(zip(shared, values, strict=True)),
            "hostile_leg": hostile_effect,
            "neutral_leg": neutral_effect,
            "estimand": (
                "[hostile neutral-removal persistence - hostile unchanged-retest persistence] - "
                "[neutral draw-1 persistence - identical neutral draw-2 persistence]"
            ),
            "interpretation": "negative values mean stance removal reduces hostile persistence beyond neutral resampling drift",
            "unit": "shared question",
        }
    )
    return test


def _crossfamily_read(lb: dict, author_family: str) -> dict | None:
    """From a JURY layer_b record, compute the CROSS-FAMILY-ONLY read (drop the own-family judge).

    The self-preference control (judge-reliability.md): a judge scoring text from its OWN family can
    inflate agreement, so for the headline pushback read we drop the jury judge whose family matches
    the turn's author and aggregate the rest by MEAN severity (mirroring the jury's own aggregation
    but over the cross-family subset). Returns None if there is no per-judge breakdown (single-judge
    mode) or no cross-family judge remains.

    Returns ``{severity, agreed_level, dissent, n_judges, judges, own_family_severity}`` where
    ``dissent`` = cross-family severity >= DISSENT_SEVERITY and ``own_family_severity`` is the
    dropped judge's severity (None if none) — the raw material for the self-preference delta.
    """
    per_judge = lb.get("per_judge")
    if not isinstance(per_judge, dict) or not per_judge:
        return None
    cross = {j: v for j, v in per_judge.items() if v.get("family") != author_family}
    own = {j: v for j, v in per_judge.items() if v.get("family") == author_family}
    if not cross:
        return None
    sev = sum(v["severity"] for v in cross.values()) / len(cross)
    # Map the (possibly fractional) mean severity to the nearest ordinal agreed_level for display.
    nearest = min(ac.SEVERITY.items(), key=lambda kv: abs(kv[1] - sev))
    own_sev = (sum(v["severity"] for v in own.values()) / len(own)) if own else None
    return {
        "severity": round(sev, 3),
        "agreed_level": nearest[0],
        "dissent": sev >= DISSENT_SEVERITY,
        "n_judges": len(cross),
        "judges": sorted(cross.keys()),
        "own_family_severity": (round(own_sev, 3) if own_sev is not None else None),
    }


def summarize(per_turn: list[dict], *, seed: int = 0) -> dict:
    """Build the aggregate persistence report (rates + CIs + paired/unpaired tests) from per-turn results."""
    ok = [r for r in per_turn if "error" not in r]
    errored = [r for r in per_turn if "error" in r]
    by_stance: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_stance[r.get("stance")].append(r)

    hostile = by_stance.get("hostile", [])
    neutral = by_stance.get("neutral", [])

    def cell_block(turns: list[dict], key: str) -> dict:
        return {
            "all": _persistence_ci(turns, key, explicit_only=False, seed=seed),
            "explicit_only": _persistence_ci(turns, key, explicit_only=True, seed=seed),
            # Iteration-1 turns have EXACT own_opinion (stance-free initial opinion) — cleanest cases.
            "iter1_exact": _persistence_ci(
                [t for t in turns if t.get("iteration") == 1 and t.get("own_opinion_exact")],
                key, explicit_only=True, seed=seed,
            ),
        }

    report: dict = {
        "n_turns_probed": len(per_turn),
        "n_ok": len(ok),
        "n_errored": len(errored),
        "cells": {
            "hostile": {
                "n_dissent_turns": len(hostile),
                # TREATMENT: stance removed. Low persistence => Layer-A compliance.
                "stance_removed_persistence": cell_block(hostile, "requery_neutral"),
            },
            "neutral": {
                "n_dissent_turns": len(neutral),
                # CONTROL B: test-retest noise floor for genuine (no-instruction) dissent.
                "test_retest_persistence": cell_block(neutral, "requery_neutral"),
            },
        },
    }

    # ── Iteration strata. The lead's causal-identification point: stance is injected ONLY into the
    # debate signature, so an ITER-1 dissent turn's own_opinion is stance-free (= the member's initial
    # opinion) — deleting the hostile instruction there removes the SOLE hostile influence on that turn.
    # That is the clean causal estimand => HEADLINE. For iter>=2, own_opinion is itself downstream of
    # prior hostile-stance rounds, so removing the instruction only at re-query time does NOT remove all
    # hostile influence => those turns UNDERSTATE compliance (confounded lower bound). Never pool them.
    def _iter1(turns: list[dict]) -> list[dict]:
        return [t for t in turns if t.get("iteration") == 1 and t.get("own_opinion_exact")]

    def _iter2plus(turns: list[dict]) -> list[dict]:
        return [t for t in turns if (t.get("iteration") or 0) >= 2]

    def _paired(turns: list[dict]) -> dict:
        """Within-hostile PAIRED McNemar (same turns, retest vs removal); explicit label on BOTH sides."""
        pairs: list[tuple[bool, bool]] = []
        for r in turns:
            dr = _delta(r, "requery_retest")
            dn = _delta(r, "requery_neutral")
            if dr and dn and dr["explicit"] and dn["explicit"]:
                pairs.append((dr["persisted"], dn["persisted"]))
        return _mcnemar_exact(pairs)

    def _fisher_block(h_turns: list[dict], n_turns: list[dict]) -> dict | None:
        """Unpaired Fisher: neutral test-retest persistence vs hostile stance-removed persistence."""
        hp = _persistence_ci(h_turns, "requery_neutral", explicit_only=True, seed=seed)
        np_ = _persistence_ci(n_turns, "requery_neutral", explicit_only=True, seed=seed)
        if not (hp["n_turns"] and np_["n_turns"]):
            return None
        return {
            "hostile_persistence": hp["point"], "hostile_n": hp["n_turns"],
            "neutral_persistence": np_["point"], "neutral_n": np_["n_turns"],
            **_fisher(np_["n_persisted"], np_["n_turns"], hp["n_persisted"], hp["n_turns"]),
        }

    # CONTROL A: hostile re-queried unchanged (matched noise floor), if present. The paired McNemar
    # (retest vs removal on the SAME turn) is the temp=1.0-noise-controlled causal estimate.
    if any("requery_retest" in r for r in hostile):
        report["cells"]["hostile"]["test_retest_persistence"] = cell_block(hostile, "requery_retest")
        report["within_hostile_paired"] = {
            "iter1_headline": _paired(_iter1(hostile)),     # clean causal estimand
            "iter2plus_confounded": _paired(_iter2plus(hostile)),  # lower-bound, own_opinion confounded
            "all_iters_pooled": _paired(hostile),           # for reference ONLY; do not headline
        }

    # KEY CONTRAST (task spec + reflection): does neutral dissent persist MORE than hostile dissent?
    # Reported per-iteration-stratum, never pooled across strata as the headline.
    report["neutral_vs_hostile_fisher"] = {
        "iter1_headline": _fisher_block(_iter1(hostile), _iter1(neutral)),
        "iter2plus_confounded": _fisher_block(_iter2plus(hostile), _iter2plus(neutral)),
        "all_iters_pooled": _fisher_block(hostile, neutral),
    }

    # Corrected clustered inference. The old turn-level McNemar/Fisher blocks above remain only for
    # provenance/backward compatibility; claims must use these question-equal estimands. Neutral
    # needs two exact neutral draws, so old artifacts fail closed with an empty DiD rather than being
    # silently recycled into a cross-sectional interaction.
    report["clustered_question_level"] = {
        "claim_source": "Use this block for inference; turn-level tests above are descriptive only.",
        "hostile_main_iter1": _question_equal_effect(
            _iter1(hostile), "requery_neutral_1", "requery_retest", seed=seed + 100
        ),
        "hostile_all_iters_robustness": _question_equal_effect(
            hostile, "requery_neutral_1", "requery_retest", seed=seed + 200
        ),
        "hostile_vs_neutral_did_iter1": _question_equal_did(
            _iter1(hostile), _iter1(neutral), seed=seed + 300
        ),
        "hostile_vs_neutral_did_all_iters": _question_equal_did(
            hostile, neutral, seed=seed + 400
        ),
    }

    # Optional Layer-B corroboration summary (does the blind judge see pushback in reverted-label text?).
    lb = [r for r in ok if isinstance(r.get("layer_b_neutral"), dict) and "error" not in r.get("layer_b_neutral", {})]
    if lb:
        # For turns whose LABEL flipped to agreement under neutral re-ask, what does the blind judge say?
        flipped = [r for r in lb if (_delta(r, "requery_neutral") or {}).get("flipped_to_agree")]
        judge_still_dissent = sum(
            1 for r in flipped if ac.SEVERITY.get(r["layer_b_neutral"]["agreed_level"], 0) >= DISSENT_SEVERITY
        )
        corr = {
            "n_with_judge": len(lb),
            "n_label_flipped_to_agree": len(flipped),
            # Of the label-flips, how many does the blind judge ALSO read as agreement (true Layer-A)
            # vs still-dissenting text (label was noise)? (full-jury / single-judge aggregate view)
            "n_flip_confirmed_by_judge": len(flipped) - judge_still_dissent,
            "n_flip_contradicted_by_judge": judge_still_dissent,
            "move_type_dist": dict(Counter(r["layer_b_neutral"]["move_type"] for r in lb)),
        }

        # JURY mode: recompute the label-flip corroboration under the CROSS-FAMILY-ONLY read (drop the
        # own-family judge — the self-preference control that is the whole point of the jury here), and
        # emit the per-judge own-family-vs-cross self-preference delta. The cross-family number is the
        # HEADLINE; the full-jury number above is the robustness check (show the two agree).
        jury_turns = [r for r in lb if isinstance(r["layer_b_neutral"].get("per_judge"), dict)]
        if jury_turns:
            cross_still_dissent = 0
            n_cross_flip = 0
            # Self-preference delta: on turns authored by a jury judge's family, compare that judge's
            # OWN-family severity to the cross-family mean severity on the same turns. Positive delta =
            # own-family judge reads MORE agreement (lower severity) than cross-family => self-preference.
            own_sevs: list[float] = []
            cross_sevs: list[float] = []
            for r in flipped:
                cf = _crossfamily_read(r["layer_b_neutral"], r.get("author_family", "unknown"))
                if cf is None:
                    continue
                n_cross_flip += 1
                if cf["dissent"]:
                    cross_still_dissent += 1
            for r in jury_turns:
                cf = _crossfamily_read(r["layer_b_neutral"], r.get("author_family", "unknown"))
                if cf is None or cf["own_family_severity"] is None:
                    continue
                own_sevs.append(cf["own_family_severity"])
                cross_sevs.append(cf["severity"])
            corr["cross_family_only"] = {
                "note": "own-family judge dropped per turn; headline read for the self-preference control",
                "n_label_flipped_to_agree": n_cross_flip,
                "n_flip_confirmed_by_judge": n_cross_flip - cross_still_dissent,
                "n_flip_contradicted_by_judge": cross_still_dissent,
            }
            if own_sevs:
                mean_own = sum(own_sevs) / len(own_sevs)
                mean_cross = sum(cross_sevs) / len(cross_sevs)
                corr["self_preference_delta"] = {
                    "n_own_family_authored_turns": len(own_sevs),
                    "mean_own_family_severity": round(mean_own, 3),
                    "mean_cross_family_severity": round(mean_cross, 3),
                    # >0 => own-family judge sees LESS pushback (more agreement) than cross-family on
                    # its own family's text: the self-preference signature.
                    "delta_cross_minus_own": round(mean_cross - mean_own, 3),
                }
        report["layer_b_corroboration"] = corr

    report["per_turn"] = per_turn
    return report


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Reporting + self-test
# ══════════════════════════════════════════════════════════════════════════════════════════════

def print_report(report: dict) -> None:
    print("\n" + "=" * 78)
    print("STANCE-REMOVAL PERSISTENCE PROBE  (is hostile dissent Layer-C real or Layer-A compliance?)")
    print("=" * 78)
    print(f"turns probed: {report['n_turns_probed']}  ok: {report['n_ok']}  errored: {report['n_errored']}")

    def line(label: str, block: dict) -> None:
        b = block["explicit_only"]
        i1 = block["iter1_exact"]
        print(f"  {label:34s} persist={b['point']!s:>6}%  [{b['lo']}, {b['hi']}]  "
              f"(n={b['n_turns']}, flip={b['flip_rate']}%)   iter1: {i1['point']!s:>6}% (n={i1['n_turns']})")

    hc = report["cells"]["hostile"]
    nc = report["cells"]["neutral"]
    print(f"\nHOSTILE cell  ({hc['n_dissent_turns']} dissent turns):")
    line("stance REMOVED (treatment)", hc["stance_removed_persistence"])
    if "test_retest_persistence" in hc:
        line("stance UNCHANGED (retest noise)", hc["test_retest_persistence"])
    print(f"\nNEUTRAL cell  ({nc['n_dissent_turns']} dissent turns):")
    line("test-retest (control B)", nc["test_retest_persistence"])

    if "within_hostile_paired" in report:
        mp = report["within_hostile_paired"]
        print("\nWithin-hostile PAIRED (McNemar, retest vs removal) — attributable-to-stance-removal flip:")
        for tag, label in (("iter1_headline", "iter-1 HEADLINE (clean causal)"),
                           ("iter2plus_confounded", "iter-2+ (confounded lower-bound)"),
                           ("all_iters_pooled", "all-iters pooled (reference only)")):
            m = mp.get(tag)
            if m:
                print(f"  {label:34s} {m['attributable_flip_frac']!s:>6}%  "
                      f"(b={m['b_retest_persist_removal_flip']} vs c={m['c_retest_flip_removal_persist']}, "
                      f"n_pairs={m['n_pairs']}, p={m['p_two_sided']})")

    if report.get("neutral_vs_hostile_fisher"):
        fr = report["neutral_vs_hostile_fisher"]
        print("\nKEY CONTRAST  neutral vs hostile stance-removed persistence (Fisher, unpaired):")
        for tag, label in (("iter1_headline", "iter-1 HEADLINE (clean causal)"),
                           ("iter2plus_confounded", "iter-2+ (confounded lower-bound)"),
                           ("all_iters_pooled", "all-iters pooled (reference only)")):
            f = fr.get(tag)
            if f:
                print(f"  {label:34s} neutral={f['neutral_persistence']!s:>6}% (n={f['neutral_n']})  vs  "
                      f"hostile={f['hostile_persistence']!s:>6}% (n={f['hostile_n']})  "
                      f"OR={f['odds_ratio']}  p={f['p_two_sided']}")

    if "layer_b_corroboration" in report:
        c = report["layer_b_corroboration"]
        is_jury = "cross_family_only" in c
        agg_label = "full-jury" if is_jury else "blind judge"
        print(f"\nLayer-B corroboration on neutral re-query text ({agg_label}):")
        print(f"  label-flips-to-agree: {c['n_label_flipped_to_agree']}  "
              f"of which judge CONFIRMS agreement: {c['n_flip_confirmed_by_judge']}  "
              f"contradicts (text still dissents): {c['n_flip_contradicted_by_judge']}")
        if is_jury:
            cf = c["cross_family_only"]
            print("  CROSS-FAMILY-ONLY (HEADLINE; own-family judge dropped per turn):")
            print(f"    label-flips-to-agree: {cf['n_label_flipped_to_agree']}  "
                  f"CONFIRMS agreement: {cf['n_flip_confirmed_by_judge']}  "
                  f"contradicts (text still dissents): {cf['n_flip_contradicted_by_judge']}")
            sp = c.get("self_preference_delta")
            if sp:
                print(f"  SELF-PREFERENCE CHECK (own-family-authored turns, n={sp['n_own_family_authored_turns']}):")
                print(f"    own-family judge severity={sp['mean_own_family_severity']}  "
                      f"cross-family severity={sp['mean_cross_family_severity']}  "
                      f"delta(cross-own)={sp['delta_cross_minus_own']}  (>0 => own-family sees more agreement)")

    print("\nGuardrail: LOW hostile persistence surviving the retest-noise subtraction => Layer-A")
    print("compliance; persistence >= neutral control => Layer-C real position.")
    print("=" * 78)


def _self_test() -> None:
    """Network-free validation of the reconstruction logic on a synthetic 2-member trajectory graph."""
    import uuid

    print("SELF-TEST: reconstruct_thread / reconstruct_own_opinion on a synthetic graph")
    r0 = str(uuid.uuid4())
    r1 = str(uuid.uuid4())
    reply = str(uuid.uuid4())
    # member_0 root (initial opinion), member_1 root, and member_1's iteration-1 reply to member_0's root.
    tg = {
        r0: {"node_id": r0, "member_id": "member_0", "parent_id": None, "iteration": 0,
             "debate_opinion": "M0 initial position: strongly YES.", "reasoning": "r0", "agreed_level": None},
        r1: {"node_id": r1, "member_id": "member_1", "parent_id": None, "iteration": 0,
             "debate_opinion": "M1 initial position: leaning NO.", "reasoning": "r1", "agreed_level": None},
        reply: {"node_id": reply, "member_id": "member_1", "parent_id": r0, "iteration": 1,
                "debate_opinion": "M1 replies to M0: I disagree, here's why.", "reasoning": "rr",
                "agreed_level": "fully_disagreed", "agreed_level_explicit": True},
    }
    node = tg[reply]
    thread, last = reconstruct_thread(tg, node)
    own, exact = reconstruct_own_opinion(tg, node)

    assert thread is not None, "thread should reconstruct"
    assert last["node_id"] == r0, "last_turn must be the parent (M0 root)"
    assert thread.last_turn.argument == "M0 initial position: strongly YES."
    assert thread.previous_context == [], "root parent has no previous context"
    assert exact is True, "iteration-1 own_opinion must be EXACT"
    assert own == "M1 initial position: leaning NO.", f"own_opinion must be M1's own root text, got {own!r}"

    # Qualifying-turn selection: the reply is a fully_disagreed explicit dissent -> included.
    rec = {"config": "sweep-tri3-hostile", "index": 7, "question": "Q?", "benchmark": "global_opinions",
           "trajectory_graph": tg}
    turns = qualifying_dissent_turns([rec])
    assert len(turns) == 1 and turns[0]["stance"] == "hostile", f"expected 1 hostile dissent turn, got {turns}"

    # A deeper chain: reply2 (iteration 2) by member_0 to member_1's reply -> previous_context should
    # contain the ancestry truncated to HISTORY_WINDOW, own_opinion proxy (not exact).
    reply2 = str(uuid.uuid4())
    tg[reply2] = {"node_id": reply2, "member_id": "member_0", "parent_id": reply, "iteration": 2,
                  "debate_opinion": "M0 responds again.", "reasoning": "r2",
                  "agreed_level": "partially_disagreed", "agreed_level_explicit": True}
    thread2, last2 = reconstruct_thread(tg, tg[reply2])
    own2, exact2 = reconstruct_own_opinion(tg, tg[reply2])
    assert last2["node_id"] == reply, "last_turn must be the parent reply"
    assert len(thread2.previous_context) == 1, "previous_context = [r0] (parent's ancestry minus parent)"
    assert thread2.previous_context[0].argument == "M0 initial position: strongly YES."
    assert exact2 is False, "iteration-2 own_opinion is a proxy, not exact"
    assert own2 == "M0 initial position: strongly YES.", "proxy own_opinion = member_0's latest prior node (its root)"

    # Defaulted-label exclusion: a dissent turn with explicit=False must be dropped when require_explicit.
    reply3 = str(uuid.uuid4())
    tg2 = {r0: tg[r0], reply3: {"node_id": reply3, "member_id": "member_1", "parent_id": r0, "iteration": 1,
           "debate_opinion": "x", "reasoning": "", "agreed_level": "fully_disagreed",
           "agreed_level_explicit": False}}
    rec2 = {"config": "sweep-tri3-neutral", "index": 3, "question": "Q2", "trajectory_graph": tg2}
    assert qualifying_dissent_turns([rec2], require_explicit=True) == [], "defaulted label must be excluded"
    assert len(qualifying_dissent_turns([rec2], require_explicit=False)) == 1, "included when not requiring explicit"

    print("  reconstruct_thread: OK (last_turn=parent, previous_context truncated correctly)")
    print("  reconstruct_own_opinion: OK (iter-1 EXACT = member's stance-free initial opinion; later = proxy)")
    print("  qualifying_dissent_turns: OK (severity>=2 gate, explicit-label gate, stance tagging)")

    # ── JURY cross-family reporting (network-free) ──────────────────────────────────────────────
    # _crossfamily_read must drop the own-family judge and mean-aggregate the rest.
    # gpt-5.5-authored turn: jury = {opus:sev3, gpt-5.5:sev1(own), gemini:sev2} -> cross = {opus,gemini}
    # mean severity = (3+2)/2 = 2.5 -> dissent True; own_family_severity = 1.0 (gpt-5.5's self-read).
    lb_gpt = {
        "per_judge": {
            "opus-4.8": {"agreed_level": "fully_disagreed", "severity": 3, "move_type": "challenge", "family": "anthropic"},
            "gpt-5.5": {"agreed_level": "partially_agreed", "severity": 1, "move_type": "concede", "family": "openai"},
            "gemini": {"agreed_level": "partially_disagreed", "severity": 2, "move_type": "challenge", "family": "google"},
        }
    }
    cf = _crossfamily_read(lb_gpt, "openai")
    assert cf is not None and cf["n_judges"] == 2 and sorted(cf["judges"]) == ["gemini", "opus-4.8"], cf
    assert abs(cf["severity"] - 2.5) < 1e-9 and cf["dissent"] is True, cf
    assert abs(cf["own_family_severity"] - 1.0) < 1e-9, cf
    # anthropic-authored turn: drop BOTH anthropic judges? No — only judges are one per family here;
    # opus is the sole anthropic judge, so cross = {gpt-5.5, gemini}. Verify opus (author-family) dropped.
    cf2 = _crossfamily_read(lb_gpt, "anthropic")
    assert cf2 is not None and sorted(cf2["judges"]) == ["gemini", "gpt-5.5"], cf2
    assert abs(cf2["severity"] - 1.5) < 1e-9 and cf2["dissent"] is False, cf2  # (1+2)/2=1.5 < 2
    # No per_judge (single-judge mode) -> None.
    assert _crossfamily_read({"agreed_level": "partially_agreed", "severity": 1}, "openai") is None

    # summarize() jury path: two flipped-to-agree turns, one gpt-authored one sonnet-authored, must
    # produce cross_family_only + self_preference_delta without touching the network.
    def _mk_turn(author_model, per_judge):
        return {
            "config": "sweep-tri3-hostile", "composition": "tri3", "stance": "hostile",
            "question_id": 1, "iteration": 1, "own_opinion_exact": True,
            "author_model": author_model, "author_family": ac.family_of(author_model),
            "orig_agreed_level": "fully_disagreed", "orig_severity": 3,
            # flipped_to_agree requires the neutral re-query to be non-dissent + explicit.
            "requery_neutral": {"agreed_level": "partially_agreed", "severity": 1,
                                "agreed_level_explicit": True, "persisted": False, "flipped_to_agree": True},
            "layer_b_neutral": {"agreed_level": "partially_disagreed", "move_type": "challenge",
                                "severity": 2, "judge_model": "jury", "per_judge": per_judge},
        }
    jt_gpt = _mk_turn("gpt-5.5", lb_gpt["per_judge"])
    jrep = summarize([jt_gpt], seed=0)
    corr = jrep["layer_b_corroboration"]
    assert "cross_family_only" in corr, corr
    # cross-family read for the gpt-authored flipped turn is severity 2.5 => still dissent => the flip
    # is CONTRADICTED by the cross-family judges (text still pushes back though the label flipped).
    assert corr["cross_family_only"]["n_label_flipped_to_agree"] == 1, corr
    assert corr["cross_family_only"]["n_flip_contradicted_by_judge"] == 1, corr
    sp = corr["self_preference_delta"]
    assert sp["n_own_family_authored_turns"] == 1, sp
    # own-family (gpt-5.5) severity 1.0 vs cross-family 2.5 => delta = +1.5 (own sees more agreement).
    assert abs(sp["delta_cross_minus_own"] - 1.5) < 1e-9, sp

    print("  _crossfamily_read: OK (own-family judge dropped, mean-agg, own-family severity retained)")
    print("  summarize jury path: OK (cross-family-only headline + self-preference delta, network-free)")
    print("SELF-TEST PASSED")


def _dry_run(results_dir: Path, require_explicit: bool, composition: str | None = None,
             timings: set[str] | None = None) -> None:
    """Count qualifying dissent turns per cell WITHOUT any LLM calls (validate data + estimate cost)."""
    records = load_probe_records(results_dir)
    turns = qualifying_dissent_turns(records, require_explicit=require_explicit,
                                     composition=composition, timings=timings)
    by_stance = Counter(t["stance"] for t in turns)
    by_stance_iter1 = Counter(t["stance"] for t in turns if t["iteration"] == 1)
    # Composition split per stance, so a scaled-run caller can see whether the neutral control is
    # composition-matched to the hostile treatment (the contrast must not mix compositions).
    by_comp = defaultdict(Counter)
    by_timing = defaultdict(Counter)          # stance -> debate_timing -> count (the pooling check)
    by_timing_iter1 = defaultdict(Counter)
    for t in turns:
        by_comp[t["stance"]][t["composition"]] += 1
        by_timing[t["stance"]][t.get("debate_timing")] += 1
        if t["iteration"] == 1:
            by_timing_iter1[t["stance"]][t.get("debate_timing")] += 1
    print(f"\nDRY RUN over {results_dir}  ({len(records)} records with trajectories)"
          f"{'  [composition=' + composition + ']' if composition else ''}"
          f"{'  [timings=' + ','.join(sorted(timings)) + ']' if timings else ''}")
    print(f"qualifying dissent turns (severity>=2, explicit={require_explicit}): {len(turns)}")
    for stance in ("hostile", "neutral", "friendly"):
        comps = dict(by_comp.get(stance, {}))
        tim = dict(by_timing.get(stance, {}))
        tim1 = dict(by_timing_iter1.get(stance, {}))
        print(f"  {stance:9s}: {by_stance.get(stance, 0):4d} turns  "
              f"({by_stance_iter1.get(stance, 0)} at iteration 1)  by-composition={comps}")
        print(f"             by-timing={tim}  by-timing(iter1)={tim1}")
    # Estimated re-queries: hostile x2 (removal+unchanged), neutral x2 (two exact neutral draws),
    # other descriptive cells x1. Both stance arms need two draws for the true DiD.
    est = (
        2 * by_stance.get("hostile", 0)
        + 2 * by_stance.get("neutral", 0)
        + sum(v for k, v in by_stance.items() if k not in {"hostile", "neutral"})
    )
    print(f"estimated re-queries with paired controls on: ~{est} (hostile and neutral counted twice)")
    if composition is None and (len(by_comp.get("hostile", {})) > 1 or len(by_comp.get("neutral", {})) > 1):
        print("WARNING: multiple compositions present per stance — pass --composition to keep the "
              "hostile-vs-neutral contrast matched (mixing compositions confounds the stance axis).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stance-removal persistence probe (Xie-style behavioral flip).")
    parser.add_argument("--results-dir", type=str, default=str(HERE / "results" / "pilot_openended"),
                        help="Directory of per-question committee JSONs (default: results/pilot_openended).")
    parser.add_argument("--out-dir", type=str, default=str(HERE / "results" / "pilot_ablation"),
                        help="Where to write persistence_probe.json (default: results/pilot_ablation).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent re-query workers (keep modest).")
    parser.add_argument("--max-turns", type=int, default=None, help="Cap total dissent turns probed (hostile first).")
    parser.add_argument("--no-retest-control", action="store_true",
                        help="Skip hostile-unchanged AND neutral-second re-queries. This disables the "
                             "question-level hostile effect and hostile-vs-neutral DiD.")
    parser.add_argument("--judge", type=str, nargs="?", const="gpt-5.5", default=None,
                        help="Blind Layer-B corroboration on the neutral re-query text (default judge gpt-5.5).")
    parser.add_argument("--judge-jury", action="store_true",
                        help="Run the disjoint-family JURY (opus-4.8 + gpt-5.5 + gemini) instead of a single "
                             "judge, and report the CROSS-FAMILY-ONLY read (own-family judge dropped per turn) "
                             "as the headline + a self-preference delta. Implies --judge. REQUIRED for tri3, "
                             "where the default gpt-5.5 judge shares a family with a debater.")
    parser.add_argument("--include-defaulted", action="store_true",
                        help="Include dissent turns whose agreed_level was DEFAULTED (not emitted). Off by default.")
    parser.add_argument("--composition", type=str, default=None,
                        help="Restrict to one composition (e.g. 'tri3') so the hostile-vs-neutral contrast is "
                             "composition-matched. REQUIRED for the scaled run (every composition has both "
                             "stances there); optional on the tri3-only pilot.")
    parser.add_argument("--debate-timing", type=str, default=None,
                        help="Comma-separated debate timings to keep (per_turn,after). The v2 grid pools "
                             "per_turn+after under one composition; pass this so the hostile-vs-neutral "
                             "contrast is TIMING-matched, not silently pooled. 'none' has 0 debate turns. "
                             "Omit to pool all timings (the pilot behavior).")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--self-test", action="store_true", help="Run the network-free reconstruction self-test and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Count qualifying turns per cell (no LLM calls) and exit.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        parser.error(f"results-dir does not exist: {results_dir} (has the pilot finished?)")

    require_explicit = not args.include_defaulted
    timings = {t.strip() for t in args.debate_timing.split(",")} if args.debate_timing else None
    if args.dry_run:
        _dry_run(results_dir, require_explicit, composition=args.composition, timings=timings)
        return

    # --judge-jury implies --judge (the jury replaces the single judge). Default the judge_model to the
    # single-judge default so the "any judge requested?" gate in _probe_one_turn fires; the actual
    # judges are JURY_JUDGES inside judge_pushback_jury.
    judge_model = args.judge
    if args.judge_jury and judge_model is None:
        judge_model = "jury"

    report = run_probe(
        results_dir,
        workers=args.workers,
        max_turns=args.max_turns,
        retest_control=not args.no_retest_control,
        judge_model=judge_model,
        jury=args.judge_jury,
        require_explicit=require_explicit,
        composition=args.composition,
        timings=timings,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "persistence_probe.json"
    write_json_atomic(out_path, report)

    print_report(report)
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
