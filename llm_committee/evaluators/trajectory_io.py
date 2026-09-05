"""
Shared foundation for the Task-#5 statistical-analysis pipeline (LLM-committee debate study).

Everything in the analysis wave (defaulted-label audit, Layer-B relabel, disagreement-evolution
rewrite, permutation null, Layer-C pluralism) needs the same primitives:

- parse a config name into (composition, stance, member_count)  [mirrors make_sweep_report.py]
- map a ``member_id`` -> its model, so H4-style per-MODEL breakdowns are possible (nodes only store
  ``member_id`` = ``member_0``/``member_1``/...; the model is positional in the composition)
- load per-question records and iterate (parent, reply) debate turns / per-(member,round) opinions
- cluster-bootstrap a statistic over QUESTIONS (the sampling unit is the question, not the turn —
  turns within a question are not independent), so every headline % gets a CI (scorecard §4.3)
- the option-space information-theory kit (add-α smoothing, Shannon entropy, JS divergence/distance
  with the scipy sqrt gotcha handled, 1−normalized-Wasserstein for ordinal Likert questions) exactly
  as specified in ``docs/prior_art/pluralism-globalopinions.md`` §M1–M4
- a tiny on-disk cache for judge verdicts, so a re-run never re-calls the (paid, stochastic) judge

Design rules honored:
- **Offline, deterministic, stdlib + numpy/scipy only.** No committee re-run, no network except the
  judge (which lives behind the cache and is only invoked by the Layer-B/C scripts, never here).
- **No heavy ``import configs``.** Importing configs.py pulls in dspy + make_lm (~18 s, and it can
  touch the network). The composition→models table is small and static, so we mirror it here with a
  pointer to the source of truth, exactly like ``disagreement_evolution.py`` already mirrors the
  member-count table. A drift guard (:func:`_assert_compositions_match`) can re-check against configs
  on demand.
- **``index`` is a stable question id across stances.** ``run_experiment.py:242`` builds jobs as
  ``[(config, idx) for config in configs for idx in indices]`` — the SAME example index is run under
  every config, so ``record["index"]`` identifies the same question across friendly/neutral/hostile
  and across compositions. That block structure is what the permutation null (shuffle stance within
  composition×question) and the question-level cluster bootstrap rely on.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════════════════════
# Composition / stance metadata — MIRRORED from configs.py (source of truth: _SWEEP_COMPOSITIONS,
# _PURE_COMMITTEES). Kept as a static table to avoid the heavy dspy import. Run
# _assert_compositions_match() if you suspect drift.
# ══════════════════════════════════════════════════════════════════════════════════════════════

SWEEP_COMPOSITIONS: dict[str, list[str]] = {
    "opus2gpt": ["opus-4.8", "opus-4.8", "gpt-5.5"],
    "gpt2opus": ["gpt-5.5", "gpt-5.5", "opus-4.8"],
    "tri3": ["opus-4.8", "gpt-5.5", "sonnet-4.6"],
    "opus2gpt2": ["opus-4.8", "opus-4.8", "gpt-5.5", "gpt-5.5"],
    "diverse4": ["opus-4.8", "gpt-5.5", "sonnet-4.6", "gemini"],
    "opus2songem": ["opus-4.8", "opus-4.8", "sonnet-4.6", "gemini"],
    "gpt2songem": ["gpt-5.5", "gpt-5.5", "sonnet-4.6", "gemini"],
    # legacy qwen-era names kept so stray old data still parses (qwen was dropped).
    "opus2qwengem": ["opus-4.8", "opus-4.8", "qwen", "gemini"],
    "gpt2qwengem": ["gpt-5.5", "gpt-5.5", "qwen", "gemini"],
}

PURE_COMMITTEES: dict[str, str] = {
    "pure-opus": "opus-4.8",
    "pure-sonnet": "sonnet-4.6",
    "pure-gpt5.5": "gpt-5.5",
    "pure-gemini": "gemini",
}

STANCES = ("friendly", "neutral", "hostile")

# Debate-timing modes — MIRRORED from llm_committee.committee_sync.DEBATE_TIMING_CHOICES (kept here to
# avoid the heavy dspy import that committee_sync pulls). "per_turn" is the v1 default and is NEVER
# written into a config name; "none"/"after" appear as a trailing "-<timing>" suffix (see
# configs.make_timing_config_name). Membership in this set is what disambiguates a trailing timing
# token from a stance/composition token when parsing a name.
DEBATE_TIMING_CHOICES = ("per_turn", "none", "after")
DEFAULT_DEBATE_TIMING = "per_turn"

# Model -> family, for cross-family judge selection (mirrors judge.JUDGE_MODEL_FAMILY).
MODEL_FAMILY: dict[str, str] = {
    "opus-4.8": "anthropic", "opus": "anthropic",
    "sonnet-4.6": "anthropic", "sonnet": "anthropic",
    "gpt-5.5": "openai", "gpt-5.5-pro": "openai",
    "gemini": "google", "gemini-flash": "google",
    "qwen": "alibaba", "qwen3.5": "alibaba",
}


def _assert_compositions_match() -> None:
    """Optional drift guard: re-import configs and assert our mirror is still correct.

    Not called on the hot path (configs import is slow + touches dspy). Call it once from a test or
    a --check flag if compositions may have changed upstream.
    """
    import configs  # noqa: PLC0415 (deliberately lazy)

    for name, members in configs._SWEEP_COMPOSITIONS.items():
        assert SWEEP_COMPOSITIONS.get(name) == members, f"drift in {name}: {SWEEP_COMPOSITIONS.get(name)} != {members}"
    for name, model in configs._PURE_COMMITTEES.items():
        assert PURE_COMMITTEES.get(name) == model, f"drift in pure {name}"


def _peel_timing(parts: list[str]) -> tuple[list[str], str]:
    """Split a ``-``-tokenized sweep name into ``(parts_without_timing, timing)``.

    A TRAILING token that is a member of :data:`DEBATE_TIMING_CHOICES` is the debate-timing mode — it
    can never be a stance (friendly/neutral/hostile) or a composition token, so membership is an
    unambiguous discriminator. Absent such a token the name is a v1 name with implicit ``per_turn``.
    The ``len(parts) >= 4`` guard ensures we only peel when ``sweep`` + composition + stance still
    remain underneath, so a v1 name is NEVER altered (its stance is never a timing word).
    """
    if len(parts) >= 4 and parts[-1] in DEBATE_TIMING_CHOICES:
        return parts[:-1], parts[-1]
    return parts, DEFAULT_DEBATE_TIMING


def parse_config(config: str) -> tuple[str | None, str | None, int | None]:
    """Parse a config name into ``(composition, stance, member_count)``.

    Mirrors ``make_sweep_report.parse_config``. Handles ``sweep-<composition>-<stance>`` and
    ``pure-<composition>`` (neutral stance). Composition may itself contain dashes only for legacy
    names; the current set has none, so we split on the fixed prefix/suffix positions.

    A v2 name may carry a trailing debate-timing suffix (``sweep-<comp>-<stance>-none|after``); it is
    peeled off HERE so the returned ``(composition, stance, member_count)`` is byte-identical to the
    v1 result for the same composition/stance. Use :func:`parse_timing` for the timing itself.

    Returns ``(None, None, None)`` for names that match neither shape.
    """
    if not config:
        return None, None, None
    if config.startswith("pure-"):
        # pure-<model> committees are 3-member, neutral stance (configs.py:190-197).
        return config, "neutral", 3
    if config.startswith("sweep-"):
        # Peel any trailing timing token first so composition/stance parse exactly as in v1.
        parts, _timing = _peel_timing(config.split("-"))
        # parts[0]="sweep", parts[-1]=stance, middle = composition
        if len(parts) < 3:
            return None, None, None
        stance = parts[-1]
        composition = "-".join(parts[1:-1])
        members = SWEEP_COMPOSITIONS.get(composition)
        member_count = len(members) if members else None
        return composition, stance, member_count
    return None, None, None


def parse_timing(config: str) -> str:
    """Parse the debate-timing mode encoded in a config name (default ``per_turn``).

    A SEPARATE accessor (rather than a 4th element of :func:`parse_config`) so the ~existing 3-tuple
    callers keep working unchanged. Only ``sweep-*`` names can carry a timing suffix; v1 names (no
    suffix), ``pure-*`` committees, baselines, and unknown names all resolve to
    :data:`DEFAULT_DEBATE_TIMING` (``"per_turn"``) — the v1 behavior of every record ever written.
    """
    if not config or not config.startswith("sweep-"):
        return DEFAULT_DEBATE_TIMING
    _, timing = _peel_timing(config.split("-"))
    return timing


def models_for_composition(composition: str | None) -> list[str] | None:
    """Return the ordered member models for a composition, or None if unknown.

    ``member_i`` maps to element ``i`` of this list (positional, matching
    ``committee_sync.py:456`` ``member_lm_map[f"member_{i}"] = member_lms[i]``).
    """
    if composition is None:
        return None
    if composition in SWEEP_COMPOSITIONS:
        return list(SWEEP_COMPOSITIONS[composition])
    if composition in PURE_COMMITTEES:
        return [PURE_COMMITTEES[composition]] * 3
    return None


def model_for_member(composition: str | None, member_id: str | None) -> str | None:
    """Map a ``member_id`` (e.g. ``"member_2"``) to its model in a given composition.

    Nodes only persist ``member_id``; the model is positional in the composition. Returns None if
    the composition or member index is unknown/out of range.
    """
    models = models_for_composition(composition)
    if models is None or not member_id:
        return None
    try:
        idx = int(str(member_id).split("_")[-1])
    except (ValueError, IndexError):
        return None
    return models[idx] if 0 <= idx < len(models) else None


def family_of(model: str | None) -> str:
    """Model family ("anthropic"/"openai"/"google"/...) for cross-family reasoning."""
    if not model:
        return "unknown"
    return MODEL_FAMILY.get(model, MODEL_FAMILY.get(model.split("/")[-1], "unknown"))


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Record loading + trajectory iteration
# ══════════════════════════════════════════════════════════════════════════════════════════════

SKIP_FILES = {
    "summary.json", "divergence.json", "disagreement_evolution.json",
    "disagreement_evolution_v2.json", "option_pluralism.json", "layer_c_pluralism.json",
    "layer_b_relabel.json", "defaulted_label_audit.json", "permutation_null.json",
    "collapse_by_question.json", "fully_agreed_null.json", "manifest.json",
}

# agreed_level severity ordering, shared with judge.py + disagreement_evolution.py.
SEVERITY: dict[str, int] = {
    "fully_agreed": 0, "partially_agreed": 1, "partially_disagreed": 2, "fully_disagreed": 3,
}
AGREED_LEVELS = ("fully_agreed", "partially_agreed", "partially_disagreed", "fully_disagreed")


def load_records(results_dir: Path, *, require_trajectory: bool = True) -> list[dict]:
    """Load per-question JSON records from a results dir, skipping analyzer outputs and errors.

    Args:
        results_dir: Directory of per-question ``{config}__{index}.json`` files.
        require_trajectory: If True (default), drop records without a ``trajectory_graph``.

    Returns:
        List of record dicts with ``_file`` set to the source filename. ERROR-grade records are
        dropped. Parse errors are skipped silently (the file may be mid-write).
    """
    records = []
    for fp in sorted(Path(results_dir).glob("*.json")):
        if fp.name in SKIP_FILES:
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("grade") == "ERROR":
            continue
        if require_trajectory and not d.get("trajectory_graph"):
            continue
        d["_file"] = fp.name
        records.append(d)
    return records


def record_cell(record: dict) -> dict:
    """Extract the (composition, stance, member_count, debate_timing, question_id) cell coordinates.

    ``debate_timing`` prefers the value persisted on the record itself (written by run_one.py, the
    belt-and-suspenders source of truth) and falls back to parsing it out of the config name, so both
    v2 records (which carry the field) and any name-only records resolve the timing consistently.
    """
    composition, stance, member_count = parse_config(record.get("config", ""))
    debate_timing = record.get("debate_timing") or parse_timing(record.get("config", ""))
    return {
        "config": record.get("config"),
        "composition": composition,
        "stance": stance,
        "member_count": member_count,
        "debate_timing": debate_timing,
        # index is the stable cross-stance question id (run_experiment.py:242). Fall back to the
        # question text hash if index is somehow absent.
        "question_id": record.get("index") if record.get("index") is not None
        else _text_key(record.get("question", "")),
        "question": record.get("question"),
    }


def iter_debate_nodes(trajectory_graph: dict) -> Iterator[dict]:
    """Yield every debate-turn node (parent_id set, agreed_level present). Skips root opinions."""
    for node in trajectory_graph.values():
        if node.get("parent_id") is None:
            continue
        yield node


def iter_parent_reply_pairs(trajectory_graph: dict) -> Iterator[dict]:
    """Yield ``(parent_argument, reply, self_reported_agreed_level, ...)`` for each debate turn.

    A reply node's ``agreed_level`` is its agreement with its PARENT's ``debate_opinion`` — so
    ``(parent.debate_opinion, reply.debate_opinion)`` is exactly the blind Layer-B judge input, and
    the reply's ``agreed_level`` is the Layer-A self-report we compare against. Mirrors
    ``judge_calibration._iter_parent_reply_pairs`` but also carries the parent identity (for the
    per-(member,target) disagreement-evolution rewrite) and the ``agreed_level_explicit`` flag.
    """
    for node in trajectory_graph.values():
        pid = node.get("parent_id")
        if not pid or pid not in trajectory_graph:
            continue
        parent = trajectory_graph[pid]
        yield {
            "node_id": node.get("node_id"),
            "parent_id": pid,
            "parent_argument": parent.get("debate_opinion", "") or "",
            "reply": node.get("debate_opinion", "") or "",
            "self_reported_agreed_level": node.get("agreed_level"),
            "agreed_level_explicit": bool(node.get("agreed_level_explicit", False)),
            "reply_member": node.get("member_id"),
            "parent_member": parent.get("member_id"),
            "iteration": node.get("iteration"),
        }


def opinions_by_member_round(trajectory_graph: dict) -> dict[str, dict[int, str]]:
    """``{member_id: {iteration: opinion_text}}``; if a member has several nodes in one iteration,
    keep the LONGEST (most substantive) — the ``divergence.py`` convention, reused so Layer-C uses
    the same per-round representative opinion as the embedding analysis.
    """
    by_member: dict[str, dict[int, str]] = defaultdict(dict)
    for node in trajectory_graph.values():
        mid, it = node.get("member_id"), node.get("iteration")
        text = node.get("debate_opinion") or ""
        if mid is None or it is None:
            continue
        if it not in by_member[mid] or len(text) > len(by_member[mid][it]):
            by_member[mid][it] = text
    return by_member


def _text_key(text: str) -> str:
    """Stable short hash of a text (fallback question id / cache key)."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Cluster bootstrap — resample QUESTIONS, not turns (turns within a question are not independent)
# ══════════════════════════════════════════════════════════════════════════════════════════════

def cluster_bootstrap_ci(
    items: list[Any],
    cluster_key: Callable[[Any], Any],
    stat_fn: Callable[[list[Any]], float | None],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Cluster (question-level) bootstrap CI for an arbitrary statistic.

    The unit of resampling is the CLUSTER (question), because members/turns within one question
    share a prompt and are correlated; resampling turns i.i.d. would understate the CI. We resample
    clusters with replacement, pool their items, and recompute ``stat_fn``.

    Args:
        items: Flat list of items (turns, member-records, per-question aggregates, ...).
        cluster_key: Maps an item to its cluster id (usually the question id).
        stat_fn: Maps a list of items to the scalar statistic (returns None on a degenerate sample,
            which is then skipped).
        n_boot: Bootstrap resamples.
        seed: RNG seed.
        ci: Percentile CI bounds.

    Returns:
        ``{point, lo, hi, n_clusters, n_items, n_valid_boot}``. ``point`` is ``stat_fn(items)`` on
        the observed data; ``lo``/``hi`` are the percentile CI (NaN if too degenerate).
    """
    point = stat_fn(items)
    clusters: dict[Any, list[Any]] = defaultdict(list)
    for it in items:
        clusters[cluster_key(it)].append(it)
    cluster_ids = list(clusters.keys())
    n_clusters = len(cluster_ids)

    result = {
        "point": _round(point), "lo": float("nan"), "hi": float("nan"),
        "n_clusters": n_clusters, "n_items": len(items), "n_valid_boot": 0,
    }
    if n_clusters < 2 or point is None:
        return result

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    idx_space = np.arange(n_clusters)
    for _ in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        sample: list[Any] = []
        for k in draw:
            sample.extend(clusters[cluster_ids[idx_space[k]]])
        s = stat_fn(sample)
        if s is not None and not (isinstance(s, float) and math.isnan(s)):
            stats.append(float(s))
    if stats:
        result["lo"] = _round(float(np.percentile(stats, ci[0])))
        result["hi"] = _round(float(np.percentile(stats, ci[1])))
        result["n_valid_boot"] = len(stats)
    return result


def proportion_stat(numerator_key: Callable[[Any], bool]) -> Callable[[list[Any]], float | None]:
    """Build a stat_fn computing the % of items satisfying ``numerator_key`` (None on empty)."""
    def _stat(items: list[Any]) -> float | None:
        if not items:
            return None
        return 100.0 * sum(1 for it in items if numerator_key(it)) / len(items)
    return _stat


def cluster_bootstrap_diff_ci(
    items: list[Any],
    cluster_key: Callable[[Any], Any],
    group_fn: Callable[[Any], Any],
    group_a: Any,
    group_b: Any,
    stat_fn: Callable[[list[Any]], float | None],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Cluster-bootstrap CI for the DIFFERENCE stat_fn(group_a) − stat_fn(group_b).

    Reflection req #2 (methodological): a headline stance gap (e.g. hostile − friendly fully_agreed)
    must ship with a DIRECT difference-CI, not an eyeballed non-overlap of the two marginal CIs
    (marginal non-overlap is a conservative, under-powered test). We resample the SHARED clusters
    (questions appear in both stance cells at the same index), pool each drawn cluster's items, split
    by group, and recompute the difference — a paired-by-question bootstrap that keeps the two arms
    correlated through the same resampled question set.

    Args:
        items: Flat item list spanning both groups.
        cluster_key: Item → cluster id (question id).
        group_fn: Item → group label; only ``group_a``/``group_b`` items are used.
        group_a, group_b: The two group labels; the reported diff is A − B.
        stat_fn: List → scalar (e.g. ``proportion_stat``); applied within each group per resample.

    Returns:
        ``{point, lo, hi, n_clusters, point_a, point_b, n_dropped_single_arm}`` — ``point`` =
        stat(A) − stat(B) over the SHARED clusters; ``lo``/``hi`` the percentile CI of the difference
        (NaN if too degenerate). ``n_clusters`` counts only shared (both-arm) clusters;
        ``n_dropped_single_arm`` counts clusters excluded for carrying just one arm.
    """
    # Group items by cluster, keeping each arm separate. The paired design (req #2) compares the two
    # arms ON THE SAME QUESTIONS, so a cluster carrying only one arm (e.g. a question that errored out
    # in the neutral cell but survived in friendly — see the gpt-5.5 adapter drop) has no within-
    # question difference and is EXCLUDED. Including it would leak an unpaired marginal shift into the
    # paired estimate. For a fully-paired contrast (all questions in both arms) this drops nothing; it
    # only bites on ragged missingness (scaled run, adapter errors).
    by_cluster: dict[Any, dict[Any, list[Any]]] = defaultdict(lambda: {group_a: [], group_b: []})
    for it in items:
        g = group_fn(it)
        if g in (group_a, group_b):
            by_cluster[cluster_key(it)][g].append(it)
    shared_ids = [cid for cid, arms in by_cluster.items() if arms[group_a] and arms[group_b]]
    n_dropped_single_arm = len(by_cluster) - len(shared_ids)
    n_clusters = len(shared_ids)

    # Point estimate over the SHARED clusters only, so ``point == point_a − point_b`` exactly and both
    # marginals describe the same paired question set the CI is built on.
    a_items = [it for cid in shared_ids for it in by_cluster[cid][group_a]]
    b_items = [it for cid in shared_ids for it in by_cluster[cid][group_b]]
    pa, pb = stat_fn(a_items), stat_fn(b_items)
    point = (pa - pb) if (pa is not None and pb is not None) else None

    result = {
        "point": _round(point), "lo": float("nan"), "hi": float("nan"),
        "n_clusters": n_clusters, "point_a": _round(pa), "point_b": _round(pb),
        "n_dropped_single_arm": n_dropped_single_arm,
    }
    if n_clusters < 2 or point is None:
        return result

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        sa: list[Any] = []
        sb: list[Any] = []
        for k in draw:
            arms = by_cluster[shared_ids[k]]
            sa.extend(arms[group_a])
            sb.extend(arms[group_b])
        va, vb = stat_fn(sa), stat_fn(sb)
        if va is not None and vb is not None:
            diffs.append(float(va - vb))
    if diffs:
        result["lo"] = _round(float(np.percentile(diffs, ci[0])))
        result["hi"] = _round(float(np.percentile(diffs, ci[1])))
        result["n_valid_boot"] = len(diffs)
    return result


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Option-space information theory (pluralism §M1–M4). log base 2 (bits); 0·log0 ≡ 0.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def smooth(mass: Iterable[float], alpha: float = 0.5) -> np.ndarray:
    """add-α (Jeffreys α=0.5) smoothing → probability vector summing to 1 (pluralism §M3).

    Apply BEFORE entropy/JSD; smooth the human vector the SAME way before any JSD.
    """
    m = np.asarray(list(mass), dtype=float)
    K = len(m)
    if K == 0:
        return m
    p = m + alpha
    total = p.sum()
    return p / total if total > 0 else np.full(K, 1.0 / K)


def entropy_bits(probs: Iterable[float]) -> float:
    """Shannon entropy in bits (pluralism §M1). Assumes a normalized distribution."""
    p = np.asarray(list(probs), dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum()) + 0.0  # +0.0 normalizes -0.0


def entropy_norm(probs: Iterable[float], k_used: int | None = None) -> float:
    """Normalized entropy H / log2(K_used) ∈ [0,1] so different-K questions compare (§M1)."""
    p = list(probs)
    k = k_used if k_used is not None else len(p)
    if k <= 1:
        return 0.0
    return entropy_bits(p) / math.log2(k)


def js_distance(P: Iterable[float], Q: Iterable[float]) -> float:
    """Jensen-Shannon DISTANCE (a true metric ∈ [0,1]); Durmus' ``Sim = 1 − this`` (§M2).

    NOTE the scipy gotcha: ``scipy.spatial.distance.jensenshannon`` returns the DISTANCE (sqrt),
    not the divergence. This wraps it and labels it unambiguously.
    """
    from scipy.spatial.distance import jensenshannon
    p = np.asarray(list(P), dtype=float)
    q = np.asarray(list(Q), dtype=float)
    d = jensenshannon(p, q, base=2)
    return float(d) if not math.isnan(d) else 0.0


def js_divergence(P: Iterable[float], Q: Iterable[float]) -> float:
    """Jensen-Shannon DIVERGENCE = JS_distance² ∈ [0,1] base-2 (½KL+½KL) (§M2)."""
    return js_distance(P, Q) ** 2


def normalized_wasserstein_similarity(P: Iterable[float], Q: Iterable[float]) -> float:
    """1 − normalized Wasserstein-1 on ordinal support {1..K} (Santurkar; pluralism §M4).

    Charges by DISTANCE moved on the Likert scale (unlike nominal JSD/entropy). Caller must have
    already excluded the refusal option and renormalized P, Q over the ordinal options.
    """
    from scipy.stats import wasserstein_distance
    p = np.asarray(list(P), dtype=float)
    q = np.asarray(list(Q), dtype=float)
    K = len(p)
    if K < 2:
        return 1.0
    support = np.arange(1, K + 1)
    w = wasserstein_distance(support, support, p, q)
    return float(1.0 - w / (K - 1))


# Ordinal (Likert) option-set detection for the Wasserstein robustness subset (§M4). Heuristic:
# option texts contain scale cues in a monotone order. Kept conservative — only clearly-ordinal
# sets qualify; everything else is treated as nominal (JSD only).
_ORDINAL_CUE_SEQUENCES = [
    ["strongly agree", "agree", "disagree", "strongly disagree"],
    ["a great deal", "a fair amount", "not too much", "not at all"],
    ["too much", "about right", "too little"],
    ["very good", "somewhat good", "somewhat bad", "very bad"],
    ["very well", "somewhat well", "not too well", "not at all well"],
    ["much better", "somewhat better", "about the same", "somewhat worse", "much worse"],
    ["very important", "somewhat important", "not too important", "not at all important"],
    ["completely agree", "mostly agree", "mostly disagree", "completely disagree"],
    ["a lot", "some", "not much", "none at all"],
    ["increase", "remain the same", "reduce"],
]


def ordinal_scale_order(options: list[str]) -> list[int] | None:
    """If ``options`` form a known ordinal (Likert) scale, return the index order along the scale.

    Returns a permutation of option indices in scale order (excluding any DK/Refused), or None if
    the set isn't recognizably ordinal. Matching is on lowercased substring membership so minor
    wording differences ("Disapprove"/"Strongly disapprove") still line up. Conservative by design:
    the Wasserstein check only runs where ordering is unambiguous (pluralism §4c-1, §M4).
    """
    if not options:
        return None
    low = [str(o).strip().lower() for o in options]
    # Identify the DK/Refused-type option to exclude from the ordinal support.
    def _is_dk(o: str) -> bool:
        return any(t in o for t in ("dk/refused", "don't know", "dont know", "refused", "no answer", "decline"))

    for seq in _ORDINAL_CUE_SEQUENCES:
        order: list[int] = []
        used: set[int] = set()
        ok = True
        for cue in seq:
            match = next((i for i, o in enumerate(low) if i not in used and cue in o), None)
            if match is None:
                ok = False
                break
            order.append(match)
            used.add(match)
        # All non-DK options must be covered by the scale (allow a trailing DK/Refused).
        remaining = [i for i in range(len(low)) if i not in used and not _is_dk(low[i])]
        if ok and not remaining and len(order) >= 3:
            return order
    return None


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Judge-verdict cache — JSONL, keyed by a hash of (task, judge_model, inputs). The judge is paid +
# stochastic (temperature=1.0), so we memoize verdicts; a re-run reuses them unless --refresh.
# ══════════════════════════════════════════════════════════════════════════════════════════════

class JudgeCache:
    """Append-only JSONL cache of judge verdicts, keyed by a hash of the call signature.

    Not thread-safe for concurrent writers; the analysis scripts call it single-threaded (or guard
    their own pool). Load once, look up by key, and ``put`` new verdicts; ``flush`` appends the new
    entries to disk.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._store: dict[str, dict] = {}
        self._dirty: list[tuple[str, dict]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._store[row["k"]] = row["v"]
                except (json.JSONDecodeError, KeyError):
                    continue

    @staticmethod
    def key(task: str, judge_model: str, *parts: str) -> str:
        h = hashlib.sha1()
        h.update(task.encode("utf-8"))
        h.update(b"\x00")
        h.update((judge_model or "").encode("utf-8"))
        for p in parts:
            h.update(b"\x00")
            h.update((p or "").encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> dict | None:
        return self._store.get(key)

    def put(self, key: str, value: dict) -> None:
        self._store[key] = value
        self._dirty.append((key, value))

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for k, v in self._dirty:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
        self._dirty.clear()


# ── misc ─────────────────────────────────────────────────────────────────────────────────────

def _round(x: float | None, nd: int = 3) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    return round(float(x), nd)


def counter_to_dist(counter: dict[str, int], keys: tuple[str, ...] = AGREED_LEVELS) -> dict[str, float]:
    """Normalize a label-count dict into a percentage distribution over ``keys``."""
    total = sum(counter.get(k, 0) for k in keys)
    if total == 0:
        return {k: 0.0 for k in keys}
    return {k: round(100.0 * counter.get(k, 0) / total, 1) for k in keys}


def write_json_atomic(path, document):
    """Write a JSON artifact atomically so a resume never accepts a truncated file."""
    import json as _json
    from pathlib import Path as _Path
    destination = _Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(
        _json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    )
    temporary.replace(destination)
