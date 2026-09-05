"""Predeclared paired statistics for fresh accuracy and clustered persistence."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as student_t


def normalize_answer(value: object) -> str:
    """Deterministic Unicode/case/punctuation normalization for frozen gold aliases."""
    # Accent folding treats canonically equivalent spellings (e.g. Montreal/Montréal) as aliases
    # while the archived raw answer remains available for audit.
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in decomposed if unicodedata.category(char) != "Mn").casefold()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\b(?:the|a|an)\b", " ", text)
    text = re.sub(r"[^\w.%+\-/]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*(.+)", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    """Use the last frozen FINAL ANSWER marker, falling back to the full response."""
    matches = _FINAL_ANSWER_RE.findall(str(text or ""))
    if not matches:
        return str(text or "").strip()
    return matches[-1].strip().splitlines()[0].strip()


def grade_aliases(prediction: str, aliases: Iterable[str]) -> dict:
    """Strict primary grade plus a transparent containment sensitivity grade.

    The primary grade requires the extracted final answer to equal a normalized official alias.
    ``relaxed_contains`` is archived only as a sensitivity for verbose model outputs and never
    silently replaces the primary grade.
    """
    final = extract_final_answer(prediction)
    normalized_prediction = normalize_answer(final)
    normalized_aliases = sorted({normalize_answer(alias) for alias in aliases if str(alias).strip()})
    strict = normalized_prediction in normalized_aliases
    padded = f" {normalize_answer(prediction)} "
    contains = any(alias and f" {alias} " in padded for alias in normalized_aliases)
    return {
        "correct": bool(strict),
        "relaxed_contains": bool(strict or contains),
        "extracted_final_answer": final,
        "normalized_final_answer": normalized_prediction,
        "normalized_aliases": normalized_aliases,
        "grader": "deterministic_alias_exact_v1",
    }


def paired_counts(a: Iterable[bool], b: Iterable[bool]) -> dict:
    pairs = [(bool(x), bool(y)) for x, y in zip(a, b, strict=True)]
    return {
        "both_correct": sum(x and y for x, y in pairs),
        "a_only": sum(x and not y for x, y in pairs),
        "b_only": sum((not x) and y for x, y in pairs),
        "neither": sum((not x) and (not y) for x, y in pairs),
        "n": len(pairs),
    }


def exact_mcnemar(a: Iterable[bool], b: Iterable[bool]) -> dict:
    counts = paired_counts(a, b)
    discordant = counts["a_only"] + counts["b_only"]
    p = binomtest(counts["a_only"], discordant, 0.5).pvalue if discordant else 1.0
    return {**counts, "n_discordant": discordant, "p_two_sided": float(p)}


def paired_risk_difference(
    a: Iterable[bool],
    b: Iterable[bool],
    *,
    confidence: float = 0.95,
) -> dict:
    """Paired risk difference with the preregistered paired-Wald/t interval."""
    values = np.asarray([float(x) - float(y) for x, y in zip(a, b, strict=True)], dtype=float)
    if values.size < 2:
        raise ValueError("paired risk difference requires at least two complete pairs")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=values.size - 1))
    lo, hi = mean - critical * se, mean + critical * se
    return {
        "n": int(values.size),
        "risk_difference": mean,
        "risk_difference_pp": 100.0 * mean,
        "se": se,
        "confidence": confidence,
        "ci": [float(lo), float(hi)],
        "ci_pp": [100.0 * float(lo), 100.0 * float(hi)],
        "method": "paired binary-difference mean; Student-t interval",
    }


def paired_tost(a: Iterable[bool], b: Iterable[bool], *, margin: float = 0.075) -> dict:
    """TOST for a paired accuracy difference using the frozen ±7.5pp margin."""
    values = np.asarray([float(x) - float(y) for x, y in zip(a, b, strict=True)], dtype=float)
    if values.size < 2:
        raise ValueError("TOST requires at least two complete pairs")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    df = values.size - 1
    if se == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        ci90 = [mean, mean]
    else:
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = float(student_t.sf(t_lower, df=df))
        p_upper = float(student_t.cdf(t_upper, df=df))
        critical = float(student_t.ppf(0.95, df=df))
        ci90 = [mean - critical * se, mean + critical * se]
    equivalent = ci90[0] > -margin and ci90[1] < margin
    return {
        "n": int(values.size),
        "mean_difference": mean,
        "mean_difference_pp": 100.0 * mean,
        "margin": margin,
        "margin_pp": 100.0 * margin,
        "ci90": [float(ci90[0]), float(ci90[1])],
        "ci90_pp": [100.0 * float(ci90[0]), 100.0 * float(ci90[1])],
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "equivalent": bool(equivalent),
        "reporting_status": "equivalent" if equivalent else "inconclusive",
        "method": "paired binary-difference TOST; Student-t 90% CI",
    }


def simulate_tost_power(
    pilot_counts: dict,
    sample_sizes: Iterable[int],
    *,
    margin: float = 0.075,
    simulations: int = 20_000,
    seed: int = 20260821,
) -> dict:
    """Estimate TOST power from budget-pilot discordance under a zero-difference design point.

    The pilot supplies the paired variance through its *total discordance rate*.  Its 24-item
    directional imbalance is too noisy to treat as the population effect, so the two discordant
    cells are symmetrized for power (A-only = B-only = total-discordance/2). This is a prospective
    equivalence calculation, not an outcome extrapolation.
    """
    keys = ("both_correct", "a_only", "b_only", "neither")
    counts = np.asarray([pilot_counts.get(key, 0) for key in keys], dtype=float)
    if counts.sum() <= 0:
        raise ValueError("pilot discordance counts are empty")
    raw_probabilities = counts / counts.sum()
    discordance_rate = float(raw_probabilities[1] + raw_probabilities[2])
    probabilities = raw_probabilities.copy()
    probabilities[1] = discordance_rate / 2.0
    probabilities[2] = discordance_rate / 2.0
    rng = np.random.default_rng(seed)
    power: dict[str, float] = {}
    sizes = [int(n) for n in sample_sizes]
    for n in sizes:
        draws = rng.multinomial(n, probabilities, size=simulations)
        means = (draws[:, 1] - draws[:, 2]) / n
        # For paired differences X in {-1,0,+1}, sum(X^2)=n_discordant. Compute the
        # sample-variance identity directly instead of materializing 20k variable-length arrays.
        sum_squares = draws[:, 1] + draws[:, 2]
        variances = (sum_squares - n * means**2) / (n - 1) if n > 1 else np.full(simulations, np.inf)
        standard_errors = np.sqrt(np.maximum(variances, 0.0) / n)
        critical = float(student_t.ppf(0.95, df=n - 1))
        equivalent = (means - critical * standard_errors > -margin) & (means + critical * standard_errors < margin)
        power[str(n)] = float(np.mean(equivalent))
    eligible = [n for n in sizes if power[str(n)] >= 0.80]
    return {
        "source": "budget_pilot_paired_discordance",
        "pilot_counts": {key: int(pilot_counts.get(key, 0)) for key in keys},
        "raw_pilot_cell_probabilities": dict(zip(keys, raw_probabilities.tolist(), strict=True)),
        "power_cell_probabilities": dict(zip(keys, probabilities.tolist(), strict=True)),
        "pilot_discordance_rate": discordance_rate,
        "design_point": "zero paired accuracy difference; discordant directions symmetrized",
        "margin": margin,
        "simulations": simulations,
        "seed": seed,
        "power_by_n": power,
        "selected_n": min(eligible) if eligible else None,
        "status": "PASS" if eligible else "FAIL",
    }


def sign_flip_test(
    values: Iterable[float],
    *,
    exact_max_units: int = 22,
    permutations: int = 100_000,
    seed: int = 20260821,
) -> dict:
    """Two-sided randomization test on independent question-level contrasts.

    Enumerating all signs is exact for small samples.  Larger samples use a deterministic Monte
    Carlo draw and the Phipson--Smyth ``(+1)/(B+1)`` correction, so a finite simulation can never
    report an impossible zero p-value.  Zeros are retained: under the sharp null their sign is
    immaterial, and retaining them keeps the declared question count transparent.
    """
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {
            "n_units": 0,
            "mean_contrast": None,
            "p_two_sided": None,
            "method": "unavailable_empty",
            "seed": seed,
        }
    observed = abs(float(vals.mean()))
    tolerance = 1e-12
    if len(vals) <= exact_max_units:
        extreme = 0
        total = 1 << len(vals)
        for mask in range(total):
            signs = np.fromiter(
                (1.0 if (mask >> idx) & 1 else -1.0 for idx in range(len(vals))),
                dtype=float,
                count=len(vals),
            )
            extreme += abs(float(np.mean(vals * signs))) >= observed - tolerance
        p_value = extreme / total
        method = "exact_enumerated_sign_flip"
        draws = total
    else:
        if permutations <= 0:
            raise ValueError("permutations must be positive")
        rng = np.random.default_rng(seed)
        extreme = 0
        # Chunking avoids allocating permutations x questions for the full 100k draw.
        remaining = int(permutations)
        while remaining:
            size = min(10_000, remaining)
            signs = rng.integers(0, 2, size=(size, len(vals)), dtype=np.int8) * 2 - 1
            candidates = np.abs((signs * vals).mean(axis=1))
            extreme += int(np.count_nonzero(candidates >= observed - tolerance))
            remaining -= size
        p_value = (extreme + 1) / (permutations + 1)
        method = "deterministic_monte_carlo_sign_flip"
        draws = int(permutations)
    return {
        "n_units": int(len(vals)),
        "mean_contrast": float(vals.mean()),
        "p_two_sided": float(p_value),
        "method": method,
        "draws": draws,
        "seed": seed,
    }


def exact_sign_flip(values: Iterable[float]) -> float:
    """Compatibility wrapper returning only the two-sided sign-flip p-value.

    The old helper raised above 22 units, which made it unusable for the 50-question E4 and
    persistence analyses.  It now delegates to :func:`sign_flip_test` and therefore remains exact
    when feasible and deterministic Monte Carlo otherwise.
    """
    result = sign_flip_test(values)
    p_value = result["p_two_sided"]
    return float("nan") if p_value is None else float(p_value)


def question_aggregate(rows: Iterable[dict], value_key: str = "contrast") -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(float(row[value_key]))
    return {question_id: float(np.mean(values)) for question_id, values in sorted(grouped.items())}


def bootstrap_mean_ci(
    values: Iterable[float], *, confidence: float = 0.95, seed: int = 20260821, n_boot: int = 20_000
) -> list[float]:
    vals = np.asarray(list(values), dtype=float)
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(x) for x in np.quantile(samples, [alpha, 1.0 - alpha])]


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(p_value)))
        adjusted[name] = running
    return adjusted
