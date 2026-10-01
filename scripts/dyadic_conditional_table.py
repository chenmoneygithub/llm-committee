"""Label-conditioned C table for the three two-member (dyadic) settings.

Two denominators are kept separate throughout:

* Occurrence: every pairable formal reply (T2-T4; T1 answers an initial answer and
  has no preceding agreement label). T1/T2 are shared by the two tone assignments
  and counted once; T3/T4 are counted per assignment.
* Change rates: only replies that were presampled for C measurement (the
  published report rows). Option and text use their own valid denominators;
  unavailable comparisons are excluded, never counted as unchanged.

Label pairs are post-hoc descriptive groups, not randomized conditions. Intervals
bootstrap whole questions. Analysis only: reads archived records, makes no model calls.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/public-history-extension30-20260928/combined"
DOCS = ROOT / "docs"
SETTINGS = [
    ("same_model", "same-model", "Same model (Terra ×3)"),
    ("same_family", "same-family", "Same family (Luna / Terra / Sol)"),
    ("mixed_family", "mixed-family", "Mixed family (Terra / Qwen / Inkling)"),
]
LABELS = ["strongly_agree", "leaning_agree", "leaning_disagree", "strongly_disagree"]
SHORT = {"strongly_agree": "SA", "leaning_agree": "LA", "leaning_disagree": "LD", "strongly_disagree": "SD"}
PAIRS = ["AB", "BC", "CA"]
DRAWS, SEED = 10_000, 20260929


def pairable_replies(question_file: pathlib.Path):
    """Yield (question_id, node_key, previous_label, current_label) once per unique reply."""
    record = json.loads(question_file.read_text())
    qid = record["question_id"]
    branches = {a: record["branches"][a]["formal_replies"]["turn_level"] for a in ("original", "alternate")}
    for pair in PAIRS:
        for t in (2, 3, 4):
            node, parent = f"{pair}-{t}", f"{pair}-{t - 1}"
            assignments = ("original",) if t == 2 else ("original", "alternate")
            if t == 2:
                assert json.dumps(branches["original"][node], sort_keys=True) == json.dumps(
                    branches["alternate"][node], sort_keys=True
                ), f"shared T2 differs: {qid} {node}"
            for a in assignments:
                cur, prev = branches[a].get(node), branches[a].get(parent)
                if cur is None or prev is None:
                    yield qid, (a, node), None, None
                    continue
                yield qid, (a, node), prev.get("agreement"), cur.get("agreement")


def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def cluster_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray):
    """Question-cluster percentile bootstrap CI of sum(num)/sum(den).

    Returns (ci, zero_denominator_draws, method). When the observed rate is exactly 0 or 1 the
    bootstrap collapses to a zero-width interval, so a Wilson interval with n = number of
    contributing questions is used instead (conservative: each question contributes >= 1 reply).
    """
    total_n, total_d = num.sum(), den.sum()
    if total_d == 0:
        return None, 0, None
    if total_n in (0, total_d):
        nq = int((den > 0).sum())
        return wilson(nq if total_n else 0, nq), 0, "wilson_questions"
    n, d = num[idx].sum(axis=1), den[idx].sum(axis=1)
    ok = d > 0
    lo, hi = np.percentile(n[ok] / d[ok], [2.5, 97.5])
    return [float(lo), float(hi)], int(DRAWS - ok.sum()), "cluster_bootstrap"


def analyse(run_dir: str, report_slug: str, title: str, rng: np.random.Generator):
    qfiles = sorted((RUNS / run_dir / "dyadic" / "questions").glob("*.json"))
    qids = [f.stem for f in qfiles]
    qpos = {q: i for i, q in enumerate(qids)}
    nq = len(qids)
    idx = rng.integers(0, nq, size=(DRAWS, nq))

    # Occurrence among all pairable replies.
    occ = {(p, c): np.zeros(nq) for p in LABELS for c in LABELS}
    pairable = np.zeros(nq)
    missing_pairable = 0
    formal_label = {}
    for f in qfiles:
        for qid, key, prev, cur in pairable_replies(f):
            if prev is None or cur is None or prev not in LABELS or cur not in LABELS:
                missing_pairable += 1
                continue
            pairable[qpos[qid]] += 1
            occ[(prev, cur)][qpos[qid]] += 1
            formal_label[(qid,) + key] = (prev, cur)

    # Change rates among C-sampled replies.
    report = json.loads((DOCS / f"turn-tone-dyadic-{report_slug}-50-public-history.json").read_text())
    rows = report["rows"]
    zeros = lambda: np.zeros(nq)  # noqa: E731
    samp = {k: zeros() for k in occ}
    opt_valid, opt_changed, opt_excluded = ({k: zeros() for k in occ} for _ in range(3))
    txt_valid, txt_concl, txt_adj, txt_excluded = ({k: zeros() for k in occ} for _ in range(4))
    label_mismatch = 0
    for r in rows:
        k = (r["previous_peer_self_label"], r["current_self_label"])
        i = qpos[r["question_id"]]
        ref = formal_label.get((r["question_id"], "original" if r["shared_prefix_event"] else r["assignment"], r["node_id"]))
        if ref != k:
            label_mismatch += 1
        samp[k][i] += 1
        if r["C_choice_changed"] is None:
            opt_excluded[k][i] += 1
        else:
            opt_valid[k][i] += 1
            opt_changed[k][i] += bool(r["C_choice_changed"])
        if r["text_label"] in ("unchanged", "adjusted", "conclusion_changed"):
            txt_valid[k][i] += 1
            txt_concl[k][i] += r["text_label"] == "conclusion_changed"
            txt_adj[k][i] += r["text_label"] == "adjusted"
        else:
            txt_excluded[k][i] += 1

    cells = []
    for k in occ:
        if occ[k].sum() == 0 and samp[k].sum() == 0:
            continue
        share_ci, _, share_m = cluster_ci(occ[k], pairable, idx)
        opt_ci, opt_skip, opt_m = cluster_ci(opt_changed[k], opt_valid[k], idx)
        txt_ci, txt_skip, txt_m = cluster_ci(txt_concl[k], txt_valid[k], idx)
        cells.append(
            {
                "previous_peer_label": k[0],
                "current_label": k[1],
                "occurrences": int(occ[k].sum()),
                "occurrence_share": float(occ[k].sum() / pairable.sum()),
                "occurrence_share_ci": share_ci,
                "occurrence_share_ci_method": share_m,
                "occurrence_questions": int((occ[k] > 0).sum()),
                "sampled": int(samp[k].sum()),
                "sampled_questions": int((samp[k] > 0).sum()),
                "option_changed": int(opt_changed[k].sum()),
                "option_valid": int(opt_valid[k].sum()),
                "option_excluded": int(opt_excluded[k].sum()),
                "option_rate_ci": opt_ci,
                "option_rate_ci_method": opt_m,
                "option_ci_zero_denominator_draws": opt_skip,
                "text_conclusion_changed": int(txt_concl[k].sum()),
                "text_reasons_adjusted": int(txt_adj[k].sum()),
                "text_valid": int(txt_valid[k].sum()),
                "text_excluded": int(txt_excluded[k].sum()),
                "text_rate_ci": txt_ci,
                "text_rate_ci_method": txt_m,
                "text_ci_zero_denominator_draws": txt_skip,
            }
        )
    order = {l: i for i, l in enumerate(LABELS)}
    cells.sort(key=lambda c: (order[c["previous_peer_label"]], order[c["current_label"]]))
    return {
        "setting": run_dir,
        "title": title,
        "questions": nq,
        "pairable_replies": int(pairable.sum()),
        "pairable_missing_or_unlabeled": missing_pairable,
        "sampled_events": len(rows),
        "sampled_label_mismatch_vs_formal_records": label_mismatch,
        "cells": cells,
    }


def pct(x):
    return f"{100 * x:.1f}%"


def ci(c, method=None):
    if c is None:
        return "—"
    mark = "†" if method == "wilson_questions" else ""
    return f"[{100 * c[0]:.0f}, {100 * c[1]:.0f}]{mark}"


def markdown(results) -> str:
    out = [
        "# Dyadic label-conditioned C table (base data)",
        "",
        "Generated by `scripts/dyadic_conditional_table.py` from archived records; no model calls.",
        "",
        "**Scope:** the three two-member settings only. Three-member runs have no local C/D;",
        "their initial→final changes belong to E1 and are reported separately.",
        "",
        "**Two denominators, never mixed:**",
        "",
        "- *Occurrences* count every pairable formal reply (T2–T4). T1 answers an initial answer",
        "  and has no preceding agreement label, so it is not assigned one. T1/T2 are shared by the",
        "  two tone assignments and counted once; T3/T4 count once per assignment.",
        "- *Change rates* use only replies presampled for C measurement. Option: changed / valid",
        "  option comparisons. Text: main-conclusion changes / valid text comparisons. Unavailable",
        "  comparisons are listed as excluded, not counted as unchanged.",
        "",
        "Label pairs are post-hoc descriptive groups, not randomized conditions; rates are",
        "associations, not causal effects of a label. 95% intervals are percentile bootstraps",
        f"over whole questions ({DRAWS:,} draws, seed {SEED}); replies are not treated as independent.",
        "† Rate exactly 0% or 100%: the bootstrap collapses to zero width, so the interval is a",
        "  Wilson interval with n = contributing questions (conservative).",
        "Labels: SA strongly agree, LA leaning agree, LD leaning disagree, SD strongly disagree.",
        "Pair = previous peer's self-label → current member's self-label.",
        "",
    ]
    for r in results:
        out += [
            f"## {r['title']}",
            "",
            f"{r['questions']} questions · {r['pairable_replies']} pairable replies "
            f"({r['pairable_missing_or_unlabeled']} missing/unlabeled) · {r['sampled_events']} C-sampled replies · "
            f"label reconstruction mismatches: {r['sampled_label_mismatch_vs_formal_records']}",
            "",
            "| Pair | Occurrences (share, 95% CI) | Qs | Sampled (Qs) | Option changed | Option 95% CI | Text conclusion changed | Text 95% CI | Reasons adjusted | Excluded (opt/text) |",
            "|---|---|---:|---|---|---|---|---|---:|---|",
        ]
        for c in r["cells"]:
            pair = f"{SHORT[c['previous_peer_label']]}→{SHORT[c['current_label']]}"
            opt = f"{c['option_changed']}/{c['option_valid']}" + (
                f" ({pct(c['option_changed'] / c['option_valid'])})" if c["option_valid"] else ""
            )
            txt = f"{c['text_conclusion_changed']}/{c['text_valid']}" + (
                f" ({pct(c['text_conclusion_changed'] / c['text_valid'])})" if c["text_valid"] else ""
            )
            out.append(
                f"| {pair} | {c['occurrences']} ({pct(c['occurrence_share'])}, {ci(c['occurrence_share_ci'])}) "
                f"| {c['occurrence_questions']} | {c['sampled']} ({c['sampled_questions']}) | {opt} "
                f"| {ci(c['option_rate_ci'], c['option_rate_ci_method'])} "
                f"| {txt} | {ci(c['text_rate_ci'], c['text_rate_ci_method'])} | {c['text_reasons_adjusted']} "
                f"| {c['option_excluded']}/{c['text_excluded']} |"
            )
        out.append("")
    out += [
        "## Draft caption (for the paper; edit freely)",
        "",
        "> **Position change by interaction type (two-member debates).** Each row is a pair of",
        "> self-reported labels: the previous peer's agreement with the current member, and the",
        "> current member's agreement with that peer. *Occurrences* count all pairable replies",
        "> (T2–T4, shared prefixes once); change rates use the presampled subset measured for",
        "> position change, with separate option and text denominators and unavailable",
        "> comparisons excluded. Label pairs are descriptive groups, not randomized conditions.",
        "> Intervals bootstrap questions. Cells with few sampled replies are imprecise.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = [analyse(run, slug, title, rng) for run, slug, title in SETTINGS]
    stem = DOCS / "dyadic-conditional-table-2026-09-29"
    stem.with_suffix(".json").write_text(
        json.dumps({"draws": DRAWS, "seed": SEED, "settings": results}, indent=1, ensure_ascii=False) + "\n"
    )
    stem.with_suffix(".md").write_text(markdown(results))
    for r in results:
        print(
            r["setting"], "pairable", r["pairable_replies"], "missing", r["pairable_missing_or_unlabeled"],
            "sampled", r["sampled_events"], "label_mismatch", r["sampled_label_mismatch_vs_formal_records"],
        )


if __name__ == "__main__":
    main()
