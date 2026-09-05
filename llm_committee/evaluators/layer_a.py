"""Layer A: what does each debater *say* its agreement is?

The cheapest disagreement signal — and the most confounded one: the agreement label is emitted by
the same model, in the same call, that received the stance instruction, so an instruction to argue
mechanically moves it. Read Layer A for what it is (the self-report), and corroborate it with
Layer B (does the text push back?), Layer C (does the position survive removing the instruction?),
or Layer D (does the token-level stance distribution move?) before treating it as disagreement.

Works on a committee result's ``trajectory_graph`` — either live ``TrajectoryNode`` objects or the
serialized dict form.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

# Severity order for the four-level self-report label.
AGREED_LEVELS = ("fully_agreed", "partially_agreed", "partially_disagreed", "fully_disagreed")
SEVERITY = {label: rank for rank, label in enumerate(AGREED_LEVELS)}


def _field(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _label(node: Any) -> str | None:
    value = _field(node, "agreed_level")
    return getattr(value, "value", value)


def iter_debate_turns(trajectory_graph: dict[str, Any]) -> Iterable[Any]:
    """Yield the debate-turn nodes (iteration >= 1); roots are initial opinions with no label."""
    for node in trajectory_graph.values():
        if (_field(node, "iteration") or 0) >= 1:
            yield node


def turn_labels(trajectory_graph: dict[str, Any], explicit_only: bool = True) -> list[dict[str, Any]]:
    """Per-turn Layer A readings.

    Returns one row per debate turn with the member, iteration, self-reported ``agreed_level``,
    ``opinion_shift``, and whether the label was explicitly emitted by the model
    (``agreed_level_explicit``) rather than filled by the schema default. With ``explicit_only``
    (the default), defaulted labels are excluded so schema fallbacks never count as measurements.
    """
    rows = []
    for node in iter_debate_turns(trajectory_graph):
        explicit = bool(_field(node, "agreed_level_explicit", False))
        if explicit_only and not explicit:
            continue
        label = _label(node)
        if label is None:
            continue
        rows.append(
            {
                "node_id": _field(node, "node_id"),
                "member_id": _field(node, "member_id"),
                "iteration": _field(node, "iteration"),
                "agreed_level": label,
                "severity": SEVERITY.get(label),
                "opinion_shift": _field(node, "opinion_shift", 0),
                "agreed_level_explicit": explicit,
            }
        )
    return rows


def summarize(trajectory_graph: dict[str, Any], explicit_only: bool = True) -> dict[str, Any]:
    """Committee-level Layer A summary: label distribution, full-agreement rate, mean shift."""
    rows = turn_labels(trajectory_graph, explicit_only=explicit_only)
    counts = Counter(row["agreed_level"] for row in rows)
    n = len(rows)
    return {
        "n_turns": n,
        "label_counts": dict(counts),
        "fully_agreed_rate": (counts.get("fully_agreed", 0) / n) if n else None,
        "dissent_rate": (
            sum(1 for r in rows if r["severity"] is not None and r["severity"] >= 2) / n
            if n
            else None
        ),
        "mean_opinion_shift": (sum(r["opinion_shift"] for r in rows) / n) if n else None,
    }
