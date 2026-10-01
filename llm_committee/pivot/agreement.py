"""Versioned peer-agreement labels; shared verbatim by A and B.

The legacy vocabulary is retained only for reading the completed reference run.
There is deliberately no old-to-new label conversion: the rubric changed.
"""

LEGACY_PROMPT_VERSION = "pivot-pilot-2026-09-25-v4-output-contracts"
LEGACY_AGREEMENT = ("fully_agreed", "partially_agreed", "partially_disagreed", "fully_disagreed")

AGREEMENT_VERSION = "peer_agreement_leaning_v1"
AGREEMENT = ("fully_agree", "leaning_agree", "leaning_disagree", "fully_disagree")
AGREEMENT_DEFINITIONS = {
    "fully_agree": (
        "Endorse the peer's central position and main supporting reasoning without substantive reservations. "
        "Minor wording corrections do not count as reservations."
    ),
    "leaning_agree": (
        "On balance support the peer's central position, but have substantive reservations about the reasoning, "
        "scope, or exceptions."
    ),
    "leaning_disagree": (
        "On balance do not accept the peer's central position, but accept meaningful parts of its content "
        "or a more limited version of that position."
    ),
    "fully_disagree": (
        "Clearly reject the peer's central position and do not endorse its main argument. "
        "Agreement with an incidental detail does not count as substantive endorsement."
    ),
}
AGREEMENT_DECISION_RULE = (
    "Identify the peer's central position first. Decide whether the response overall supports or opposes that "
    "position; then distinguish substantive reservations or partial endorsement from an unqualified judgment. "
    "Do not count agreeing versus disagreeing sentences. Leaning describes the overall stance toward the peer's "
    "message, not uncertainty about how to classify it. Politeness, hostile wording, and minor factual or wording "
    "corrections do not by themselves determine the label. Do not force a direction when no stance is expressed "
    "or the overall stance cannot be determined."
)
AGREEMENT_RUBRIC_TEXT = (
    "Peer-agreement rubric (about the incoming contribution, not the original survey statement):\n"
    + "\n".join(f"{label}: {definition}" for label, definition in AGREEMENT_DEFINITIONS.items())
    + "\n"
    + AGREEMENT_DECISION_RULE
)


def agreement_manifest() -> dict:
    """New serializable copy, so run metadata cannot mutate the shared rubric."""
    return {
        "version": AGREEMENT_VERSION,
        "labels": list(AGREEMENT),
        "definitions": dict(AGREEMENT_DEFINITIONS),
        "decision_rule": AGREEMENT_DECISION_RULE,
        "shared_prompt_text": AGREEMENT_RUBRIC_TEXT,
        "layers": ["A", "B"],
        "not_judgeable": {"A": None, "B": ["no_position", "unjudgeable"]},
        "legacy_relabeling": "forbidden; a new generation/judgment protocol, not a display-name mapping",
    }
