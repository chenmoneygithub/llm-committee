"""Strong/leaning peer-agreement rubric; never relabel archived fully responses."""

AGREEMENT_VERSION = "peer_agreement_strong_leaning_v2"
AGREEMENT = ("strongly_agree", "leaning_agree", "leaning_disagree", "strongly_disagree")
AGREEMENT_DEFINITIONS = {
    "strongly_agree": (
        "Clearly and substantially support the peer's central position and regard its main case as persuasive. "
        "Agreement with every detail is not required: minor reservations or disagreements about noncentral "
        "details can coexist with strong overall endorsement."
    ),
    "leaning_agree": (
        "On balance support the peer's central position, but have substantive reservations about the reasoning, "
        "scope, or exceptions that materially limit the strength of overall endorsement."
    ),
    "leaning_disagree": (
        "On balance oppose the peer's central position, but accept meaningful parts of its central case "
        "or a more limited version of that position, materially qualifying the overall opposition."
    ),
    "strongly_disagree": (
        "Clearly and substantially reject the peer's central position and find its main case unpersuasive. "
        "Disagreement with every detail is not required: accepting incidental points can coexist with "
        "strong overall opposition."
    ),
}
AGREEMENT_DECISION_RULE = (
    "Identify the peer's central position first. Decide whether the response overall supports or opposes it; "
    "then distinguish strong overall endorsement/opposition from a qualified, on-balance leaning. "
    "Strongly refers to substantive agreement or disagreement, NOT aggressive wording, politeness, rhetorical "
    "force, certainty, or confidence. Leaning refers to qualified support/opposition, not uncertainty about "
    "how to classify a response. Substantive reservations or partial endorsement that materially qualify "
    "the overall position distinguish leaning from strongly; minor or incidental details do not. "
    "Do not count agreeing versus disagreeing sentences. Do not force a direction when no stance is "
    "expressed or the overall stance cannot be determined."
)
AGREEMENT_RUBRIC_TEXT = (
    "Peer-agreement rubric (about the incoming contribution, not the original survey statement):\n"
    + "\n".join(f"{name}: {definition}" for name, definition in AGREEMENT_DEFINITIONS.items())
    + "\n"
    + AGREEMENT_DECISION_RULE
)


def agreement_manifest():
    return {
        "version": AGREEMENT_VERSION,
        "labels": list(AGREEMENT),
        "definitions": dict(AGREEMENT_DEFINITIONS),
        "decision_rule": AGREEMENT_DECISION_RULE,
        "shared_prompt_text": AGREEMENT_RUBRIC_TEXT,
        "layers": ["A", "B"],
        "not_judgeable": {"A": None, "B": ["no_position", "unjudgeable"]},
        "legacy_relabeling": "forbidden; generate and judge anew, never convert fully labels",
    }
