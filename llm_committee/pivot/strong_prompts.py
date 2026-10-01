"""Only A/B rubric changes; retain the archived neutral C/D and tone instructions."""

from . import prompts as base
from .models import Message, canonical
from .strong_agreement import AGREEMENT, AGREEMENT_RUBRIC_TEXT

PROMPT_VERSION = "pivot-dyadic-2026-09-26-v6-strong-agreement"
REPLY_SCHEMA = base.schema(reply=base.STRING, agreement={"type": ["string", "null"], "enum": [*AGREEMENT, None]})
B_SCHEMA = base.schema(
    label={"type": "string", "enum": [*AGREEMENT, "no_position", "unjudgeable"]}, evidence=base.STRING
)
REPLY_RULE = (
    "Respond directly to the specified incoming peer contribution, using only this branch. "
    "State your substantive response and reasons in the reply field (normally 100–200 words). "
    "Also report your agreement with that peer contribution, NOT your agreement with the original question.\n"
    + AGREEMENT_RUBRIC_TEXT
    + "\nApply the rubric to your own substantive response. Use null only if you cannot report a position "
    "or determine its overall direction. Return a JSON object with reply and agreement."
)
DEBATE_OUTPUT_CONTRACT = (
    "Now respond to the incoming peer contribution. Your final answer must be exactly one "
    "valid JSON object with exactly two fields: reply and agreement. Put the complete "
    "substantive reply in the reply string and your self-reported peer-agreement label in "
    "agreement, using the definitions above. Do not output the reply as standalone text. "
    "Do not put a Markdown code fence, heading, or any other text before or after the JSON. "
    "The final answer must begin with { and end with }. Escape quotation marks and line "
    "breaks inside JSON strings. Required JSON schema:\n" + canonical(REPLY_SCHEMA)
)


def debate_messages(question, tone, initial, route, replies, node_id):
    archived = base.debate_messages(question, tone, initial, route, replies, node_id)
    return (
        Message("developer", base.BASE + "\n" + base.TONE_TEXT[tone] + "\n" + REPLY_RULE),
        *archived[1:-1],
        Message("user", DEBATE_OUTPUT_CONTRACT),
    )


def b_messages(question, peer, reply):
    return (
        Message(
            "developer",
            "Judge the reply's substantive agreement with the peer contribution, not with the survey "
            "question. Ignore politeness, assigned roles and apparent model identities. You are not given "
            "self-reported agreement or tone instructions.\n"
            + AGREEMENT_RUBRIC_TEXT
            + "\nApply the rubric to the substantive response expressed in the reply. Use no_position for "
            "no substantive stance; unjudgeable for insufficient or ambiguous evidence. Return label and "
            "a short supporting quotation/explanation as JSON.",
        ),
        *base.b_messages(question, peer, reply)[1:],
    )
