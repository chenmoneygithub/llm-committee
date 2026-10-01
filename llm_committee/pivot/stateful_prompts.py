"""Integrated debate/position protocol; archived prompt builders stay unchanged."""

from __future__ import annotations

from . import prompts
from .models import Message, canonical
from .strong_agreement import AGREEMENT, AGREEMENT_RUBRIC_TEXT

PROMPT_VERSION = "stateful-debate-2026-09-27-v1"
PUBLIC_HISTORY_PROMPT_VERSION = "public-history-debate-2026-09-27-v1"


def position_schema(question):
    return prompts.schema(
        choice={"type": ["string", "null"], "enum": [*question.labels, None]},
        position=prompts.STRING,
    )


def reply_schema(question):
    return prompts.schema(
        reply=prompts.STRING,
        agreement={"type": ["string", "null"], "enum": [*AGREEMENT, None]},
        **position_schema(question)["properties"],
    )


def initial_context(question, member):
    return (
        Message("developer", prompts.BASE),
        Message(
            "user",
            canonical(
                {
                    "question": question.text,
                    "options": dict(zip(question.labels, question.options, strict=True)),
                    "your_member_id": member,
                }
            ),
        ),
    )


def initial_messages(question, member):
    return (
        *initial_context(question, member),
        Message(
            "user",
            "Give your independent initial position on the original question. No peer answers are available. "
            "Return one JSON object with choice (an original option letter, or null if none fits) and "
            "position (your complete position and reasons, normally 100–200 words). The choice and text "
            "must express the same view. Do not invent a peer reply or an agreement label. JSON schema:\n"
            + canonical(position_schema(question)),
        ),
    )


def own_state(member, initial, route, replies, node_id):
    """Walk only this node's ancestors; never use a model-name/global latest state."""
    node = route.get(node_id)
    if member != node.receiver:
        raise ValueError("Position owner must be the scheduled receiver")
    previous = route.previous_own(node_id)
    value = replies[previous.id] if previous else initial[member]
    return {
        "source": previous.id if previous else f"initial/{member}",
        "choice": value["choice"],
        "position": value["position"],
    }


def turn_context(question, initial, route, replies, node_id, *, incoming=None, explicit_position=True):
    """One copy of each public message, with optional explicit pre-turn state.

    Only public replies enter the transcript. Other members' structured positions,
    self-labels, historical tone instructions and all measurement results stay out.
    The latest peer message is separated from its ancestors, not duplicated there.
    With explicit_position=False, updated private choice/position fields are never
    read; original answers remain part of the public history.
    """
    node = route.get(node_id)
    path = route.path(node_id)
    state = own_state(node.receiver, initial, route, replies, node_id) if explicit_position else None
    messages = list(initial_context(question, node.receiver))
    seen_initials = set()

    def add_initial(member):
        # If this is the current position, its full text is supplied below once.
        if member in seen_initials or (state is not None and state["source"] == f"initial/{member}"):
            return
        seen_initials.add(member)
        messages.append(
            Message(
                "assistant" if member == node.receiver else "user",
                canonical(
                    {"member": member, "source": f"initial/{member}", "contribution": initial[member]["position"]}
                ),
            )
        )

    add_initial(node.receiver)
    if node.parent:
        add_initial(path[0].sender)
    for ancestor in path[:-2]:
        messages.append(
            Message(
                "assistant" if ancestor.receiver == node.receiver else "user",
                canonical(
                    {
                        "member": ancestor.receiver,
                        "source": ancestor.id,
                        "reply_to": ancestor.parent or f"initial/{ancestor.sender}",
                        "contribution": replies[ancestor.id]["reply"],
                    }
                ),
            )
        )
    if state is not None:
        messages.append(Message("user", canonical({"your_current_position": state})))
    if incoming is None:
        incoming = replies[node.parent]["reply"] if node.parent else initial[node.sender]["position"]
    messages.append(
        Message(
            "user",
            canonical(
                {
                    "incoming_peer_message": {
                        "member": node.sender,
                        "source": node.parent or f"initial/{node.sender}",
                        "contribution": incoming,
                    }
                }
            ),
        )
    )
    return tuple(messages)


def debate_messages(question, tone, initial, route, replies, node_id, *, explicit_position=True):
    return (
        *turn_context(question, initial, route, replies, node_id, explicit_position=explicit_position),
        Message(
            "user",
            "For THIS turn, use the following debate instruction:\n"
            + (prompts.TONE_TEXT[tone] or "No additional friendly or hostile tone instruction.")
            + (
                "\nUse the entire supplied branch history and your_current_position to consider the incoming "
                if explicit_position
                else "\nUse the supplied public branch history to consider the incoming "
            )
            + "peer message. Direct your reply ONLY to incoming_peer_message, not to another earlier turn. "
            "In this same response, report your own resulting position on the ORIGINAL QUESTION. "
            "Your position may remain unchanged: do not change it merely because you were asked to report it. "
            "Return exactly one JSON object with four fields, in this order:\n"
            "reply: your substantive reply to the incoming peer message, normally 100–200 words.\n"
            "agreement: your agreement with that incoming message, NOT with the original question.\n"
            "choice: the original option letter expressing your resulting own view, or null if none fits.\n"
            "position: your complete resulting own position and reasons, normally 80–150 words. "
            "Keep this as one coherent paragraph, not separate claim/reason fields. The choice and position "
            "must be consistent with the reasoning expressed in your reply.\n"
            + AGREEMENT_RUBRIC_TEXT
            + "\nUse null for agreement only if you cannot determine its overall direction. "
            "Do not add fields, code fences or text outside the final JSON. Required schema:\n"
            + canonical(reply_schema(question)),
        ),
    )


def choice_messages(question, member, initial, route, replies, node_id, *, explicit_position=True):
    context = (
        initial_context(question, member)
        if node_id is None
        else turn_context(question, initial, route, replies, node_id, explicit_position=explicit_position)
    )
    return (
        *context,
        Message(
            "user",
            "Based on this discussion input, select the ORIGINAL QUESTION option that best expresses "
            "your own considered view. This is not your agreement with the peer. Select the closest "
            "listed option. Output exactly one of these letters and nothing else: " + ", ".join(question.labels),
        ),
    )


def text_messages(question, initial, route, replies, node_id, incoming, *, explicit_position=True):
    context = turn_context(
        question, initial, route, replies, node_id, incoming=incoming, explicit_position=explicit_position
    )
    target = "your_current_position.position"
    if not explicit_position:
        # A measurement target, not state fed back to debate or D1. Do not supply
        # the old option letter. Both D2 arms see exactly the same full text.
        state = own_state(route.get(node_id).receiver, initial, route, replies, node_id)
        context = (
            *context[:-1],
            Message(
                "user", canonical({"position_to_evaluate": {"source": state["source"], "text": state["position"]}})
            ),
            context[-1],
        )
        target = "position_to_evaluate.text"
    return (
        *context,
        Message(
            "user",
            "After considering incoming_peer_message, how much do you now agree with the full text in "
            f"{target}, considering it as a whole? This is YOUR OWN position recorded "
            "before this incoming message, not agreement with the peer. Evaluate that exact text, not a "
            "newly revised position. Output exactly one letter and nothing else: "
            "A = completely disagree; B = mostly disagree; C = somewhat disagree; "
            "D = neither agree nor disagree; E = somewhat agree; F = mostly agree; G = completely agree.",
        ),
    )


def parse_choice(text, question):
    if text not in question.labels:
        raise ValueError("D-choice must return exactly one original option letter, no prose or thinking")
    return {"choice": text}
