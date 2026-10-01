"""Explicit, serializable contracts; no global member state or default agreement label."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .agreement import AGREEMENT as AGREEMENT

TONES = ("friendly", "neutral", "hostile")
TEXT_SHIFT = {"unchanged": 0.0, "adjusted": 0.5, "conclusion_changed": 1.0, "unjudgeable": None}
SCREEN_JUDGES = ("gpt-5.5", "opus-4.8", "gemini-3.5-flash")
ROSTERS = {
    "same_model": ("gpt-5.6-terra",) * 3,
    "same_family": ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"),
    "mixed_family": ("gpt-5.6-terra", "Qwen/Qwen3.8-27B", "thinkingmachines/Inkling"),
}
OPEN_MODELS = frozenset(ROSTERS["mixed_family"][1:])
MAIN_STUDY_REUSE_POLICY = {
    "pilot_questions_may_be_in_main_80": True,
    "reuse_results_only_under_final_protocol": True,
    "synthetic_results_are_study_data": False,
    "note": (
        "Pilot membership does not exclude a question. Resolve input-context issues before inclusion. "
        "Keep changed-protocol runs as development records; rerun affected measurements under the "
        "final protocol. Do not select questions or retain runs based on observed effects."
    ),
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Screen:
    """May preserve an archived aggregate; never invent individual judge votes."""

    judges: tuple[str, ...]
    opinion_votes: int
    source: str
    source_sha256: str
    evidence_kind: str = "archived_aggregate"

    def validate(self) -> None:
        if set(self.judges) != set(SCREEN_JUDGES) or len(self.judges) != 3:
            raise ValueError("Screen must use the three original screening judges")
        if type(self.opinion_votes) is not int or not 2 <= self.opinion_votes <= 3:
            raise ValueError("Question needs at least two debatable_opinion votes")
        if not self.source or len(self.source_sha256) != 64:
            raise ValueError("Screening provenance and source SHA-256 are required")


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: tuple[str, ...]
    screen: Screen
    source: str = "GlobalOpinionQA"

    @property
    def fingerprint(self) -> str:
        return digest({"question": self.text, "options": self.options})

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(chr(65 + i) for i in range(len(self.options)))

    def validate(self) -> None:
        if not self.id or not self.text.strip() or not 2 <= len(self.options) <= 26:
            raise ValueError("Question needs an ID, full text, and 2–26 original options")
        if any(not option.strip() for option in self.options):
            raise ValueError("Blank answer option")
        self.screen.validate()

    @classmethod
    def from_dict(cls, value: dict) -> Question:
        screen = dict(value["screen"])
        screen["judges"] = tuple(screen["judges"])
        result = cls(
            value["id"],
            value["text"],
            tuple(value["options"]),
            Screen(**screen),
            value.get("source", "GlobalOpinionQA"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class PilotConfig:
    """Engineering defaults, recorded in the manifest; not final study hyperparameters."""

    roster: str = "mixed_family"
    seed: int = 20260925
    judge_model: str | None = None  # Explicit opt-in, not a silently approved model.
    chairman_model: str = "gpt-5.6-terra"
    debate_effort: str = "medium"
    chairman_effort: str = "medium"
    initial_tokens: int = 4096
    debate_tokens: int = 4096
    position_tokens: int = 1200
    probability_tokens: int = 128  # Includes native message framing; public answer is one letter.
    chairman_tokens: int = 4096
    judge_tokens: int = 2048
    cache: bool = False  # Enable explicitly; record cache writes as well as reads.
    closed_provider: str = "databricks"
    databricks_profile: str = "un"

    @property
    def members(self) -> tuple[str, ...]:
        return ROSTERS[self.roster]

    def validate(self) -> None:
        if self.closed_provider not in ("databricks", "direct"):
            raise ValueError("Select Databricks or direct closed-model access explicitly")
        if not self.databricks_profile.strip():
            raise ValueError("Databricks profile cannot be blank")
        if self.closed_provider == "databricks" and self.cache:
            raise ValueError("Explicit cache breakpoints are not supported by this Databricks adapter")
        if self.roster not in ROSTERS:
            raise ValueError("Unknown roster")
        if self.debate_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Formal debate must have reasoning enabled")
        if self.roster == "mixed_family" and self.debate_effort not in ("low", "medium", "xhigh"):
            raise ValueError("Qwen3.8 supports low/medium/xhigh; do not silently map reasoning efforts")
        if self.chairman_model != "gpt-5.6-terra":
            raise ValueError("Protocol fixes the chairman to Terra")
        if self.judge_model not in (None, "gemini-3.8-flash"):
            raise ValueError("This pilot adapter supports only the explicitly selected Gemini judge")
        if any(
            type(n) is not int or n <= 0 or n > 128000
            for n in (
                self.initial_tokens,
                self.debate_tokens,
                self.position_tokens,
                self.probability_tokens,
                self.chairman_tokens,
                self.judge_tokens,
            )
        ):
            raise ValueError("Invalid output-token limit")


@dataclass(frozen=True)
class Message:
    role: str
    text: str


@dataclass(frozen=True)
class Request:
    key: str
    purpose: str
    model: str
    messages: tuple[Message, ...]
    effort: str
    max_output_tokens: int
    schema: dict | None = None
    cache: bool = False
    candidate_labels: tuple[str, ...] = ()
    scoring: dict | None = None  # Exact previously sampled prefix, not a re-rendered conversation.

    def document(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int  # Includes reasoning tokens, not just visible text.
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    status: str = "completed"
    raw: dict | None = None
    readout: dict | None = None
