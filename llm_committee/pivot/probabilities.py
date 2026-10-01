"""Candidate-conditional probabilities at a recorded output position, at read T=1."""

from __future__ import annotations

import math

from .models import digest

RATING_LABELS = tuple("ABCDEFG")
TOPK_ZERO_FILL_POLICY = {
    "name": "topk_zero_fill_v1",
    "reported_topk": 20,
    "missing_candidate_probability": 0.0,
    "normalization": "returned candidate probabilities only",
    "supplemental_scoring": False,
    "interpretation": "Top-20 truncation approximation; absent candidates are not known to have true zero probability",
}
LOGPROB_ATOL = 0.01  # Recorded engineering alignment check, not a distribution rescaling.
LOGPROB_MISMATCH_MESSAGES = (
    "Supplemental scores disagree with recorded top-k at the same prefix",
    "Supplemental scoring does not reproduce the sampled-token logprob",
)


class LogprobMismatch(ValueError):
    """Same-prefix score disagreement: preserve as missing data, never normalize it."""


def uses_topk_zero_fill(manifest: dict) -> bool:
    """New manifests opt in explicitly; archived manifests retain exact-scoring semantics."""
    policy = manifest.get("probability_readout")
    if policy is None:
        return False
    if policy != TOPK_ZERO_FILL_POLICY:
        raise ValueError("Unknown probability-readout policy")
    return True


def candidate_ids(tokenizer, labels: tuple[str, ...]) -> dict[str, int]:
    ids = {}
    for label in labels:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1 or tokenizer.decode(tokens) != label:
            raise ValueError(f"Candidate {label!r} is not exactly one round-trippable token")
        ids[label] = int(tokens[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError("Candidate token IDs are not distinct")
    return ids


def locate_visible_start(tokenizer, tokens: list[int], text: str) -> int:
    """Locate the actual visible-answer start, retaining generated native framing.

    No re-tokenization of the prefix and no search for a convenient later letter.
    Require a unique exact content match with a token boundary at its first character.
    """
    if not text or text != text.lstrip():
        raise ValueError("Direct output must begin with the requested letter, without whitespace")
    decoded = tokenizer.decode(tokens)
    if not isinstance(decoded, str) or decoded.count(text) != 1:
        raise ValueError("Cannot uniquely align parsed content with raw sampled tokens")
    start = decoded.index(text)
    # At most one decode per candidate boundary; typical direct outputs have <10 framing tokens.
    for index in range(len(tokens)):
        prefix = tokenizer.decode(tokens[:index])
        if len(prefix) == start and decoded.startswith(prefix):
            first = tokenizer.decode([tokens[index]])
            if not first or not text.startswith(first):
                raise ValueError("First visible token straddles a framing/content boundary")
            return index
        if len(prefix) > start:
            break
    raise ValueError("Visible answer does not begin at an identifiable token boundary")


def capture_readout(tokenizer, prompt_ids, tokens, text, labels, topk, sample_logprobs) -> dict:
    index = locate_visible_start(tokenizer, tokens, text)
    ids = candidate_ids(tokenizer, labels)
    if text.splitlines()[0] in labels and tokens[index] != ids[text.splitlines()[0]]:
        raise ValueError("Choice is not emitted as its canonical single token")
    prefix = list(prompt_ids) + list(tokens[:index])
    pairs = topk[index] if topk and len(topk) > index and topk[index] else []
    scores = {int(token): float(score) for token, score in pairs}
    observed = float(sample_logprobs[index]) if sample_logprobs is not None else None
    return {
        "candidate_ids": ids,
        "candidate_logprobs": {label: scores[token] for label, token in ids.items() if token in scores},
        "prefix_token_ids": prefix,
        "prefix_sha256": digest(prefix),
        "sampled_token_id": int(tokens[index]),
        "sampled_token_logprob": observed,
        "sampled_output_index": index,
        "source": "sample_topk",
        "sampling_temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "read_temperature": 1.0,
    }


def distribution(readout: dict, labels: tuple[str, ...], *, missing_as_zero: bool = False) -> dict:
    """Normalize observed candidates; optionally approximate absent top-k candidates as zero.

    Keep observed logprobs unchanged: logprob 0 means probability 1, not probability 0.
    The explicit default preserves exact-score checks for archived analyses.
    """
    if readout.get("read_temperature") != 1.0:
        raise ValueError("Read temperature must remain 1")
    if readout.get("sampling_temperature") != 1.0 or readout.get("top_p") != 1.0 or readout.get("top_k") != -1:
        raise ValueError("Probability read must use untruncated temperature-1 sampling")
    if set(readout["candidate_ids"]) != set(labels):
        raise ValueError("Candidate identity mismatch")
    scores = readout["candidate_logprobs"]
    if not set(scores).issubset(labels):
        raise ValueError("Unexpected candidate log probabilities")
    if not scores:
        raise ValueError("No candidate probabilities returned; cannot normalize")
    missing = [label for label in labels if label not in scores]
    if missing and not missing_as_zero:
        raise ValueError("Incomplete candidate log probabilities; missing scores are not zeros")
    if missing_as_zero and readout["source"] not in ("sample_topk", "synthetic_sample_topk"):
        raise ValueError("Zero-fill analysis must use the original sampled top-k, not supplemental scores")
    if any(not math.isfinite(v) or v > 1e-6 for v in scores.values()):
        raise ValueError("Invalid candidate log probabilities")
    peak = max(scores.values())
    normalizer = sum(math.exp(value - peak) for value in scores.values())
    log_mass = peak + math.log(normalizer)
    if log_mass > 1e-4:
        raise ValueError("Candidate probability mass exceeds one")
    result = {
        "probabilities": {
            label: math.exp(scores[label] - peak) / normalizer if label in scores else 0.0 for label in labels
        },
        "candidate_logprobs": dict(scores),
        "candidate_mass": math.exp(log_mass),
        "candidate_log_mass": log_mass,
        "source": readout["source"],
        "prefix_sha256": readout["prefix_sha256"],
        "read_temperature": 1.0,
    }
    if missing_as_zero:
        result.update(
            probability_readout_policy=TOPK_ZERO_FILL_POLICY["name"],
            missing_candidate_labels=missing,
            zero_fill_applied=bool(missing),
            candidate_mass_is_lower_bound=bool(missing),
        )
    return result


def merge_scored_readout(original: dict, scored: dict, labels: tuple[str, ...]) -> dict:
    for field in ("prefix_sha256", "candidate_ids", "sampled_token_id"):
        if original[field] != scored[field]:
            raise ValueError(f"Supplemental scoring changed {field}")
    for label, value in original["candidate_logprobs"].items():
        if abs(value - scored["candidate_logprobs"][label]) > LOGPROB_ATOL:
            raise LogprobMismatch(LOGPROB_MISMATCH_MESSAGES[0])
    observed = original.get("sampled_token_logprob")
    if observed is not None and abs(observed - scored["sampled_token_logprob"]) > LOGPROB_ATOL:
        raise LogprobMismatch(LOGPROB_MISMATCH_MESSAGES[1])
    merged = {**original, **scored}
    distribution(merged, labels)
    return merged


def choice_change(before: dict, after: dict, reference_choice: str | None) -> float | None:
    if reference_choice is None:
        return None
    return 100 * (after["probabilities"][reference_choice] - before["probabilities"][reference_choice])


def own_position_difference(argument: dict, control: dict) -> float:
    return sum(
        (i + 1) * (argument["probabilities"][label] - control["probabilities"][label])
        for i, label in enumerate(RATING_LABELS)
    )
