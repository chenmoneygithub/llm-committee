"""Evaluators for committee debate: is the disagreement real?

Each layer answers a different question about the same debate, and none should be read as another
(the study behind this library found they genuinely come apart):

=======  ============================================  ==============================================
Layer    Question                                      Entry point
=======  ============================================  ==============================================
A        Does the model *say* it disagrees?            ``layer_a.turn_labels`` / ``layer_a.summarize``
B        Does the reply *text* actually push back?     ``layer_b.judge_pushback`` (condition-blind
                                                       judge; use a model family outside the
                                                       committee)
C        Does the position survive deleting the        ``layer_c`` re-issues a turn with the stance
         stance instruction that elicited it?          instruction removed, paired with a matched
                                                       retained re-ask as the resampling noise floor
D        Does the debater's token-level stance         ``layer_d`` (needs open weights: reads the
         distribution move?                            next-token logprob distribution over a Likert
                                                       probe)
=======  ============================================  ==============================================

Layer A is free but confounded (the same call receives the stance instruction and emits the
label). Layer B needs one judge call per turn. Layer C needs re-queries of the original member
models. Layer D needs OpenAI-compatible endpoints that expose logprobs (see
``endpoints.example.json``).
"""
from llm_committee.evaluators import layer_a  # noqa: F401

__all__ = ["layer_a", "layer_b", "layer_c", "layer_d", "stats", "trajectory_io"]


def __getattr__(name):
    # layer_b/c/d pull in heavier deps (dspy, scipy, numpy) — import lazily on first access.
    if name in __all__:
        import importlib

        return importlib.import_module(f"llm_committee.evaluators.{name}")
    raise AttributeError(name)
