"""The core user flow: question + config -> debate -> choose your disagreement evaluator.

1. A committee instance takes a question and a config (including a predefined debate tone:
   neutral / friendly / hostile) and the debate fires up.
2. The full trajectory is captured; the evaluator layers A/B/C/D each read a different
   disagreement signal from it — pick the ones you need.
"""
import dspy

from llm_committee import LLMCommitteeSync
from llm_committee.evaluators import layer_a

# --- 1. Configure the committee: members, chairman, and a debate tone -----------------------
chairman_lm = dspy.LM("openai/gpt-4o")
member_lms = [
    dspy.LM("openai/gpt-5-mini"),
    dspy.LM("openrouter/qwen/qwen3-32b"),
    dspy.LM("openrouter/google/gemini-2.5-flash-lite"),
]

committee = LLMCommitteeSync(
    num_members=3,
    chairman_lm=chairman_lm,
    member_lms=member_lms,
    max_iterations=2,
    debate_stance="hostile",  # predefined tones: "neutral" | "friendly" | "hostile"
)

result = committee(
    committee_task="Debate the statement and reach a considered position.",
    agent_task="",
    task_input={"question": "Remote work makes teams more productive. Agree or disagree?"},
    task_context={},
)
print("Final answer:\n", result.final_judgement)

# --- 2. Evaluate the disagreement, layer by layer -------------------------------------------

# Layer A (free, confounded): what each debater SAYS its agreement is. The tone instruction
# mechanically moves this label, so never read it alone as evidence of real disagreement.
print("\nLayer A:", layer_a.summarize(result.trajectory_graph))

# Layer B (one judge call per turn): does the reply TEXT actually push back? Use a judge from a
# model family outside the committee, and show it only the parent turn and the reply — never the
# label or the tone condition.
#   from llm_committee.evaluators import layer_b
#   verdict = layer_b.judge_pushback(parent_argument=..., reply=..., judge="gemini")

# Layer C (re-queries the member models): does the dissent survive DELETING the tone instruction
# that elicited it? The probe re-asks each dissenting turn twice — once without the instruction,
# once with it retained (the resampling noise floor) — and reports the paired difference.
#   from llm_committee.evaluators import layer_c
#   layer_c.register_task_spec("my-task", committee_task="Debate the statement ...")

# Layer D (open-weight members only): does the debater's token-level stance distribution move?
# Point LLM_COMMITTEE_ENDPOINTS at OpenAI-compatible endpoints that expose logprobs
# (see llm_committee/evaluators/endpoints.example.json).
#   from llm_committee.evaluators import layer_d
