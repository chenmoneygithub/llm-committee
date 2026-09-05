"""
LLM Committee: Single-process synchronous version for research/debugging
"""

import random
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any

import dspy
from pydantic import BaseModel, model_validator


class AgreedLevel(Enum):
    fully_agreed = "fully_agreed"
    partially_agreed = "partially_agreed"
    partially_disagreed = "partially_disagreed"
    fully_disagreed = "fully_disagreed"


# Debate-stance instructions injected into the MEMBER DEBATE signature only (not initial opinion
# or aggregation), to study how a member's predisposition toward (dis)agreement affects debate
# dynamics. "neutral" leaves the base signature unchanged.
DEBATE_STANCE_INSTRUCTIONS = {
    "neutral": "",
    "friendly": (
        "\n\nDEBATE STANCE — COOPERATIVE: Approach this debate looking for common ground. Where "
        "another member makes a reasonable point, acknowledge it and build on it. Prefer "
        "synthesis and consensus; concede points readily when the other side has merit. Only "
        "maintain disagreement when you have a strong, specific reason."
    ),
    "hostile": (
        "\n\nDEBATE STANCE — ADVERSARIAL: Your job is to stress-test every position, including "
        "your own. Actively look for weaknesses, hidden assumptions, missing evidence, and "
        "counterexamples in other members' arguments. Do NOT agree merely to be agreeable or to "
        "reach consensus — agreement is only acceptable when you are genuinely convinced after "
        "trying hard to refute. Steelman the opposing view, then attack it. If you still hold a "
        "different position, say so plainly and defend it. Productive disagreement is the goal; "
        "premature consensus is a failure."
    ),
}


# Debate-timing modes control WHEN (and whether) members debate before the chairman aggregates.
# Studied to separate "debate every round" from "draft then debate once" from "no debate":
#   "per_turn": members see each other and refine every round for max_iterations rounds (DEFAULT;
#               the original behavior — behavior-preserving).
#   "none":     no debate at all — members' initial independent opinions feed straight to the
#               chairman (the Self-MoA / no-interaction baseline; 0 debate rounds).
#   "after":    members draft independently, then a SINGLE debate round over those drafts
#               (regardless of max_iterations), then aggregate ("debate after independent drafting").
DEBATE_TIMING_CHOICES = ("per_turn", "none", "after")


class DebateTurn(BaseModel):
    judge_id: str
    reasoning: str
    argument: str
    agreed_level: AgreedLevel | None


class MemberOpinion(BaseModel):
    member_id: str
    opinion: str
    reasoning: str = ""


class DebateOpinion(BaseModel):
    opinion: str = ""
    agreed_level: AgreedLevel = AgreedLevel.partially_agreed
    # Descriptive fields default to empty so a weaker member that omits one (common with smaller /
    # reasoning models) degrades gracefully instead of raising a ValidationError that crashes the
    # entire committee debate. opinion + agreed_level are the load-bearing fields.
    reasoning: str = ""
    confidence: str = ""  # "high", "medium", or "low" - how confident are you in this position?
    key_uncertainty: str = ""  # What are you most uncertain about? (optional)
    opinion_shift: int = 0  # Scale 0-4: 0=no change, 1=slight, 2=moderate, 3=significant, 4=complete reversal
    shift_explanation: str = ""  # Why did your opinion shift?
    # Measurement-integrity flag: True iff the model actually EMITTED agreed_level, False if it fell
    # back to the partially_agreed default above. Because the default is itself a valid enum value,
    # a missing field is otherwise indistinguishable from a genuine "partially_agreed" — which
    # silently inflates the partial-agreement rate (the pure-sonnet 85.5% finding may be partly this
    # artifact). We record presence at construction so analysis can exclude/flag defaulted labels.
    # Not part of the LM's output schema (see MemberDebateSignature): DSPy populates the descriptive
    # fields, and this validator observes whether agreed_level was among them.
    agreed_level_explicit: bool = False

    @model_validator(mode="before")
    @classmethod
    def _record_agreed_level_presence(cls, data):
        """Set agreed_level_explicit from whether agreed_level was present in the raw input.

        Runs before field defaults are applied, so a missing/None agreed_level is detectable here
        even though the field itself defaults to partially_agreed. Only acts on dict-shaped input
        (the normal DSPy/JSON construction path); passes other shapes through untouched.
        """
        if isinstance(data, dict):
            present = data.get("agreed_level") is not None
            # Don't clobber an explicitly-provided flag (e.g. when reconstructing from saved JSON).
            data.setdefault("agreed_level_explicit", present)
        return data


class DebateThread(BaseModel):
    """A debate thread containing context and the specific turn to respond to."""

    last_turn: DebateTurn  # The specific turn you must respond to
    previous_context: list[DebateTurn]  # Earlier turns for context (optional reference)


@dataclass
class TrajectoryNode:
    """A node in the trajectory tree representing one debate evolution step.

    Each member maintains a trajectory tree where nodes represent opinion evolution.
    Nodes are immutable - new opinions create new child nodes.
    """

    node_id: str  # Unique identifier for this node
    member_id: str  # Which member owns this trajectory
    iteration: int  # Which debate iteration this occurred in
    parent_id: str | None  # Parent node ID (None for root/initial opinion)
    children_ids: list[str]  # List of child node IDs

    # Opinion evolution
    debate_opinion: str  # Response to debate arguments
    reasoning: str  # LLM's reasoning
    agreed_level: AgreedLevel  # Agreement level with last round's debate opinion

    # Opinion change tracking
    opinion_shift: int = 0  # Scale 0-4: how much did this member's opinion change?
    shift_explanation: str = ""  # Why did the opinion shift?

    # Measurement-integrity: True iff the model actually emitted agreed_level (vs the schema default).
    # Propagated from DebateOpinion.agreed_level_explicit so offline analysis can flag/exclude
    # defaulted labels. Roots (initial opinions) have no agreed_level, so this stays False for them.
    agreed_level_explicit: bool = False

    # Optional tool-use trajectory (only populated on root nodes when use_tools is enabled).
    # Holds the dspy.ReAct trajectory dict (thought_i/tool_name_i/tool_args_i/observation_i keys).
    # Defaults to None so existing code and JSON serialization are unaffected.
    tool_trajectory: dict | None = None


# DSPy Signatures for LLM Committee
class MemberInitialThoughtSignature(dspy.Signature):
    """Form your initial independent opinion on the committee task.

    You are a committee member working to accomplish a specific task.
    The task may involve evaluating something, refining content, brainstorming ideas,
    critiquing work, or other collaborative activities.

    CRITICAL - Understanding Your Role:
    - COMMITTEE TASK: What should THIS COMMITTEE accomplish? (your primary objective)
    - AGENT TASK: What was the original agent trying to do? (context for your work)
    - Focus entirely on accomplishing the committee_task

    Your approach should align with what the committee_task asks:
    - Read the committee_task carefully to understand success criteria
    - Consider the task_input as your primary information source
    - Use task_context for additional relevant information
    - Form your independent opinion on how to best accomplish the committee_task

    EPISTEMIC HUMILITY:
    - Express uncertainty when appropriate (e.g., "I tentatively think...", "I lean toward...")
    - Acknowledge limitations in your reasoning
    - Avoid overconfident claims unless you have strong evidence
    """

    committee_task: str = dspy.InputField(
        desc=(
            "What should this committee accomplish? (e.g., 'Evaluate if the agent's solution is correct', "
            "'Refine this essay for clarity', 'Brainstorm marketing strategies')"
        )
    )
    agent_task: str = dspy.InputField(
        desc=(
            "What was the original agent trying to do? "
            "(provides context for committee's work, may be empty if not applicable)"
        )
    )
    task_input: dict[str, Any] = dspy.InputField(
        desc=("Primary information for the committee task " "(e.g., question, draft content, problem statement)")
    )
    task_context: dict[str, Any] = dspy.InputField(
        desc=("Additional context " "(e.g., agent's output to evaluate, constraints, requirements, style guides)")
    )

    initial_opinion: str = dspy.OutputField(
        desc=(
            "Your independent opinion on how to accomplish the committee_task. "
            "Be specific about your reasoning and express uncertainty when appropriate."
        )
    )


class MemberDebateSignature(dspy.Signature):
    """Engage in committee debate by responding to other members' arguments.

    You are working to accomplish the COMMITTEE TASK. The committee task may involve
    evaluating something, refining content, brainstorming solutions, or other collaborative work.

    CRITICAL - Understanding Your Role:
    - COMMITTEE TASK: What should THIS COMMITTEE accomplish? (your primary objective)
    - AGENT TASK: What was the original agent trying to do? (context for your work)
    - Focus on accomplishing the committee_task through collaborative debate

    You will receive debate threads. Each thread contains:
    - last_turn: The specific argument you MUST respond to
    - previous_context: Earlier turns in this thread (for background only)

    CRITICAL INSTRUCTIONS FOR DEBATE RESPONSES:

    1. RESPOND TO last_turn ONLY: Your debate_opinions[i] should directly address what is said
       in debate_threads[i].last_turn. Quote or reference their specific argument.

       Example GOOD response: "I agree with your point that the solution is correct because..."
       Example BAD response: "The agent's output is correct because..." (ignores the last turn)

    2. AGREED_LEVEL = Agreement with last_turn's position:

       - If last_turn supports position X and you agree with X → fully_agreed
       - If last_turn supports position X but you support opposite → fully_disagreed
       - If last_turn supports position X but you're unsure → partially_agreed or partially_disagreed

       DO NOT set agreed_level based on earlier turns in previous_context!

    3. EPISTEMIC HUMILITY - Express uncertainty when appropriate:

       - Use tentative language when you're not certain: "I tentatively think...",
         "My current view is...", "I lean toward..."
       - Acknowledge gaps in your reasoning: "I'm uncertain about X, but...",
         "While I don't have conclusive evidence..."
       - Avoid overconfident claims like "definitely", "objectively", "clearly wrong"
         unless you have ironclad evidence
       - When making claims about correctness, acknowledge you're making an inference
       - Include your confidence level (high/medium/low) and key uncertainties

    4. DEBATE NORMS - Guidelines for productive discussion:

       a) Don't flip easily: Don't change your position unless the other member
          presents genuinely new insights or reveals flaws in your reasoning.

       b) Ask for evidence: If another member makes confident assertions,
          ask them to justify WHY. What makes them so sure?

       c) Identify the disagreement: Are you disagreeing about:
          - What the committee task is asking? (interpretation)
          - Assessment criteria? (what makes something "correct" or "better")
          - Specific evidence or reasoning?

          Pinpoint the source of disagreement before conceding.

       d) Steelman, don't strawman: Engage with the strongest version of the other's
          argument, not a weak mischaracterization.

    For per-thread responses:
    - debate_opinions[i] responds ONLY to debate_threads[i].last_turn
    - previous_context is for background - don't respond to those turns
    - Do NOT use information from debate_threads[j] when forming debate_opinions[i]

    After all per-thread responses, synthesize a refined_opinion drawing on ALL threads.
    The goal is truth, not winning.
    """

    committee_task: str = dspy.InputField(
        desc="What should this committee accomplish? Read this carefully for success criteria."
    )
    agent_task: str = dspy.InputField(
        desc="What was the original agent trying to do? (provides context for committee's work)"
    )
    task_input: dict[str, Any] = dspy.InputField(desc="Primary information for the committee task.")
    task_context: dict[str, Any] = dspy.InputField(
        desc="Additional context (e.g., agent's output, constraints, requirements)."
    )
    own_opinion: str = dspy.InputField(desc="Your current opinion on how to accomplish the committee task.")
    debate_threads: list[DebateThread] = dspy.InputField(
        desc=(
            "Debate threads from other members. Each thread has: "
            "(1) last_turn - the specific argument you MUST respond to, "
            "(2) previous_context - earlier turns for background. "
            "You must respond to last_turn, not previous_context."
        )
    )
    my_id: str = dspy.InputField(desc="Your member ID.")

    debate_opinions: list[DebateOpinion] = dspy.OutputField(
        desc=(
            "Per-thread responses. Must have the SAME length and order as debate_threads. "
            "debate_opinions[i] is your direct RESPONSE to debate_threads[i].last_turn. "
            "The agreed_level reflects agreement with that SPECIFIC turn's position. "
            "Include confidence (high/medium/low) and key_uncertainty if applicable. "
            "Express epistemic humility - use tentative language when uncertain. "
            "\n\nOPINION SHIFT TRACKING - Assess if your view changed from your previous position:\n"
            "- opinion_shift (int, 0-4): How much did your opinion change?\n"
            "  * 0 = No change: Maintained exact same position\n"
            "  * 1 = Slight refinement: Minor wording adjustments, same core view\n"
            "  * 2 = Moderate shift: Adjusted some reasoning or details, core position similar\n"
            "  * 3 = Significant change: Changed position on key aspects\n"
            "  * 4 = Complete reversal: Fundamentally opposite view from before\n"
            "- shift_explanation (str): If opinion_shift > 0, explain what new insight caused the change. "
            "What specific argument or evidence made you reconsider? If opinion_shift = 0, leave empty."
        )
    )
    refined_opinion: str = dspy.OutputField(
        desc=(
            "Your AGGREGATED opinion after considering ALL threads together. "
            "Combine evidence and insights from every thread into one coherent opinion "
            "on whether the agent's OUTPUT correctly accomplishes the TASK. "
            "Express uncertainty when appropriate."
        )
    )


class AggregationSignature(dspy.Signature):
    """Synthesize all committee members' refined opinions into a final judgment.

    You are the committee chairman making the final decision. You have N members who have
    thoroughly debated how to accomplish the COMMITTEE TASK.
    Each member has provided their final refined opinion after multiple rounds of discussion.

    CRITICAL - Understanding Your Role:
    - COMMITTEE TASK: What should this committee accomplish? (the primary objective)
    - AGENT TASK: What was the original agent trying to do? (context for the work)

    YOUR PRIMARY JOB - SYNTHESIZE THE DEBATE:
    - Your PRIMARY responsibility is to synthesize what the members discussed and concluded
    - DO NOT independently re-do the work from scratch
    - DO NOT introduce new concerns or criteria that members didn't raise
    - Trust the members' analysis - they already thoroughly worked on the task
    - The task_input and task_context are provided ONLY for reference if you need to:
      * Verify a specific factual claim made by a member
      * Resolve an ambiguity when members cite different parts of the content
      * Understand context when member opinions reference specific details
    - DO NOT use task_input/task_context to do your own independent work
    - DO NOT reject outputs for reasons the members didn't identify

    SYNTHESIS APPROACH:
    - Read the committee_task carefully to understand what criteria the members were using
    - Consider both the substance of arguments AND the confidence/uncertainty expressed
    - A well-reasoned uncertain opinion may be more valuable than an overconfident wrong one
    - Don't automatically favor the most confident-sounding member
    - If members disagree, identify the source of disagreement (criteria? interpretation? evidence?)
    - Weigh the strength of reasoning, not just the confidence of assertion

    EPISTEMIC HUMILITY:
    - If the committee is divided or uncertain, acknowledge this in your judgment
    - Don't force false certainty - it's OK to say "the evidence is mixed"
    - Your final decision should reflect the balance of evidence and reasoning

    Your task: Synthesize these diverse perspectives into one coherent final judgment that:
    - Acknowledges areas of consensus among members
    - Addresses points of disagreement fairly and identifies their source
    - Weighs the quality of reasoning, not just confidence levels
    - Provides an overall evaluation with clear reasoning
    - Gives a final verdict on the committee task
    """

    committee_task: str = dspy.InputField(
        desc="What should this committee accomplish? Read this carefully for success criteria."
    )
    agent_task: str = dspy.InputField(
        desc="What was the original agent trying to do? (provides context for committee's work)"
    )
    task_input: dict[str, Any] = dspy.InputField(desc="Primary information for the committee task (for reference only)")
    task_context: dict[str, Any] = dspy.InputField(
        desc="Additional context (e.g., agent's output, constraints) (for reference only)"
    )
    member_opinions: list[MemberOpinion] = dspy.InputField(
        desc=(
            "All committee members' final refined opinions after debate. "
            "Each contains member_id and their opinion. "
            "Pay attention to expressed uncertainty and confidence levels."
        )
    )
    final_judgement: str = dspy.OutputField(
        desc=(
            "Comprehensive final judgment synthesizing all member perspectives on the committee task. "
            "Include: consensus points, disagreements (and their source), "
            "quality of reasoning from each side, overall evaluation, and final verdict. "
            "Acknowledge uncertainty if the committee is divided."
        )
    )
    output_accepted: bool = dspy.OutputField(
        desc=(
            "Final committee decision on whether the committee_task was successfully accomplished. "
            "True if the balance of evidence and reasoning supports a positive conclusion, False otherwise. "
            "This should reflect the strength of arguments, not just vote counting."
        )
    )


@dataclass
class MemberState:
    """Stateful container for a committee member tracking their current trajectory node"""

    member_id: str
    opinion_history: list[str]


class LLMCommitteeSync(dspy.Module):
    """Synchronous single-process version of LLM Committee"""

    def __init__(
        self,
        num_members: int = 5,
        chairman_lm: str = None,
        member_lms: list[str] = None,
        max_iterations: int = 5,
        members_to_explore: int = 2,
        max_messages_per_member: int = 3,
        history_window: int = 3,
        growth_iterations: int = 1,
        custom_instructions: str = "",
        use_tools: bool = False,
        tools: list | None = None,
        react_max_iters: int = 5,
        debate_stance: str = "neutral",
        debate_timing: str = "per_turn",
    ):
        """Initialize LLM Committee with tree-based trajectory tracking.

        Args:
            num_members: Number of committee members (default: 5)
            member_lms: Optional list of LM instances, one per member (default: None, uses global LM)
            max_iterations: Maximum debate iterations (default: 5)
            members_to_explore: Number of members to send to (exploration) (default: 2)
            max_messages_per_member: Maximum messages to process per member per iteration (default: 3)
            history_window: Number of recent debate rounds to send to LLM (default: 3)
            growth_iterations: Number of iterations to grow before converging (default: 1)
            custom_instructions: Optional task-specific instructions to inject into all signatures (default: "")
            use_tools: If True, members may call tools (e.g. web search) while forming their INITIAL
                opinion via dspy.ReAct. Debate turns remain text-based. (default: False)
            tools: List of callables to expose to ReAct during the initial-opinion phase. If use_tools
                is True and tools is None, llm_committee.tools.default_tools() is used. (default: None)
            react_max_iters: Maximum ReAct reasoning/acting iterations per member for the initial
                opinion (only used when use_tools is True). (default: 5)
            debate_stance: ``friendly`` | ``neutral`` | ``hostile`` — a predisposition injected into
                the debate signature only. (default: "neutral")
            debate_timing: ``per_turn`` | ``none`` | ``after`` — WHEN debate happens (see
                DEBATE_TIMING_CHOICES). ``per_turn`` (default) is the original behavior: members
                refine every round for max_iterations rounds. ``none`` runs no debate (initial
                opinions feed straight to the chairman; Self-MoA baseline). ``after`` runs exactly
                one debate round over the independent drafts regardless of max_iterations.

        Note: During growth phase (iteration < growth_iterations), each trajectory is routed to
              members_to_explore random members. After growth phase, trajectories converge by
              routing to ancestors selected based on agreement level.

        Example:
            # Use different LMs for different members
            member_lms = [
                dspy.OpenAI(model="gpt-4"),
                dspy.Claude(model="claude-3-opus"),
                dspy.OpenAI(model="gpt-3.5-turbo"),
            ]
            committee = LLMCommitteeSync(num_members=3, member_lms=member_lms)
        """
        super().__init__()  # Initialize dspy.Module parent

        self.num_members = num_members
        self.max_iterations = max_iterations
        self.members_to_explore = members_to_explore
        self.max_messages_per_member = max_messages_per_member
        self.history_window = history_window
        self.growth_iterations = growth_iterations

        self.chairman_lm = chairman_lm
        # Build member_lm_map from member_lms list
        self.member_lm_map = {}
        if member_lms:
            for i, lm in enumerate(member_lms):
                self.member_lm_map[f"member_{i}"] = lm

        self._member_ids = [f"member_{i}" for i in range(num_members)]
        self.custom_instructions = custom_instructions

        # Tool-use configuration (applies to the INITIAL-opinion phase only; debate stays text-based).
        self.use_tools = use_tools
        self.react_max_iters = react_max_iters
        if use_tools and tools is None:
            # Import lazily so the dependency is only required when tools are actually requested.
            from llm_committee.tools import default_tools

            tools = default_tools()
        self.tools = tools

        # Resolve the debate-stance instruction (friendly/neutral/hostile). Injected into the
        # DEBATE signature only — initial opinions and aggregation stay neutral so the stance's
        # effect on debate dynamics is isolated.
        self.debate_stance = debate_stance
        if debate_stance not in DEBATE_STANCE_INSTRUCTIONS:
            raise ValueError(
                f"Unknown debate_stance '{debate_stance}'. Choices: {list(DEBATE_STANCE_INSTRUCTIONS)}"
            )
        stance_text = DEBATE_STANCE_INSTRUCTIONS[debate_stance]

        # Debate-timing mode controls WHEN debate happens relative to drafting (see
        # DEBATE_TIMING_CHOICES). "per_turn" reproduces the original behavior exactly; the effective
        # debate-round count is resolved in forward() so max_iterations is left untouched.
        self.debate_timing = debate_timing
        if debate_timing not in DEBATE_TIMING_CHOICES:
            raise ValueError(
                f"Unknown debate_timing '{debate_timing}'. Choices: {list(DEBATE_TIMING_CHOICES)}"
            )

        # Create signatures (with custom instructions if provided)
        if custom_instructions:
            # Use DSPy's built-in with_instructions() method
            initial_sig = MemberInitialThoughtSignature.with_instructions(custom_instructions)
            debate_sig = MemberDebateSignature.with_instructions(custom_instructions)
            aggregation_sig = AggregationSignature.with_instructions(custom_instructions)
        else:
            # Use base signatures
            initial_sig = MemberInitialThoughtSignature
            debate_sig = MemberDebateSignature
            aggregation_sig = AggregationSignature

        # Append the debate stance to the debate signature's instructions (after custom_instructions
        # if any). neutral => empty => unchanged base behavior.
        if stance_text:
            base_debate_instructions = debate_sig.instructions
            debate_sig = debate_sig.with_instructions(base_debate_instructions + stance_text)

        # Instantiate the DSPy modules with the appropriate signatures.
        # When tools are enabled, the initial-opinion step becomes a dspy.ReAct agent so members can
        # gather evidence (e.g. web search) before forming their independent opinion. ReAct reads the
        # signature's instructions/fields directly, so custom_instructions are already baked into
        # initial_sig above; if ReAct construction fails for any reason we fall back to the base
        # signature and warn rather than crashing the committee.
        if use_tools:
            try:
                self.member_initial = dspy.ReAct(initial_sig, tools=self.tools, max_iters=self.react_max_iters)
            except Exception as e:  # noqa: BLE001 - degrade gracefully to the base signature
                warnings.warn(
                    f"Failed to construct dspy.ReAct for the initial-opinion phase "
                    f"({type(e).__name__}: {e}); falling back to ReAct on the base "
                    f"MemberInitialThoughtSignature without custom instructions.",
                    stacklevel=2,
                )
                self.member_initial = dspy.ReAct(
                    MemberInitialThoughtSignature, tools=self.tools, max_iters=self.react_max_iters
                )
        else:
            self.member_initial = dspy.ChainOfThought(initial_sig)
        self.member_debate = dspy.ChainOfThought(debate_sig)
        self.aggregation = dspy.ChainOfThought(aggregation_sig)

    def _get_trajectory_path(self, trajectory_graph: dict[str, TrajectoryNode], node_id: str) -> list[TrajectoryNode]:
        """Get path from root to given node in trajectory tree.

        Args:
            trajectory_graph: The trajectory graph to query
            node_id: Target node ID

        Returns:
            List of nodes from root to target (inclusive)
        """
        path = []
        current_id = node_id

        while current_id is not None:
            node = trajectory_graph[current_id]
            path.append(node)
            current_id = node.parent_id

        return list(reversed(path))  # Root to target

    def _select_ancestor_by_agreement(self, trajectory: list[TrajectoryNode], exclude_member: str = None) -> str:
        """Select one ancestor probabilistically based on agreement level and recency.

        Lower agreement and more recent interactions = higher probability of selection.
        Recency is prioritized slightly over agreement level.

        Args:
            trajectory: The trajectory path (list of nodes from root to current)
            exclude_member: Member ID to exclude from selection (typically the last node's member)

        Returns:
            Selected ancestor member ID, or None if no valid ancestors
        """
        if len(trajectory) <= 1:  # Only root node, no ancestors to debate with
            return None

        # Map agreement levels to base weights (lower agreement = higher weight)
        agreement_weight_map = {
            AgreedLevel.fully_disagreed: 4.0,
            AgreedLevel.partially_disagreed: 3.0,
            AgreedLevel.partially_agreed: 2.0,
            AgreedLevel.fully_agreed: 1.0,
            None: 2,  # Default for root or missing agreement info
        }

        position_scores = []
        # Process nodes in reverse order to calculate recency (more recent = lower position_from_end)
        for i, node in enumerate(trajectory):
            member_id = node.member_id

            # Skip if this is the excluded member
            if exclude_member and member_id == exclude_member:
                continue

            agreed_level = node.agreed_level
            position_from_end = len(trajectory) - i

            recency_normalized = 1.0 - (position_from_end / len(trajectory))
            recency_weight = 1.0 + (recency_normalized * 4.0)  # Range: [1.0, 5.0]
            # Get agreement weight
            agreement_weight = agreement_weight_map.get(agreed_level, 2)
            score = 0.6 * recency_weight + 0.4 * agreement_weight
            position_scores.append((member_id, score))

        # If no valid candidates after exclusion, return None
        if not position_scores:
            return None

        selected = random.choices(position_scores, weights=[pos_score[1] for pos_score in position_scores], k=1)[0]
        return selected[0]

    def _create_trajectory_node(
        self,
        trajectory_graph: dict[str, TrajectoryNode],
        member_id: str,
        parent_id: str,
        iteration: int,
        debate_opinion: str,
        reasoning: str,
        agreed_level: AgreedLevel,
        opinion_shift: int = 0,
        shift_explanation: str = "",
        agreed_level_explicit: bool = False,
    ) -> tuple[TrajectoryNode, int]:
        """Create a new trajectory node and add to graph.

        Args:
            trajectory_graph: The trajectory graph to add the node to
            member_id: Member who owns this node
            parent_id: Parent node ID
            debate_opinion: Response to debate
            reasoning: LLM reasoning
            agreed_level: Agreement level
            opinion_shift: How much did opinion change (0-4)
            shift_explanation: Why did opinion shift

        Returns:
            Tuple of (new TrajectoryNode, updated timestamp)
        """
        node = TrajectoryNode(
            node_id=str(uuid.uuid4()),
            member_id=member_id,
            parent_id=parent_id,
            iteration=iteration,
            debate_opinion=debate_opinion,
            reasoning=reasoning,
            agreed_level=agreed_level,
            children_ids=[],
            opinion_shift=opinion_shift,
            shift_explanation=shift_explanation,
            agreed_level_explicit=agreed_level_explicit,
        )

        trajectory_graph[node.node_id] = node
        trajectory_graph[parent_id].children_ids.append(node.node_id)
        return node

    def _get_member_initial_thought(
        self,
        member_id: str,
        committee_task: str,
        agent_task: str,
        task_input: dict,
        task_context: dict,
    ) -> dict:
        """Get initial thought for one member (for parallel execution).

        Args:
            member_id: ID of the member
            committee_task: What should the committee accomplish?
            agent_task: What was the original agent trying to do?
            task_input: Primary information for the committee task
            task_context: Additional context

        Returns:
            Dict with member_id, initial_opinion, reasoning, and tool_trajectory (None unless tools
            are active, in which case it holds the dspy.ReAct trajectory dict).
        """
        member_lm = self.member_lm_map.get(member_id)

        if member_lm:
            with dspy.context(lm=member_lm):
                initial = self.member_initial(
                    committee_task=committee_task,
                    agent_task=agent_task,
                    task_input=task_input,
                    task_context=task_context,
                )
        else:
            raise ValueError(f"No LM found for member {member_id}")

        # ReAct predictions expose the tool-call trajectory under .trajectory; ChainOfThought does not.
        # Capture it only in tool mode. Both ReAct (via its extract ChainOfThought) and ChainOfThought
        # surface a .reasoning field, but read it defensively in case a ReAct path omits it.
        tool_trajectory = getattr(initial, "trajectory", None) if self.use_tools else None
        reasoning = getattr(initial, "reasoning", None)
        if reasoning is None:
            reasoning = ""

        return {
            "member_id": member_id,
            "initial_opinion": initial.initial_opinion,
            "reasoning": reasoning,
            "tool_trajectory": tool_trajectory,
        }

    def _process_member_debates(
        self,
        member_id: str,
        member_states: dict[str, MemberState],
        member_received_trajectories: dict[str, list[list[TrajectoryNode]]],
        trajectory_graph: dict[str, TrajectoryNode],
        iteration: int,
        committee_task: str,
        agent_task: str,
        task_input: dict,
        task_context: dict,
    ) -> dict | None:
        """Process sampled messages for one member (batched in one LLM call).

        Args:
            member_id: ID of the member
            member_states: Dict of all member states
            member_received_trajectories: Dict of all member received trajectories
            trajectory_graph: The trajectory graph
            iteration: Current iteration number
            committee_task: What should the committee accomplish?
            agent_task: What was the original agent trying to do?
            task_input: Primary information for the committee task
            task_context: Additional context

        Returns:
            Dict with member_id, new_nodes_data, aggregated_opinion, and num_processed, or None if no messages
        """
        state = member_states[member_id]
        received_trajectories = member_received_trajectories[member_id]

        # Sample max_messages_per_member messages from all received
        trajectories = random.sample(
            received_trajectories,
            k=min(self.max_messages_per_member, len(received_trajectories)),
        )

        # Build all debate threads for batched processing
        all_debate_threads = []

        for trajectory in trajectories:
            # Get recent trajectory history
            truncated_trajectory = trajectory[-self.history_window :]

            # Split into previous context and last turn
            if len(truncated_trajectory) == 1:
                # Only one turn - it's both the last turn and has no previous context
                previous_context = []
                last_turn_node = truncated_trajectory[0]
            else:
                # Multiple turns - split them
                previous_context_nodes = truncated_trajectory[:-1]
                last_turn_node = truncated_trajectory[-1]
                previous_context = [
                    DebateTurn(
                        judge_id=node.member_id,
                        argument=node.debate_opinion,
                        agreed_level=node.agreed_level,
                        reasoning=node.reasoning,
                    )
                    for node in previous_context_nodes
                ]

            # Create the last turn
            last_turn = DebateTurn(
                judge_id=last_turn_node.member_id,
                argument=last_turn_node.debate_opinion,
                agreed_level=last_turn_node.agreed_level,
                reasoning=last_turn_node.reasoning,
            )

            # Create the debate thread
            debate_thread = DebateThread(
                last_turn=last_turn,
                previous_context=previous_context,
            )

            all_debate_threads.append(debate_thread)

        # ONE LLM call for all messages
        member_lm = self.member_lm_map.get(member_id)

        with dspy.context(lm=member_lm):
            debate_result = self.member_debate(
                committee_task=committee_task,
                agent_task=agent_task,
                task_input=task_input,
                task_context=task_context,
                own_opinion=state.opinion_history[-1],
                debate_threads=all_debate_threads,
                my_id=member_id,
            )

        member_states[member_id].opinion_history.append(debate_result.refined_opinion)

        debate_opinions = debate_result.debate_opinions
        if debate_opinions is None:
            # The debate step occasionally yields no opinions (e.g. malformed LM output);
            # skip graph expansion for this member rather than crashing on a None iterable.
            warnings.warn(
                f"debate_opinions is None for member_id={member_id}; "
                f"skipping trajectory expansion. Last trajectory node: {trajectories[0][-1]}",
                stacklevel=2,
            )
            # Return the OWNING member_id explicitly. The caller must not re-derive it from
            # trajectories[0][-1].member_id: on this no-expansion path the last node still belongs
            # to the PARTNER member being responded to, which would mis-key the buffer.
            return member_id, trajectories

        # zip() truncates to the shorter of (trajectories, debate_opinions). A model that returns
        # fewer opinions than threads silently drops the tail trajectories, and flakier models drop
        # more — differential data loss that biases exactly the cross-model comparisons this study
        # makes. Count and warn so the loss is visible in logs (and quantifiable) rather than silent.
        n_threads, n_opinions = len(trajectories), len(debate_opinions)
        if n_opinions != n_threads:
            dropped = max(0, n_threads - n_opinions)
            warnings.warn(
                f"member_id={member_id}: got {n_opinions} debate_opinions for {n_threads} threads "
                f"({n_threads - n_opinions:+d}); {min(n_threads, n_opinions)} paired, "
                f"{dropped} thread(s) dropped.",
                stacklevel=2,
            )
            # Persist the drop so offline analysis can quantify differential (per-model) data loss
            # instead of it vanishing into discarded stderr. Thread-safe: _process_member_debates
            # runs in a ThreadPoolExecutor. Keyed by (member, iteration) for per-model attribution.
            if dropped:
                with self._thread_drop_lock:
                    self._thread_drops.append(
                        {"member_id": member_id, "iteration": iteration + 1,
                         "n_threads": n_threads, "n_opinions": n_opinions, "dropped": dropped}
                    )

        for i, (trajectory, debate_opinion) in enumerate(zip(trajectories, debate_opinions, strict=False)):
            last_node = trajectory[-1]
            agreed_level = debate_opinion.agreed_level
            new_node = self._create_trajectory_node(
                trajectory_graph,
                member_id,
                last_node.node_id,
                iteration + 1,
                debate_opinion.opinion,
                debate_opinion.reasoning,
                agreed_level,
                opinion_shift=debate_opinion.opinion_shift,
                shift_explanation=debate_opinion.shift_explanation,
                agreed_level_explicit=getattr(debate_opinion, "agreed_level_explicit", False),
            )
            trajectory.append(new_node)

        return member_id, trajectories

    def forward(
        self,
        committee_task: str,
        agent_task: str = "",
        task_input: dict = None,
        task_context: dict = None,
        verbose: bool = False,
    ) -> dspy.Prediction:
        """
        Primary inference method for the committee (DSPy Module pattern).

        Args:
            committee_task: What should this committee accomplish?
            agent_task: What was the original agent trying to do? (optional context)
            task_input: Primary information for the committee task
            task_context: Additional context (e.g., agent's output, constraints)
            verbose: Print debug information

        Returns:
            dspy.Prediction with final_judgement, output_accepted, member_opinions,
            trajectory_graph, and opinion shift metrics
        """
        # Handle None defaults
        if task_input is None:
            task_input = {}
        if task_context is None:
            task_context = {}
        # Initialize local state for this evaluation (stateless)
        trajectory_graph: dict[str, TrajectoryNode] = {}
        # Per-run thread-drop ledger (see _process_member_debates): records every zip-truncation
        # data-loss event so it's persisted in the result rather than lost to stderr. Re-init each
        # forward() call so the module stays stateless across questions. Lock guards the concurrent
        # debate-processing threads.
        import threading

        self._thread_drops: list[dict] = []
        self._thread_drop_lock = threading.Lock()

        if verbose:
            print("=" * 80)
            print("STARTING LLM COMMITTEE")
            print("=" * 80)
            print(f"Committee task: {committee_task}")
            print(f"Agent task: {agent_task}")

        # Step 1: Members form initial thoughts and create root trajectory nodes - PARALLEL
        if verbose:
            print(f"\n[STEP 1] {self.num_members} members forming initial thoughts [parallel]...")

        # Get all initial thoughts in parallel
        member_states = {}
        member_trajectories_buffer = {}
        member_received_trajectories = defaultdict(list)

        with ThreadPoolExecutor(max_workers=self.num_members) as executor:
            futures = [
                executor.submit(
                    self._get_member_initial_thought,
                    f"member_{i}",
                    committee_task,
                    agent_task,
                    task_input,
                    task_context,
                )
                for i in range(self.num_members)
            ]

            # Collect results and create root nodes
            for future in as_completed(futures):
                result = future.result()
                member_id = result["member_id"]

                # Create root trajectory node (parent_id=None)
                member_root_node = TrajectoryNode(
                    node_id=str(uuid.uuid4()),
                    member_id=member_id,
                    iteration=0,
                    parent_id=None,
                    debate_opinion=result["initial_opinion"],
                    reasoning=result["reasoning"],
                    agreed_level=None,
                    children_ids=[],
                    tool_trajectory=result.get("tool_trajectory"),
                )
                trajectory_graph[member_root_node.node_id] = member_root_node

                # Initialize member state
                member_states[member_id] = MemberState(
                    member_id=member_id,
                    opinion_history=[result["initial_opinion"]],
                )
                member_trajectories_buffer[member_id] = [[member_root_node]]

                if verbose:
                    print(f"  {member_id}: {result['initial_opinion'][:100]}...")

        # Step 2: Debate iterations (tree-based with controlled exploration)
        #
        # debate_timing resolves HOW MANY debate rounds actually run, without touching
        # max_iterations (so per_turn stays byte-for-byte identical to the original):
        #   per_turn -> max_iterations (unchanged loop)
        #   none     -> 0  (skip the loop entirely: initial drafts feed straight to the chairman)
        #   after    -> 1  (exactly one debate round over the independent drafts, regardless of
        #                   max_iterations)
        # The 0-round ("none") case is safe below: member_trajectories_buffer /
        # member_received_trajectories are only read INSIDE the loop, opinion_history[-1] falls back
        # to the initial opinion, and _compute_opinion_shifts skips root nodes (all nodes are roots
        # here), so it returns well-formed empty metrics.
        if self.debate_timing == "none":
            effective_iterations = 0
        elif self.debate_timing == "after":
            effective_iterations = 1
        else:  # "per_turn"
            effective_iterations = self.max_iterations

        for iteration in range(effective_iterations):
            if verbose:
                print(f"\n[STEP 2.{iteration + 1}] Debate iteration {iteration + 1}/{effective_iterations}")

            # Phase 1: Route messages from the buffer to the members
            # Growth phase: mutliple members are randomly selected
            # Convergence phase: only 1 ancestor (selected by agreement level)
            in_growth_phase = iteration < self.growth_iterations

            for member_id, member_trajectories in member_trajectories_buffer.items():
                for trajectory in member_trajectories:
                    # # Early stopping: skip routing if last response was fully agreed
                    # last_node = trajectory[-1]
                    # if last_node.agreed_level == AgreedLevel.fully_agreed:
                    #     continue

                    if in_growth_phase:
                        # Growth phase: randomly select members_to_explore members to route the message to
                        member_idx = int(member_id[len("member_") :])
                        other_members = self._member_ids[:member_idx] + self._member_ids[member_idx + 1 :]
                        targets = random.sample(
                            other_members,
                            k=min(self.members_to_explore, len(other_members)),
                        )
                        for target_id in targets:
                            member_received_trajectories[target_id].append(trajectory.copy())
                    else:
                        # Convergence phase: select ancestor but exclude the member who just responded
                        last_node = trajectory[-1]
                        target_id = self._select_ancestor_by_agreement(trajectory, exclude_member=last_node.member_id)
                        # Only route if a valid target was found
                        if target_id:
                            member_received_trajectories[target_id].append(trajectory.copy())
                        elif verbose:
                            # This shouldn't happen in convergence phase - log for debugging
                            print(
                                f"  WARNING: No valid target for trajectory from {member_id} "
                                f"(length: {len(trajectory)})"
                            )

            if verbose:
                total_messages = sum(len(trajectories) for trajectories in member_received_trajectories.values())
                print(f"  Routed {total_messages} messages to {len(member_received_trajectories)} members")

            # Phase 2: Sample and process messages in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        self._process_member_debates,
                        mid,
                        member_states,
                        member_received_trajectories,
                        trajectory_graph,
                        iteration,
                        committee_task,
                        agent_task,
                        task_input,
                        task_context,
                    )
                    for mid in member_states.keys()
                    if member_received_trajectories[mid]  # Only if has messages
                ]

                # Clear the buffer and add the new trajectories to the buffer
                member_trajectories_buffer = defaultdict(list)
                for future in as_completed(futures):
                    # _process_member_debates returns (owning_member_id, trajectories). Use the
                    # explicit member_id rather than re-deriving from the last node's owner, which
                    # is wrong on the no-expansion path (last node belongs to the partner member).
                    member_id, trajectories = future.result()
                    if trajectories:
                        member_trajectories_buffer[member_id] = trajectories

                # Clear the received trajectories for next iteration
                member_received_trajectories = defaultdict(list)

            if verbose:
                print(f"  Finished iteration {iteration + 1}/{effective_iterations}!")
                print(f"  Total nodes in graph: {len(trajectory_graph)}")

        # Step 3: Aggregate final judgment from all refined opinions
        if verbose:
            print("\n[STEP 3] Aggregating committee members' refined opinions...")

        # Collect all members' final refined opinions from current nodes
        member_opinions = []
        for member_id, state in member_states.items():
            member_opinions.append(MemberOpinion(member_id=member_id, opinion=state.opinion_history[-1]))

            if verbose:
                print(f"  {member_id}: {state.opinion_history[-1][:100]}...")

        # Chairman aggregates all opinions into final judgment
        if verbose:
            print("\n[STEP 4] Chairman synthesizing final judgment...")

        with dspy.context(lm=self.chairman_lm):
            final_result = self.aggregation(
                committee_task=committee_task,
                agent_task=agent_task,
                task_input=task_input,
                task_context=task_context,
                member_opinions=member_opinions,
            )

        # Prepare result as dspy.Prediction
        if verbose:
            print("\n" + "=" * 80)
            print("FINAL RESULT")
            print("=" * 80)
            print(f"Judgment: {final_result.final_judgement[:200]}...")
            print(f"Trajectory nodes created: {len(trajectory_graph)}")
            print("=" * 80)

        # Compute opinion shift metrics
        shift_data = self._compute_opinion_shifts(trajectory_graph)

        return dspy.Prediction(
            final_judgement=final_result.final_judgement,
            output_accepted=final_result.output_accepted,
            member_opinions=[{"member_id": mo.member_id, "opinion": mo.opinion} for mo in member_opinions],
            # Actual debate rounds run under debate_timing. Equals max_iterations for the default
            # "per_turn" mode (behavior-preserving); 0 for "none"; 1 for "after".
            num_iterations=effective_iterations,
            trajectory_graph=trajectory_graph,
            num_nodes=len(trajectory_graph),
            opinion_shifts=shift_data["opinion_shifts"],
            convergence_metrics=shift_data["convergence_metrics"],
            thread_drops=list(self._thread_drops),
        )

    def get_leaf_nodes(self, trajectory_graph: dict[str, TrajectoryNode]) -> list[TrajectoryNode]:
        """Get all leaf nodes (endpoints) in the trajectory tree.

        Leaf nodes are nodes with no children - they represent the final positions
        in each debate trajectory.

        Args:
            trajectory_graph: The trajectory graph to analyze

        Returns:
            List of leaf TrajectoryNode objects
        """
        leaf_nodes = []
        for node_id, node in trajectory_graph.items():
            if len(node.children_ids) == 0:  # No children = leaf node
                leaf_nodes.append(node)
        return leaf_nodes

    def format_trajectory_as_json(self, trajectory_graph: dict[str, TrajectoryNode], leaf_node: TrajectoryNode) -> dict:
        """Format a complete trajectory path as readable JSON.

        Traces from root to the given leaf node and formats as a structured
        conversation showing debate evolution.

        Args:
            trajectory_graph: The trajectory graph
            leaf_node: The leaf node to trace back from

        Returns:
            Dict with trajectory metadata and turn-by-turn debate
        """
        # Get the path from root to leaf
        path = self._get_trajectory_path(trajectory_graph, leaf_node.node_id)

        # Format as structured conversation
        trajectory_data = {
            "member_id": leaf_node.member_id,
            "total_turns": len(path),
            "final_iteration": leaf_node.iteration,
            "turns": [],
        }

        for i, node in enumerate(path):
            turn = {
                "turn_number": i,
                "iteration": node.iteration,
                "member_id": node.member_id,
                "opinion": node.debate_opinion,
                "reasoning": node.reasoning,
                "agreed_level": node.agreed_level.value if node.agreed_level else None,
                "opinion_shift": node.opinion_shift,
                "shift_explanation": node.shift_explanation,
                "node_id": node.node_id,
                "parent_id": node.parent_id,
            }
            trajectory_data["turns"].append(turn)

        return trajectory_data

    def _compute_opinion_shifts(self, trajectory_graph: dict[str, TrajectoryNode]) -> dict:
        shifts_by_member = defaultdict(list)
        shift_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Count shifts by level
        members_who_shifted = set()

        for node_id, node in trajectory_graph.items():
            if node.parent_id is None:  # Skip root nodes (no shift for initial opinions)
                continue

            # Extract shift info from node
            opinion_shift = node.opinion_shift
            shifts_by_member[node.member_id].append(
                {
                    "iteration": node.iteration,
                    "shift": opinion_shift,
                    "explanation": node.shift_explanation,
                }
            )

            # Track shift statistics
            shift_counts[opinion_shift] += 1
            if opinion_shift > 0:  # Any shift (1-4)
                members_who_shifted.add(node.member_id)

        return {
            "opinion_shifts": dict(shifts_by_member),
            "convergence_metrics": {
                "shift_counts": shift_counts,  # {0: n, 1: n, 2: n, 3: n, 4: n}
                "total_shifts": sum(shift_counts[i] for i in range(1, 5)),
                "significant_shifts": shift_counts[3] + shift_counts[4],  # Level 3-4
                "members_who_shifted": sorted(members_who_shifted),
            },
        }
