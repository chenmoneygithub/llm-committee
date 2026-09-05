"""Basic usage example for LLM Committee.

Runs a small 3-member committee on a simple judge task using Databricks-served
``opus-4.8`` for both members and the chairman, then prints trajectory statistics and
saves the trajectory graph to disk for post-hoc analysis.

Run with the project venv::

    .venv/bin/python examples/basic_usage.py
"""

import dspy

from llm_committee import LLMCommitteeSync
from llm_committee.databricks_lm import make_lm
from llm_committee.trajectory_utils import (
    get_trajectory_statistics,
    save_trajectory,
    visualize_trajectory_tree,
)


def main() -> None:
    """Run a basic committee judgement on a single question/answer pair."""
    # One LM instance per member plus one for the chairman. All point at the same
    # Databricks endpoint here, but member_lms accepts a heterogeneous list.
    member_lms = [make_lm("opus-4.8", max_tokens=2048) for _ in range(3)]
    chairman_lm = make_lm("opus-4.8", max_tokens=2048)

    # Configure a global LM as a fallback (members/chairman use their own instances).
    dspy.settings.configure(lm=chairman_lm)

    # Build the committee.
    committee = LLMCommitteeSync(
        num_members=3,  # Committee size (must match len(member_lms)).
        member_lms=member_lms,  # One LM per member.
        chairman_lm=chairman_lm,  # Chairman aggregates the final judgement.
        max_iterations=2,  # Debate rounds.
        members_to_explore=2,  # Members to route each trajectory to during growth.
        growth_iterations=1,  # Iterations to grow before converging.
        max_messages_per_member=2,  # Max messages processed per member per round.
        history_window=3,  # Send last 3 turns of a trajectory to the LLM.
    )

    # The committee's job, and the inputs it reasons over.
    committee_task = (
        "Decide whether the agent's answer correctly and helpfully responds to the user's question. "
        "Accept the answer only if it is factually correct and addresses the question."
    )
    task_input = {
        "question": "What is the capital of France, and roughly how many people live there?",
    }
    task_context = {
        "agent_answer": (
            "The capital of France is Paris, and its metropolitan area is home to "
            "around 11-12 million people."
        ),
    }

    # Run the committee with the current public API (forward / __call__).
    result = committee(
        committee_task=committee_task,
        agent_task="Answer the user's geography question.",
        task_input=task_input,
        task_context=task_context,
        verbose=True,
    )

    # Display results. `result` is a dspy.Prediction - read attributes, not dict keys.
    print("\n" + "=" * 80)
    print("COMMITTEE RESULTS")
    print("=" * 80)
    print(f"\nFinal Judgement: {result.final_judgement[:300]}...")
    print(f"Output Accepted: {result.output_accepted}")
    print(f"\nTotal nodes created: {result.num_nodes}")

    # Detailed statistics over the trajectory graph (a dict[node_id -> TrajectoryNode]).
    trajectory_graph = result.trajectory_graph
    stats = get_trajectory_statistics(trajectory_graph)
    print("\nTrajectory Statistics:")
    print(f"  Root nodes: {stats['root_nodes']}")
    print(f"  Debate nodes: {stats['debate_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Avg branching factor: {stats['avg_branching_factor']:.2f}")
    print(f"  Agreement counts: {stats['agreement_counts']}")

    # Visualize one member's trajectory (pick the first root node).
    root_nodes = [n for n in trajectory_graph.values() if n.parent_id is None]
    if root_nodes:
        print("\n" + "=" * 80)
        print(visualize_trajectory_tree(trajectory_graph, root_nodes[0].node_id))

    # Save the trajectory for post-analysis. save_trajectory expects a plain dict of
    # result metadata, so adapt the Prediction's attributes into one.
    result_meta = {
        "final_judgement": result.final_judgement,
        "output_accepted": result.output_accepted,
        "member_opinions": result.member_opinions,
        "num_nodes": result.num_nodes,
        "num_iterations": result.num_iterations,
    }
    print("\n" + "=" * 80)
    save_trajectory(trajectory_graph, "saved_results/committee_trajectory.json", result_meta)
    print("=" * 80)


if __name__ == "__main__":
    main()
