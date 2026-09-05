"""
Trajectory analysis utilities for LLM Committee.

Provides visualization, statistics, and persistence for trajectory trees.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import dspy

from llm_committee.committee_sync import AgreedLevel, TrajectoryNode


def get_trajectory_path(trajectory_graph: dict[str, TrajectoryNode], node_id: str) -> list[TrajectoryNode]:
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


def get_member_trajectory(trajectory_graph: dict[str, TrajectoryNode], node_id: str) -> list[TrajectoryNode]:
    """Get the full trajectory path for a specific node.

    Args:
        trajectory_graph: The trajectory graph containing the nodes
        node_id: The node ID to get trajectory for

    Returns:
        List of TrajectoryNode objects from root to current position
    """
    return get_trajectory_path(trajectory_graph, node_id)


def visualize_trajectory_tree(trajectory_graph: dict[str, TrajectoryNode], node_id: str) -> str:
    """Create a text visualization of a trajectory path.

    Args:
        trajectory_graph: The trajectory graph containing the nodes
        node_id: The node ID to visualize path for

    Returns:
        Text tree visualization
    """
    trajectory = get_trajectory_path(trajectory_graph, node_id)
    member_id = trajectory[0].member_id if trajectory else "unknown"

    lines = [f"Trajectory for {member_id}:", "=" * 60]

    for i, node in enumerate(trajectory):
        indent = "  " * i
        prefix = "└─ " if i > 0 else ""

        lines.append(f"{indent}{prefix}[Iteration {node.iteration}]")
        lines.append(f"{indent}   Opinion: {node.debate_opinion[:80]}...")

        if node.agreed_level:
            lines.append(f"{indent}   Agreement: {node.agreed_level.value}")

    return "\n".join(lines)


def visualize_trajectory_graph(trajectory_graph: dict[str, TrajectoryNode], output_path: str | Path = None) -> None:
    """Create a graphical visualization of the full trajectory tree.

    Args:
        trajectory_graph: The trajectory graph containing the nodes
        output_path: Optional path to save the visualization (PNG/PDF). If None, displays interactively.

    Requires matplotlib and networkx to be installed.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("Please install matplotlib and networkx: pip install matplotlib networkx")
        return

    # Build networkx graph
    G = nx.DiGraph()

    # Color map for members
    member_ids = list(set(n.member_id for n in trajectory_graph.values()))
    colors = plt.cm.tab10(range(len(member_ids)))
    member_colors = {mid: colors[i] for i, mid in enumerate(member_ids)}

    # Add nodes and edges
    node_colors = []
    labels = {}
    for node_id, node in trajectory_graph.items():
        G.add_node(node_id)
        node_colors.append(member_colors[node.member_id])
        labels[node_id] = f"{node.member_id[-1]}:{node.iteration}"

        if node.parent_id:
            G.add_edge(node.parent_id, node_id)

    # Create layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        labels=labels,
        node_size=500,
        font_size=8,
        arrows=True,
        arrowsize=10,
    )

    # Legend
    for mid, color in member_colors.items():
        plt.scatter([], [], c=[color], label=mid, s=100)
    plt.legend(loc="upper left", fontsize=8)

    plt.title("Trajectory Tree Visualization")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def get_trajectory_statistics(trajectory_graph: dict[str, TrajectoryNode]) -> dict:
    """Get statistics about the trajectory graph.

    Args:
        trajectory_graph: The trajectory graph containing the nodes

    Returns:
        Dict with trajectory statistics
    """
    # Count nodes by type
    root_nodes = sum(1 for n in trajectory_graph.values() if n.parent_id is None)
    debate_nodes = sum(1 for n in trajectory_graph.values() if n.parent_id is not None)

    # Count by agreement level
    agreement_counts = dict.fromkeys(AgreedLevel, 0)
    for node in trajectory_graph.values():
        if node.agreed_level:
            agreement_counts[node.agreed_level] += 1

    # Calculate depth statistics (depth = distance from root)
    all_depths = []
    for node in trajectory_graph.values():
        depth = len(get_trajectory_path(trajectory_graph, node.node_id)) - 1
        all_depths.append(depth)

    # Calculate branching factor
    branching_factors = [len(n.children_ids) for n in trajectory_graph.values() if n.children_ids]

    return {
        "total_nodes": len(trajectory_graph),
        "root_nodes": root_nodes,
        "debate_nodes": debate_nodes,
        "agreement_counts": {k.value: v for k, v in agreement_counts.items()},
        "avg_depth": sum(all_depths) / len(all_depths) if all_depths else 0,
        "max_depth": max(all_depths) if all_depths else 0,
        "avg_branching_factor": sum(branching_factors) / len(branching_factors) if branching_factors else 0,
    }


def save_trajectory(
    trajectory_graph: dict[str, TrajectoryNode],
    filepath: str | Path,
    result: dict = None,
    committee_params: dict = None,
) -> None:
    """Save trajectory graph to JSON file.

    Args:
        trajectory_graph: The trajectory graph to save
        filepath: Path to save JSON file
        result: Optional result dict from evaluate() containing additional metadata
        committee_params: Optional dict of committee parameters
    """
    filepath = Path(filepath)

    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert TrajectoryNode objects to dicts
    serialized_graph = {}
    for node_id, node in trajectory_graph.items():
        node_dict = asdict(node)
        # Convert AgreedLevel enum to string for JSON serialization
        if node_dict.get("agreed_level"):
            node_dict["agreed_level"] = (
                node_dict["agreed_level"].value
                if hasattr(node_dict["agreed_level"], "value")
                else node_dict["agreed_level"]
            )
        serialized_graph[node_id] = node_dict

    # Build complete save data
    save_data = {
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "version": "0.2.0",
            "committee_params": committee_params or {},
        },
        "trajectory_graph": serialized_graph,
    }

    # Add result data if provided
    if result:
        save_data.update(
            {
                "final_judgement": result.get("final_judgement"),
                "reasoning": result.get("reasoning"),
                "member_opinions": result.get("member_opinions"),
                "output_accepted": result.get("output_accepted"),
                "num_nodes": result.get("num_nodes"),
                "opening": result.get("opening"),
                "num_iterations": result.get("num_iterations"),
            }
        )

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"Saved trajectory to {filepath}")
    print(f"  - {len(serialized_graph)} nodes")
    if result:
        print(f"  - {len(result.get('member_opinions', []))} members")


def load_trajectory(filepath: str | Path) -> dict:
    """Load saved trajectory graph from JSON file.

    Args:
        filepath: Path to saved JSON file

    Returns:
        Dict with loaded results including trajectory_graph with TrajectoryNode objects
    """
    filepath = Path(filepath)

    with open(filepath, encoding="utf-8") as f:
        save_data = json.load(f)

    # Reconstruct TrajectoryNode objects from dicts
    trajectory_graph = {}
    for node_id, node_dict in save_data["trajectory_graph"].items():
        # Convert agreed_level string back to enum
        if node_dict.get("agreed_level"):
            node_dict["agreed_level"] = AgreedLevel(node_dict["agreed_level"])
        trajectory_graph[node_id] = TrajectoryNode(**node_dict)

    # Build result dict
    result = {
        "final_judgement": save_data.get("final_judgement"),
        "reasoning": save_data.get("reasoning"),
        "member_opinions": save_data.get("member_opinions"),
        "output_accepted": save_data.get("output_accepted"),
        "trajectory_graph": trajectory_graph,
        "num_nodes": save_data.get("num_nodes"),
        "opening": save_data.get("opening"),
        "num_iterations": save_data.get("num_iterations"),
        "metadata": save_data["metadata"],
    }

    print(f"Loaded trajectory from {filepath}")
    print(f"  - Saved at: {save_data['metadata']['saved_at']}")
    print(f"  - {len(trajectory_graph)} nodes")
    if result.get("member_opinions"):
        print(f"  - {len(result['member_opinions'])} members")

    return result


class TrajectoryTurnAnalysisSignature(dspy.Signature):
    """Analyze what new information or insights emerged at a specific debate turn.

    You are analyzing a multi-agent debate trajectory to understand the information
    dynamics - what new perspectives, evidence, or reasoning emerged at each turn.

    NOTE: This signature is only used for turns 1+. Turn 0 (initial opinion) is
    skipped since there's no previous context to measure information gain against.

    Focus on identifying:
    1. **New Information**: Facts, evidence, or considerations not mentioned in previous turns
    2. **New Perspectives**: Different angles or framings of the problem
    3. **New Arguments**: Novel reasoning or logical connections
    4. **Redundancy**: Information that was already covered in previous turns

    Be specific about what is genuinely NEW versus what is repetition or rephrasing.
    """

    task_description: str = dspy.InputField(
        desc="What task was the committee working on? (provides context for analysis)"
    )
    previous_turns: str = dspy.InputField(
        desc="All previous debate turns concatenated (for reference). May be empty for turn 0."
    )
    current_turn: dict = dspy.InputField(
        desc=(
            "The current turn to analyze (contains: turn_number, member_id, "
            "opinion, reasoning, agreed_level, opinion_shift)"
        )
    )

    new_information: str = dspy.OutputField(
        desc=(
            "What genuinely NEW information, evidence, or facts emerged in this turn "
            "that were not in previous turns? Be specific. If nothing new, say 'No new information.'"
        )
    )
    new_perspectives: str = dspy.OutputField(
        desc=(
            "What NEW perspectives, angles, or framings emerged? How does this turn "
            "view the problem differently? If nothing new, say 'No new perspectives.'"
        )
    )
    new_arguments: str = dspy.OutputField(
        desc=(
            "What NEW reasoning, logical connections, or arguments emerged? "
            "Be specific about the novel logical steps. If nothing new, say 'No new arguments.'"
        )
    )
    redundancy_assessment: str = dspy.OutputField(
        desc=(
            "What parts of this turn are redundant or repetitive compared to previous turns? "
            "Be honest about overlap."
        )
    )
    information_gain_score: int = dspy.OutputField(
        desc=(
            "On a scale of 0-10, how much NEW information did this turn contribute? "
            "(0=completely redundant, 10=highly novel insights)"
        )
    )


class TrajectoryAnalyzer(dspy.Module):
    """Analyzes debate trajectories to identify information gain at each turn.

    This module uses an LLM to qualitatively analyze what new information,
    perspectives, or arguments emerged at each debate turn, helping understand
    the information dynamics of multi-agent debate.
    """

    def __init__(self, analysis_lm, output_path: str = None):
        """Initialize the trajectory analyzer.

        Args:
            analysis_lm: LM to anlayze the debate trajectories
            output_path: Optional file path to save markdown report (default: None)
        """
        super().__init__()
        self.analysis_lm = analysis_lm
        self.output_path = output_path
        self.analyzer = dspy.ChainOfThought(TrajectoryTurnAnalysisSignature)

    def forward(
        self,
        trajectory_graph: dict,
        committee,
        task_description: str,
        verbose: bool = False,
    ) -> dict:
        """Analyze all trajectories in the graph and optionally save to markdown.

        Args:
            trajectory_graph: The trajectory graph from committee result
            committee: LLMCommitteeSync instance (to access helper methods)
            task_description: Description of what the committee was working on
            verbose: Print analysis results

        Returns:
            Dict with analyses for all trajectories and aggregate statistics
        """
        # Get all leaf nodes (complete trajectories)
        leaf_nodes = committee.get_leaf_nodes(trajectory_graph)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Analyzing {len(leaf_nodes)} complete trajectories")
            print(f"{'='*80}")

        # Analyze each trajectory
        all_trajectory_analyses = []

        for i, leaf_node in enumerate(leaf_nodes):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Trajectory {i+1}: Member {leaf_node.member_id}")
                print(f"{'='*80}")

            # Format trajectory as JSON
            trajectory_json = committee.format_trajectory_as_json(trajectory_graph, leaf_node)

            # Analyze this trajectory
            trajectory_analysis = self._analyze_single_trajectory(trajectory_json, task_description, verbose=verbose)

            all_trajectory_analyses.append(trajectory_analysis)

        # Calculate cross-trajectory statistics
        cross_stats = self._calculate_cross_trajectory_stats(all_trajectory_analyses)

        # Generate interpretation based on overall average gain
        avg_gain = cross_stats["overall_average_gain"]
        if avg_gain < 4:
            interpretation = (
                "⚠️  Low information gain suggests high redundancy\n"
                "   → Members may be repeating similar arguments without adding new information"
            )
        elif avg_gain > 7:
            interpretation = (
                "✓ High information gain suggests diverse, novel contributions\n"
                "   → Members are bringing genuinely new perspectives at each turn"
            )
        else:
            interpretation = "→ Moderate information gain - some novelty but also redundancy"

        # Flatten structure for easier access
        result = {
            "task_description": task_description,
            "total_trajectories": len(all_trajectory_analyses),
            "trajectory_analyses": all_trajectory_analyses,
            # Flatten cross_trajectory_stats to top level for easier access
            "overall_average_gain": cross_stats["overall_average_gain"],
            "overall_high_gain_turns": cross_stats["total_high_gain_turns"],
            "overall_low_gain_turns": cross_stats["total_low_gain_turns"],
            "all_scores": cross_stats["all_scores"],
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to markdown if path provided
        if self.output_path:
            self._save_markdown_report(result, self.output_path)
            if verbose:
                print(f"\n{'='*80}")
                print(f"Report saved to: {self.output_path}")
                print(f"{'='*80}")

        return result

    def _analyze_single_trajectory(self, trajectory_data: dict, task_description: str, verbose: bool = False) -> dict:
        """Analyze a single trajectory's information gain."""
        turns = trajectory_data["turns"]
        analyses = []

        for i, turn in enumerate(turns):
            # Turn 0 is the initial opinion - no previous context to compare against
            if i == 0:
                turn_analysis = {
                    "turn_number": 0,
                    "member_id": turn["member_id"],
                    "new_information": "N/A - Initial opinion",
                    "new_perspectives": "N/A - Initial opinion",
                    "new_arguments": "N/A - Initial opinion",
                    "redundancy_assessment": "N/A - Initial opinion",
                    "information_gain_score": None,  # Not a number for turn 0
                }
                analyses.append(turn_analysis)

                if verbose:
                    print(f"\n{'='*80}")
                    print(f"Turn 0 Analysis (by {turn['member_id']})")
                    print(f"{'='*80}")
                    print("Skipping analysis - this is the initial opinion (baseline)")

                continue

            # Build previous turns context for turns 1+
            previous_turns = "\n\n".join(
                [
                    f"Turn {j} (by {t['member_id']}):\n" f"Opinion: {t['opinion']}\n" f"Reasoning: {t['reasoning']}"
                    for j, t in enumerate(turns[:i])
                ]
            )

            with dspy.context(lm=self.analysis_lm):
                result = self.analyzer(
                    task_description=task_description,
                    previous_turns=previous_turns,
                    current_turn=turn,
                )

            turn_analysis = {
                "turn_number": i,
                "member_id": turn["member_id"],
                "new_information": result.new_information,
                "new_perspectives": result.new_perspectives,
                "new_arguments": result.new_arguments,
                "redundancy_assessment": result.redundancy_assessment,
                "information_gain_score": result.information_gain_score,
            }
            analyses.append(turn_analysis)

            if verbose:
                print(f"\n{'='*80}")
                print(f"Turn {i} Analysis (by {turn['member_id']})")
                print(f"{'='*80}")
                print(f"New Information: {result.new_information}")
                print(f"New Perspectives: {result.new_perspectives}")
                print(f"New Arguments: {result.new_arguments}")
                print(f"Redundancy: {result.redundancy_assessment}")
                print(f"Information Gain Score: {result.information_gain_score}/10")

        # Calculate aggregate statistics for this trajectory (excluding turn 0)
        scored_analyses = [a for a in analyses if a["information_gain_score"] is not None]
        total_scored_turns = len(scored_analyses)
        avg_info_gain = (
            sum(a["information_gain_score"] for a in scored_analyses) / total_scored_turns
            if total_scored_turns > 0
            else 0
        )
        high_gain_turns = sum(1 for a in scored_analyses if a["information_gain_score"] >= 7)
        low_gain_turns = sum(1 for a in scored_analyses if a["information_gain_score"] <= 3)

        return {
            "trajectory_id": f"{trajectory_data['member_id']}_trajectory",
            "member_id": trajectory_data["member_id"],
            "total_turns": len(analyses),
            "turn_analyses": analyses,
            "aggregate_stats": {
                "average_information_gain": avg_info_gain,
                "high_gain_turns": high_gain_turns,  # Score >= 7
                "low_gain_turns": low_gain_turns,  # Score <= 3
                "information_gain_scores": [a["information_gain_score"] for a in scored_analyses],
            },
        }

    def _calculate_cross_trajectory_stats(self, all_analyses: list[dict]) -> dict:
        """Calculate statistics across all trajectories."""
        all_scores = []
        for analysis in all_analyses:
            all_scores.extend(analysis["aggregate_stats"]["information_gain_scores"])

        if not all_scores:
            return {
                "overall_average_gain": 0,
                "total_high_gain_turns": 0,
                "total_low_gain_turns": 0,
                "all_scores": [],
            }

        avg_overall = sum(all_scores) / len(all_scores)
        total_high_gain = sum(a["aggregate_stats"]["high_gain_turns"] for a in all_analyses)
        total_low_gain = sum(a["aggregate_stats"]["low_gain_turns"] for a in all_analyses)

        return {
            "overall_average_gain": avg_overall,
            "total_high_gain_turns": total_high_gain,
            "total_low_gain_turns": total_low_gain,
            "all_scores": all_scores,
        }

    def _save_markdown_report(self, result: dict, output_path: str):
        """Save analysis results as a formatted markdown report."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            # Header
            f.write("# Trajectory Analysis Report\n\n")
            f.write(f"**Generated:** {result['timestamp']}\n\n")
            f.write(f"**Task:** {result['task_description']}\n\n")
            f.write(f"**Total Trajectories:** {result['total_trajectories']}\n\n")
            f.write("---\n\n")

            # Overall Statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Overall Average Information Gain:** {result['overall_average_gain']:.2f}/10\n")
            f.write(f"- **Total High-Gain Turns (≥7):** {result['overall_high_gain_turns']}\n")
            f.write(f"- **Total Low-Gain Turns (≤3):** {result['overall_low_gain_turns']}\n\n")

            # Interpretation
            f.write("### Interpretation\n\n")
            f.write(f"{result['interpretation']}\n\n")

            f.write("---\n\n")

            # Per-Trajectory Analysis
            f.write("## Per-Trajectory Analysis\n\n")

            for traj_analysis in result["trajectory_analyses"]:
                traj_id = traj_analysis["trajectory_id"]
                member_id = traj_analysis["member_id"]
                traj_stats = traj_analysis["aggregate_stats"]

                f.write(f"### Trajectory: {member_id}\n\n")
                f.write(f"- **Total Turns:** {traj_analysis['total_turns']}\n")
                f.write(f"- **Average Information Gain:** {traj_stats['average_information_gain']:.2f}/10\n")
                f.write(f"- **High-Gain Turns:** {traj_stats['high_gain_turns']}\n")
                f.write(f"- **Low-Gain Turns:** {traj_stats['low_gain_turns']}\n")
                f.write(f"- **Score Progression:** {traj_stats['information_gain_scores']}\n\n")

                # Turn-by-turn breakdown
                f.write("#### Turn-by-Turn Breakdown\n\n")
                for turn_analysis in traj_analysis["turn_analyses"]:
                    turn_num = turn_analysis["turn_number"]
                    score = turn_analysis["information_gain_score"]

                    # Format score display (N/A for turn 0)
                    score_display = "N/A" if score is None else f"{score}/10"
                    f.write(f"**Turn {turn_num}** (Score: {score_display})\n\n")
                    f.write(f"- **New Information:** {turn_analysis['new_information']}\n")
                    f.write(f"- **New Perspectives:** {turn_analysis['new_perspectives']}\n")
                    f.write(f"- **New Arguments:** {turn_analysis['new_arguments']}\n")
                    f.write(f"- **Redundancy:** {turn_analysis['redundancy_assessment']}\n\n")

                f.write("---\n\n")

            # Footer with raw JSON
            f.write("## Raw Data\n\n")
            f.write("Complete analysis data in JSON format:\n\n")
            f.write("```json\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n```\n")
