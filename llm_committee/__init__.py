"""
LLM Committee: Multi-agent debate system for collaborative inference-time scaling.

This package provides a committee-based system where LLM agents debate and
refine their opinions through structured discussion with tree-based trajectory
tracking, opinion shift analysis, and qualitative information gain assessment.
"""

from llm_committee.__version__ import __version__

__all__ = ["LLMCommitteeSync", "TrajectoryAnalyzer", "__version__"]


def __getattr__(name):
    # The new offline planner/CLI does not need DSPy, Databricks, or legacy clients.
    # Preserve the existing top-level imports without eagerly loading them.
    if name == "LLMCommitteeSync":
        from llm_committee.committee_sync import LLMCommitteeSync

        return LLMCommitteeSync
    if name == "TrajectoryAnalyzer":
        from llm_committee.trajectory_utils import TrajectoryAnalyzer

        return TrajectoryAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
