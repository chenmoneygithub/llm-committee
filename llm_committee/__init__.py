"""
LLM Committee: Multi-agent debate system for collaborative inference-time scaling.

This package provides a committee-based system where LLM agents debate and
refine their opinions through structured discussion with tree-based trajectory
tracking, opinion shift analysis, and qualitative information gain assessment.
"""

from llm_committee.__version__ import __version__
from llm_committee.committee_sync import LLMCommitteeSync
from llm_committee.trajectory_utils import TrajectoryAnalyzer

__all__ = ["LLMCommitteeSync", "TrajectoryAnalyzer", "__version__"]
