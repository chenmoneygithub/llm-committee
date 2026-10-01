"""Branch-isolated experiment engine for the September 2026 protocol.

The legacy DSPy committee and evaluators remain available under their old imports.
This package never creates clients, reads credentials, or calls a model on import.
"""

from .models import PilotConfig, Question, Screen
from .planning import plan_question

__all__ = ["PilotConfig", "Question", "Screen", "plan_question"]
