"""Abstract contract for policy evaluation modules."""

from abc import ABC, abstractmethod
from typing import Dict


class EvaluatorModule(ABC):
    """Runs policy evaluation and returns aggregated metrics."""

    @abstractmethod
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """Evaluate current policies and return summary metrics."""
