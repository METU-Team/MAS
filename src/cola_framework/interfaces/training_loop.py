"""Abstract contract for training loop orchestrators."""

from abc import ABC, abstractmethod
from typing import Dict


class TrainingLoopModule(ABC):
    """Coordinates environment interaction and optimization updates."""

    @abstractmethod
    def run(self) -> Dict[str, object]:
        """Execute training and return run statistics."""
