"""Abstract contract for training update modules."""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class UpdateModule(ABC):
    """Consumes a sampled batch and performs one optimization step."""

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """Run one update and return scalar metrics."""
