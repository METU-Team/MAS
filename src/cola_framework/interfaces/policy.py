"""Abstract contract for policy (actor) modules."""

from abc import ABC, abstractmethod

import torch


class PolicyModule(ABC):
    """Maps local observation plus consensus embedding to an action."""

    @abstractmethod
    def forward(self, obs: torch.Tensor, consensus_emb: torch.Tensor) -> torch.Tensor:
        """Return action tensor in the algorithm's action space."""
