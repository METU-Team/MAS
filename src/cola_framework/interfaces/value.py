"""Abstract contract for centralized value function modules (MAPPO / on-policy)."""

from abc import ABC, abstractmethod

import torch


class ValueModule(ABC):
    """Maps centralized training inputs to a scalar state value."""

    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        all_consensus_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return value tensor with shape [B, 1] or [1]."""
