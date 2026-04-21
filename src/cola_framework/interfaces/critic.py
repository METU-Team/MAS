"""Abstract contract for centralized critic modules."""

from abc import ABC, abstractmethod

import torch


class CriticModule(ABC):
    """Maps centralized training inputs to scalar Q-values."""

    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        all_consensus_emb: torch.Tensor,
        all_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q-value tensor with shape [B, 1] or [1]."""
