"""Abstract contract for history encoder modules."""

from abc import ABC, abstractmethod

import torch


class HistoryEncoderModule(ABC):
    """Maps an observation sequence to a fixed-size context vector.

    Implementations must accept tensors with shape [B, T, obs_dim] and
    return [B, out_dim].
    """

    obs_dim: int
    out_dim: int

    @abstractmethod
    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Encode observation history sequence into a context vector."""
