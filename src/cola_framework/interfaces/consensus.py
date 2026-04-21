"""Abstract contract for consensus builder modules."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class ConsensusModule(ABC):
    """Defines the API used by training and acting components."""

    @abstractmethod
    def forward(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return consensus loss and per-agent consensus labels for a batch."""

    @abstractmethod
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """Infer consensus labels for one timestep without gradient tracking."""

    @abstractmethod
    def ema_update_teacher(self) -> None:
        """Update teacher parameters from student parameters using EMA."""
