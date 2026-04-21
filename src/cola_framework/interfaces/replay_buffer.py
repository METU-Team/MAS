"""Abstract replay buffer contract for off-policy MARL."""

from abc import ABC, abstractmethod
from typing import Dict

import torch


class ExperienceReplay(ABC):
    """Defines a swappable replay buffer API."""

    @abstractmethod
    def push(
        self,
        obs,
        state,
        actions,
        rewards,
        next_obs,
        next_state,
        dones,
    ) -> None:
        """Store one transition."""

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a minibatch."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of stored transitions."""
