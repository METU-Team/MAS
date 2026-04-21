"""Abstract environment contract for COLA-compatible MARL environments."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class MultiAgentEnvironment(ABC):
    """Defines a swappable environment interface for the framework."""

    n_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset environment and return local observations and global state proxy."""

    @abstractmethod
    def step(self, action_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step environment with joint action and return next_obs, next_state, rewards, dones."""
