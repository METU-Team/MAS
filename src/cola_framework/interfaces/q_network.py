"""Abstract contract for per-agent Q-network modules (QMIX / discrete actions)."""

from abc import ABC, abstractmethod

import torch


class QNetworkModule(ABC):
    """Maps local observation and consensus embedding to per-action Q-values."""

    @abstractmethod
    def forward(self, obs: torch.Tensor, consensus_emb: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all discrete actions.

        obs          : [B, obs_dim] or [obs_dim]
        consensus_emb: [B, emb_dim] or [emb_dim]
        returns      : [B, n_actions] or [n_actions]
        """
