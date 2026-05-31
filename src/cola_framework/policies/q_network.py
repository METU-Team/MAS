"""Per-agent Q-network for discrete-action QMIX training."""

import torch
import torch.nn as nn

from cola_framework.interfaces.q_network import QNetworkModule


class QNetwork(QNetworkModule, nn.Module):
    """Maps local observation and consensus embedding to per-action Q-values.

    Architecture mirrors Actor/CentralizedValueFunction: a 2-layer ReLU MLP
    with a linear output head producing one Q-value per discrete action.
    No activation is applied to the output (Q-values can be any real number).
    """

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor, consensus_emb: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all discrete actions.

        obs          : [B, obs_dim] or [obs_dim]
        consensus_emb: [B, emb_dim] or [emb_dim]
        returns      : [B, n_actions] or [n_actions]
        """
        if obs.ndim != consensus_emb.ndim:
            raise ValueError("obs and consensus_emb must have the same rank.")
        if obs.ndim not in (1, 2):
            raise ValueError("obs and consensus_emb must be rank-1 or rank-2 tensors.")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs last dim must be {self.obs_dim}, got {obs.shape[-1]}.")
        if consensus_emb.shape[-1] != self.emb_dim:
            raise ValueError(f"consensus_emb last dim must be {self.emb_dim}.")

        return self.net(torch.cat([obs, consensus_emb], dim=-1))
