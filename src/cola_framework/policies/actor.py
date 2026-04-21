"""Actor network used by each agent in MADDPG-style COLA."""

import torch
import torch.nn as nn

from cola_framework.interfaces.policy import PolicyModule


class Actor(nn.Module, PolicyModule):
    """Per-agent actor network.

    Input: local observation concatenated with consensus embedding.
    Output: bounded continuous action in [-1, 1] via tanh.
    """

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, consensus_emb: torch.Tensor) -> torch.Tensor:
        """Compute bounded action from local inputs.

        obs: [B, obs_dim] or [obs_dim]
        consensus_emb: [B, emb_dim] or [emb_dim]
        returns: [B, action_dim] or [action_dim]
        """
        if obs.ndim != consensus_emb.ndim:
            raise ValueError("obs and consensus_emb must have the same rank.")
        if obs.ndim not in (1, 2):
            raise ValueError("obs and consensus_emb must be rank-1 or rank-2 tensors.")

        if obs.shape[-1] != self.obs_dim:
            raise ValueError("Last dim of obs must match obs_dim.")
        if consensus_emb.shape[-1] != self.emb_dim:
            raise ValueError("Last dim of consensus_emb must match emb_dim.")

        # Keep module flexible for acting and training: single sample or batch.
        x = torch.cat([obs, consensus_emb], dim=-1)
        return self.net(x)
