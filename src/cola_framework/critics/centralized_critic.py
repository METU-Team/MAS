"""Centralized critic used by MADDPG-style COLA training."""

import torch
import torch.nn as nn

from cola_framework.interfaces.critic import CriticModule


class Critic(nn.Module, CriticModule):
    """Per-agent centralized critic.

    Input combines global state proxy, all consensus embeddings, and all actions.
    Output is a scalar Q-value.
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        emb_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.emb_dim = emb_dim
        self.action_dim = action_dim

        input_dim = state_dim + n_agents * emb_dim + n_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        all_consensus_emb: torch.Tensor,
        all_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values from centralized information.

        state: [B, state_dim] or [state_dim]
        all_consensus_emb: [B, n_agents * emb_dim] or [n_agents * emb_dim]
        all_actions: [B, n_agents * action_dim] or [n_agents * action_dim]
        returns: [B, 1] or [1]
        """
        if state.ndim != all_consensus_emb.ndim or state.ndim != all_actions.ndim:
            raise ValueError("state, all_consensus_emb, and all_actions must have same rank.")
        if state.ndim not in (1, 2):
            raise ValueError("Inputs must be rank-1 or rank-2 tensors.")

        if state.shape[-1] != self.state_dim:
            raise ValueError("Last dim of state must match state_dim.")
        if all_consensus_emb.shape[-1] != self.n_agents * self.emb_dim:
            raise ValueError("Last dim of all_consensus_emb must be n_agents * emb_dim.")
        if all_actions.shape[-1] != self.n_agents * self.action_dim:
            raise ValueError("Last dim of all_actions must be n_agents * action_dim.")

        # Centralized training view: concatenate all shared context and actions.
        x = torch.cat([state, all_consensus_emb, all_actions], dim=-1)
        return self.net(x)
