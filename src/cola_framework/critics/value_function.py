"""Centralized state-value function for MAPPO-style on-policy training."""

import torch
import torch.nn as nn

from cola_framework.interfaces.value import ValueModule


class CentralizedValueFunction(ValueModule, nn.Module):
    """Per-agent centralized value function V(s, all_emb).

    Unlike the Q-critic used in MADDPG (which also takes actions as input),
    the value function maps the global state and all consensus embeddings to a
    scalar state value. There are no actions in the input, which is the standard
    for on-policy actor-critic (PPO/MAPPO).
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        emb_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.emb_dim = emb_dim

        input_dim = state_dim + n_agents * emb_dim
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
    ) -> torch.Tensor:
        """Estimate scalar state value from centralized inputs.

        state            : [B, state_dim] or [state_dim]
        all_consensus_emb: [B, n_agents * emb_dim] or [n_agents * emb_dim]
        returns          : [B, 1] or [1]
        """
        if state.ndim != all_consensus_emb.ndim:
            raise ValueError("state and all_consensus_emb must have the same rank.")
        if state.ndim not in (1, 2):
            raise ValueError("Inputs must be rank-1 or rank-2 tensors.")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state last dim must be {self.state_dim}, got {state.shape[-1]}.")
        if all_consensus_emb.shape[-1] != self.n_agents * self.emb_dim:
            raise ValueError(
                f"all_consensus_emb last dim must be {self.n_agents * self.emb_dim}, "
                f"got {all_consensus_emb.shape[-1]}."
            )

        x = torch.cat([state, all_consensus_emb], dim=-1)
        return self.net(x)
