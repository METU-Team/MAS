"""QMIX monotonic mixing network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    """Combines per-agent Q-values into a joint Q_tot using a hypernetwork.

    The mixing is guaranteed to be monotonically non-decreasing in each
    individual Q-value, satisfying the IGM (Individual-Global-Max) condition
    required for decentralised greedy action selection to be globally optimal.

    Monotonicity is enforced by passing hypernetwork weight outputs through
    abs() before using them as mixing weights.

    Reference: QMIX (Rashid et al., 2018).
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mix_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mix_hidden_dim = mix_hidden_dim

        # Hypernetworks generate mixing weights from global state.
        # W1: state → [n_agents × mix_hidden]  (abs → non-negative)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mix_hidden_dim)
        # b1: state → [mix_hidden]              (no abs; bias can be negative)
        self.hyper_b1 = nn.Linear(state_dim, mix_hidden_dim)
        # W2: state → [mix_hidden × 1]           (abs → non-negative)
        self.hyper_w2 = nn.Linear(state_dim, mix_hidden_dim)
        # b2: deeper head for expressiveness
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mix_hidden_dim),
            nn.ReLU(),
            nn.Linear(mix_hidden_dim, 1),
        )

    def forward(
        self,
        individual_qs: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Mix per-agent Q-values into a joint Q_tot.

        individual_qs: [B, n_agents] — Q-values for the taken action per agent
        state        : [B, state_dim] — global state for hypernetwork conditioning
        returns      : [B, 1]
        """
        B = individual_qs.shape[0]

        # Generate mixing weights from state.
        W1 = self.hyper_w1(state).abs().reshape(B, self.n_agents, self.mix_hidden_dim)
        b1 = self.hyper_b1(state).reshape(B, 1, self.mix_hidden_dim)
        W2 = self.hyper_w2(state).abs().reshape(B, self.mix_hidden_dim, 1)
        b2 = self.hyper_b2(state).reshape(B, 1, 1)

        # Forward pass through the two-layer monotonic mixing network.
        x = individual_qs.unsqueeze(1)           # [B, 1, n_agents]
        hidden = F.elu(torch.bmm(x, W1) + b1)   # [B, 1, mix_hidden]
        q_tot = torch.bmm(hidden, W2) + b2       # [B, 1, 1]
        return q_tot.squeeze(-1)                  # [B, 1]
