"""Gaussian stochastic actor for on-policy MAPPO-style training."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from cola_framework.interfaces.policy import PolicyModule


class GaussianActor(PolicyModule, nn.Module):
    """Per-agent stochastic actor with a squashed Gaussian policy.

    The output action is bounded to [-1, 1] via a tanh transformation. Log
    probabilities include the Jacobian correction for the tanh squashing so
    that they are consistent whether computed during sampling (rollout) or
    re-evaluation (PPO update).

    forward()          → mean action (deterministic, used for evaluation)
    sample()           → (action, log_prob) for rollout collection
    evaluate_actions() → (log_prob, entropy) for PPO ratio computation
    """

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim

        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Learned log standard deviation — shared across the batch dimension.
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def _distribution(self, obs: torch.Tensor, consensus_emb: torch.Tensor) -> Normal:
        x = torch.cat([obs, consensus_emb], dim=-1)
        mean = self.mean_net(x)
        std = self.log_std.exp().clamp(min=1e-6).expand_as(mean)
        return Normal(mean, std)

    # ── PolicyModule interface ───────────────────────────────────────────────

    def forward(self, obs: torch.Tensor, consensus_emb: torch.Tensor) -> torch.Tensor:
        """Return mean (deterministic) action bounded to [-1, 1].

        Used for evaluation without exploration noise.

        obs          : [B, obs_dim] or [obs_dim]
        consensus_emb: [B, emb_dim] or [emb_dim]
        returns      : [B, action_dim] or [action_dim]
        """
        return torch.tanh(self._distribution(obs, consensus_emb).mean)

    # ── MAPPO-specific methods ───────────────────────────────────────────────

    def sample(
        self,
        obs: torch.Tensor,
        consensus_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log probability.

        Uses reparameterized sampling so that gradients can flow through the
        action if needed. The log probability includes the tanh Jacobian
        correction so it is consistent with evaluate_actions().

        obs          : [obs_dim] — single timestep (no batch dim)
        consensus_emb: [emb_dim]
        returns      :
            action   : [action_dim] in [-1, 1]
            log_prob : scalar
        """
        dist = self._distribution(obs, consensus_emb)
        u = dist.rsample()                   # pre-tanh sample
        action = torch.tanh(u)               # squashed to [-1, 1]
        log_prob = dist.log_prob(u).sum(-1)  # sum over action dimensions
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        consensus_emb: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability and entropy for given stored actions.

        This is called during PPO updates where actions were collected under
        the old policy. The inverse tanh (atanh) recovers the pre-squash
        action so that the Gaussian log-prob can be evaluated exactly.

        obs          : [B, obs_dim]
        consensus_emb: [B, emb_dim]
        actions      : [B, action_dim] in [-1, 1] — stored from rollout
        returns      :
            log_prob : [B]
            entropy  : [B]
        """
        dist = self._distribution(obs, consensus_emb)

        # Clamp slightly inside (-1, 1) to keep atanh numerically stable.
        u = torch.atanh(actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = dist.log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1.0 - actions.pow(2) + 1e-6).sum(-1)

        # Entropy of the Gaussian base distribution (exact analytic form).
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy
