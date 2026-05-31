"""Null consensus builder — MADDPG baseline without COLA signal."""

from typing import Tuple

import torch
import torch.nn as nn

from cola_framework.interfaces.consensus import ConsensusModule


class NullConsensusBuilder(nn.Module, ConsensusModule):
    """Drop-in replacement for ConsensusBuilder that provides no coordination signal.

    Agents receive a constant zero-class embedding regardless of observation,
    so any improvement over this baseline is attributable solely to COLA.

    The dummy student Linear keeps the same opt_cb = Adam(builder.student.parameters())
    wiring in train_cola.py working without any special-casing.
    """

    def __init__(self, k: int = 4) -> None:
        super().__init__()
        self.k = k
        # Dummy parameter so Adam(self.student.parameters()) never gets an empty list.
        # It is never used in a way that affects RL loss — its gradient is always 0.
        self.student = nn.Linear(1, 1, bias=False)

    def forward(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero loss and all-zero consensus labels.

        obs_batch: [B, n_agents, obs_dim]
        returns:
            loss_cb: scalar 0.0 (connected to dummy parameter so backward() succeeds)
            consensus: [B, n_agents] int64 zeros
        """
        if obs_batch.ndim != 3:
            raise ValueError("obs_batch must have shape [B, n_agents, obs_dim].")
        B, n, _ = obs_batch.shape
        # Multiply by 0 so gradient through dummy param is exactly 0,
        # but backward() on this tensor does not raise.
        loss_cb = self.student.weight.sum() * 0.0
        consensus = torch.zeros(B, n, dtype=torch.int64, device=obs_batch.device)
        return loss_cb, consensus

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """Return all-zero consensus labels for one timestep.

        obs: [n_agents, obs_dim]
        returns: [n_agents] int64 zeros
        """
        if obs.ndim != 2:
            raise ValueError("obs must have shape [n_agents, obs_dim].")
        return torch.zeros(obs.shape[0], dtype=torch.int64, device=obs.device)

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        """No-op — no teacher network exists in the null builder."""
