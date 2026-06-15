"""Shuffled-label consensus builder — control ablation for COLA.

Runs the *real* ConsensusBuilder internally (so the consensus loss still trains
a genuine view-invariant representation), but permutes the resulting labels
across the agent axis within each batch element before handing them to the RL
update. This breaks the alignment between an agent's observation and its label
while preserving the marginal label distribution. Comparing this against the
real builder isolates the value of the *correct per-agent assignment* versus
merely having a well-distributed set of labels in the batch.
"""

from typing import Tuple

import torch
import torch.nn as nn

from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.interfaces.consensus import ConsensusModule


class ShuffledLabelConsensusBuilder(nn.Module, ConsensusModule):
    """Delegates to a real ConsensusBuilder, then shuffles labels per batch element.

    The inner builder is trained normally (its ``loss_cb`` is passed through
    untouched and its student receives gradients). Only the *content* delivered
    to the actor/critic is corrupted, via a fresh random permutation of the agent
    axis for every batch element.

    ``self.student`` and ``self.k`` are exposed so the existing
    ``Adam(builder.student.parameters())`` wiring and the updater's entropy
    computation work without special-casing.
    """

    def __init__(
        self,
        obs_dim: int,
        k: int = 4,
        hidden_dim: int = 64,
        tau_teacher: float = 0.04,
        tau_student: float = 0.1,
        ema_teacher: float = 0.996,
        ema_center: float = 0.9,
        tau_teacher_warmup_start: float = 0.0,
        tau_teacher_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.inner = ConsensusBuilder(
            obs_dim=obs_dim,
            k=k,
            hidden_dim=hidden_dim,
            tau_teacher=tau_teacher,
            tau_student=tau_student,
            ema_teacher=ema_teacher,
            ema_center=ema_center,
            tau_teacher_warmup_start=tau_teacher_warmup_start,
            tau_teacher_warmup_steps=tau_teacher_warmup_steps,
        )
        self.k = k
        # Expose the inner student so opt_cb = Adam(builder.student.parameters())
        # optimizes the real consensus network unchanged.
        self.student = self.inner.student

    @staticmethod
    def _shuffle_agents(labels: torch.Tensor) -> torch.Tensor:
        """Independently permute the agent axis of each batch element.

        labels: [B, n_agents] int64
        returns: [B, n_agents] int64
        """
        B, n = labels.shape
        # One independent permutation per batch element, stacked into [B, n].
        perms = torch.stack(
            [torch.randperm(n, device=labels.device) for _ in range(B)],
            dim=0,
        )
        return torch.gather(labels, 1, perms)

    def forward(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the inner CB, return its loss and per-batch-shuffled labels.

        obs_batch: [B, n_agents, obs_dim]
        returns:
            loss_cb: the inner builder's loss (unchanged, so the CB keeps learning)
            consensus: [B, n_agents] int64, inner labels permuted along the agent axis
        """
        if obs_batch.ndim != 3:
            raise ValueError("obs_batch must have shape [B, n_agents, obs_dim].")
        loss_cb, labels = self.inner(obs_batch)
        shuffled = self._shuffle_agents(labels)
        return loss_cb, shuffled

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """Infer inner labels for one timestep, then permute across agents.

        obs: [n_agents, obs_dim]
        returns: [n_agents] int64
        """
        if obs.ndim != 2:
            raise ValueError("obs must have shape [n_agents, obs_dim].")
        labels = self.inner.infer(obs)
        perm = torch.randperm(labels.shape[0], device=labels.device)
        return labels[perm]

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        """Forward the EMA update to the inner builder's teacher."""
        self.inner.ema_update_teacher()
