"""History-aware DINO-style consensus builder.

This module keeps the same external API as the existing consensus builder:
`forward`, `infer`, and `ema_update_teacher`.
"""

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cola_framework.interfaces.consensus import ConsensusModule
from cola_framework.interfaces.history_encoder import HistoryEncoderModule


class HistoryAwareConsensusBuilder(nn.Module, ConsensusModule):
    """Consensus builder that uses observation history windows.

    Input contract:
    - forward(obs_seq_batch): [B, n_agents, T, obs_dim]
    - infer(obs_seq): [n_agents, T, obs_dim]

    Output contract remains identical to the original builder.
    """

    def __init__(
        self,
        encoder: HistoryEncoderModule,
        k: int = 4,
        mlp_hidden: int = 64,
        tau_teacher: float = 0.04,
        tau_student: float = 0.1,
        ema_teacher: float = 0.996,
        ema_center: float = 0.9,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an nn.Module implementing HistoryEncoderModule.")
        if not hasattr(encoder, "out_dim"):
            raise ValueError("encoder must expose out_dim.")

        self.k = k
        self.tau_teacher = tau_teacher
        self.tau_student = tau_student
        self.ema_teacher = ema_teacher
        self.ema_center = ema_center

        # Student branch: history encoder + projection head.
        self.student_encoder = encoder
        self.student_head = nn.Sequential(
            nn.Linear(int(encoder.out_dim), mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, k),
        )

        # Teacher branch: EMA copy, strictly gradient-free.
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        for param in self.teacher_encoder.parameters():
            param.requires_grad_(False)
        for param in self.teacher_head.parameters():
            param.requires_grad_(False)

        self.register_buffer("center", torch.zeros(k, dtype=torch.float32))

    def _student_logits(self, obs_seq: torch.Tensor) -> torch.Tensor:
        features = self.student_encoder(obs_seq)
        return self.student_head(features)

    @torch.no_grad()
    def _teacher_logits(self, obs_seq: torch.Tensor) -> torch.Tensor:
        features = self.teacher_encoder(obs_seq)
        return self.teacher_head(features)

    def forward(self, obs_seq_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute consensus loss and labels for sequence inputs.

        obs_seq_batch: [B, n_agents, T, obs_dim]
        returns:
            loss_cb: scalar tensor, gradients to student branch only
            consensus: [B, n_agents] int64 labels
        """
        if obs_seq_batch.ndim != 4:
            raise ValueError("obs_seq_batch must have shape [B, n_agents, T, obs_dim].")

        batch_size, n_agents, t_len, obs_dim = obs_seq_batch.shape

        # Flatten agent axis to encode all agent windows in one vectorized call.
        flat_seq = obs_seq_batch.reshape(batch_size * n_agents, t_len, obs_dim)

        student_logits = self._student_logits(flat_seq).reshape(batch_size, n_agents, self.k)
        with torch.no_grad():
            teacher_logits = self._teacher_logits(flat_seq).reshape(batch_size, n_agents, self.k)

        student_probs = F.softmax(student_logits / self.tau_student, dim=-1)

        # Center teacher logits before softmax to reduce class-collapse risk.
        centered_teacher = teacher_logits - self.center.view(1, 1, self.k)
        teacher_probs = F.softmax(centered_teacher / self.tau_teacher, dim=-1)

        # Pairwise DINO-style cross-entropy over all agent pairs (a != b).
        log_student_probs = torch.log(student_probs + 1e-8)
        pairwise_dot = torch.einsum("bak,bck->bac", teacher_probs.detach(), log_student_probs)
        agent_mask = (~torch.eye(n_agents, dtype=torch.bool, device=obs_seq_batch.device)).float()
        loss_cb = -(pairwise_dot * agent_mask.view(1, n_agents, n_agents)).mean(dim=0).sum()

        with torch.no_grad():
            # Update center as EMA over teacher logits (not probabilities).
            batch_center = teacher_logits.reshape(-1, self.k).mean(dim=0)
            self.center.mul_(self.ema_center).add_((1.0 - self.ema_center) * batch_center)

        consensus = student_probs.detach().argmax(dim=-1)
        return loss_cb, consensus

    @torch.no_grad()
    def infer(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Infer per-agent consensus labels for one environment step.

        obs_seq: [n_agents, T, obs_dim]
        returns: [n_agents] int64
        """
        if obs_seq.ndim != 3:
            raise ValueError("obs_seq must have shape [n_agents, T, obs_dim].")
        probs = F.softmax(self._student_logits(obs_seq) / self.tau_student, dim=-1)
        return probs.argmax(dim=-1)

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        """EMA update for teacher branch; call after student optimizer.step()."""
        momentum = self.ema_teacher
        for s_p, t_p in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            t_p.data.mul_(momentum).add_((1.0 - momentum) * s_p.data)
        for s_p, t_p in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            t_p.data.mul_(momentum).add_((1.0 - momentum) * s_p.data)
