"""DINO-style consensus builder for cooperative MARL."""

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cola_framework.interfaces.consensus import ConsensusModule


class ConsensusBuilder(nn.Module, ConsensusModule):
    """Learns a shared discrete label from local observations.

    The student is optimized by gradient descent, while the teacher is an
    exponential moving average of the student. Centering is applied to teacher
    logits to reduce representation collapse.
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
        self.k = k
        self.tau_teacher = tau_teacher
        self.tau_student = tau_student
        self.ema_teacher = ema_teacher
        self.ema_center = ema_center

        # Optional teacher-temperature warmup: start with a *softer* (higher) temp
        # so early teacher targets are not over-sharp (a known DINO collapse
        # trigger), then linearly anneal to the paper's tau_teacher. Disabled when
        # warmup_start <= 0 or warmup_steps <= 0 (then tau_teacher is constant).
        self.tau_teacher_warmup_start = tau_teacher_warmup_start
        self.tau_teacher_warmup_steps = tau_teacher_warmup_steps
        self.register_buffer("_update_count", torch.zeros((), dtype=torch.long))

        self.student = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k),
        )

        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            # Teacher must stay gradient-free; it is updated by EMA only.
            param.requires_grad_(False)

        self.register_buffer("center", torch.zeros(k, dtype=torch.float32))

    def _current_tau_teacher(self) -> float:
        """Effective teacher temperature, applying the optional warmup schedule."""
        if self.tau_teacher_warmup_start <= 0.0 or self.tau_teacher_warmup_steps <= 0:
            return self.tau_teacher
        t = float(self._update_count.item())
        frac = min(1.0, t / float(self.tau_teacher_warmup_steps))
        return self.tau_teacher_warmup_start + frac * (self.tau_teacher - self.tau_teacher_warmup_start)

    def _student_logits(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.student(obs_batch)

    @torch.no_grad()
    def _teacher_logits(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.teacher(obs_batch)

    def forward(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute consensus loss and labels.

        obs_batch: [B, n_agents, obs_dim]
        returns:
            loss_cb: scalar tensor, gradients flow to student only
            consensus: [B, n_agents] int64 labels from student predictions
        """
        if obs_batch.ndim != 3:
            raise ValueError("obs_batch must have shape [B, n_agents, obs_dim].")

        batch_size, n_agents, obs_dim = obs_batch.shape
        flat_obs = obs_batch.reshape(batch_size * n_agents, obs_dim)

        student_logits = self._student_logits(flat_obs).reshape(batch_size, n_agents, self.k)
        with torch.no_grad():
            teacher_logits = self._teacher_logits(flat_obs).reshape(batch_size, n_agents, self.k)

        student_probs = F.softmax(student_logits / self.tau_student, dim=-1)

        # Center teacher logits before softmax to reduce collapse to one class.
        tau_teacher = self._current_tau_teacher()
        centered_teacher = teacher_logits - self.center.view(1, 1, self.k)
        teacher_probs = F.softmax(centered_teacher / tau_teacher, dim=-1)

        # Pairwise cross-entropy over all agent pairs (a != b):
        # H(P_T(z^a), P_S(z^b)) = -sum_k P_T^k log P_S^k.
        log_student_probs = torch.log(student_probs + 1e-8)
        pairwise_dot = torch.einsum("bak,bck->bac", teacher_probs.detach(), log_student_probs)
        agent_mask = (~torch.eye(n_agents, dtype=torch.bool, device=obs_batch.device)).float()
        loss_cb = -(pairwise_dot * agent_mask.view(1, n_agents, n_agents)).mean(dim=0).sum()

        with torch.no_grad():
            # Update center from raw teacher logits as an EMA running mean.
            batch_center = teacher_logits.reshape(-1, self.k).mean(dim=0)
            self.center.mul_(self.ema_center).add_((1.0 - self.ema_center) * batch_center)
            # Advance the warmup clock (one training forward == one update).
            self._update_count += 1

        consensus = student_probs.detach().argmax(dim=-1)
        return loss_cb, consensus

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """Infer consensus labels for one timestep.

        obs: [n_agents, obs_dim]
        returns: [n_agents] int64
        """
        if obs.ndim != 2:
            raise ValueError("obs must have shape [n_agents, obs_dim].")
        probs = F.softmax(self.student(obs) / self.tau_student, dim=-1)
        return probs.argmax(dim=-1)

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        """EMA update for teacher weights, called after student optimizer.step()."""
        m = self.ema_teacher
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(m).add_((1.0 - m) * student_param.data)
