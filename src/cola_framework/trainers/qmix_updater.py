"""QMIX update logic wired for COLA modules."""

import copy
from typing import Dict, List

import torch
import torch.nn.functional as F

from cola_framework.interfaces.updater import UpdateModule


def _polyak(online: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Soft Polyak update: target ← τ * online + (1 − τ) * target."""
    for src, dst in zip(online.parameters(), target.parameters()):
        dst.data.mul_(1.0 - tau).add_(tau * src.data)


class QMIXUpdater(UpdateModule):
    """Coordinates one QMIX update step for COLA + QMIX.

    Update flow per call:
      1. Consensus builder forward + student optimizer step + EMA teacher update.
      2. Recompute embeddings (detached) with updated student.
      3. Compute per-agent Q-values for taken actions; mix with MixingNetwork.
      4. Compute double-Q-style TD target with target networks.
      5. MSE loss on Q_tot; gradient clip; optimizer step.
      6. Soft Polyak update of target Q-networks and target mixing network.

    All parameters (Q-networks + mixing network) share a single optimizer,
    matching the original QMIX paper's end-to-end training scheme.
    """

    def __init__(
        self,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        q_networks: List[torch.nn.Module],
        target_q_networks: List[torch.nn.Module],
        mixing_network: torch.nn.Module,
        target_mixing_network: torch.nn.Module,
        opt_cb: torch.optim.Optimizer,
        opt_qmix: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau_polyak: float = 0.005,
        max_grad_norm: float = 10.0,
    ) -> None:
        n_agents = len(q_networks)
        if n_agents == 0:
            raise ValueError("q_networks list cannot be empty.")
        if len(target_q_networks) != n_agents:
            raise ValueError("target_q_networks length must match q_networks.")

        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.q_networks = q_networks
        self.target_q_networks = target_q_networks
        self.mixing_network = mixing_network
        self.target_mixing_network = target_mixing_network
        self.opt_cb = opt_cb
        self.opt_qmix = opt_qmix
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.max_grad_norm = max_grad_norm

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """Run one QMIX update.

        batch keys:
            obs       : [B, n_agents, obs_dim]
            state     : [B, state_dim]
            actions   : [B, n_agents, 1] float — discrete action indices
            rewards   : [B, n_agents]
            next_obs  : [B, n_agents, obs_dim]
            next_state: [B, state_dim]
            dones     : [B, n_agents]
        """
        obs = batch["obs"]
        state = batch["state"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        next_state = batch["next_state"]
        dones = batch["dones"]

        B, n_agents, _ = obs.shape
        # Recover integer action indices from float storage.
        action_idx = actions[:, :, 0].long()   # [B, n_agents]

        # ── 1. Consensus builder update ──────────────────────────────────────
        loss_cb, consensus = self.consensus_builder(obs)
        self.opt_cb.zero_grad()
        loss_cb.backward()
        self.opt_cb.step()
        self.consensus_builder.ema_update_teacher()

        consensus_agreement = (
            (consensus == consensus[:, :1]).all(dim=1).float().mean().item()
        )

        with torch.no_grad():
            flat_c = consensus.reshape(-1)
            k = self.consensus_builder.k
            counts = torch.zeros(k, device=flat_c.device, dtype=torch.float32)
            for i in range(k):
                counts[i] = (flat_c == i).sum().float()
            probs = counts / counts.sum().clamp(min=1.0)
            consensus_entropy = -(probs * (probs + 1e-8).log()).sum().item()

        # ── 2. Embeddings (detached — RL loss must not reach CB) ─────────────
        with torch.no_grad():
            emb = self.embedding_layer(consensus)           # [B, n_agents, emb_dim]
            next_consensus = torch.stack(
                [self.consensus_builder.infer(next_obs[b]) for b in range(B)], dim=0
            )
            next_emb = self.embedding_layer(next_consensus)  # [B, n_agents, emb_dim]

        # ── 3. Online Q-values for taken actions ─────────────────────────────
        chosen_qs = []
        for a in range(n_agents):
            q_all = self.q_networks[a](obs[:, a, :], emb[:, a, :])  # [B, n_actions]
            q_a = q_all.gather(-1, action_idx[:, a : a + 1])         # [B, 1]
            chosen_qs.append(q_a)
        chosen_qs = torch.cat(chosen_qs, dim=-1)  # [B, n_agents]

        # Mix: Q_tot from online networks and current state.
        q_tot = self.mixing_network(chosen_qs, state)  # [B, 1]

        # ── 4. TD target with target networks ────────────────────────────────
        with torch.no_grad():
            target_max_qs = []
            for a in range(n_agents):
                tq = self.target_q_networks[a](next_obs[:, a, :], next_emb[:, a, :])
                target_max_qs.append(tq.max(dim=-1, keepdim=True)[0])
            target_max_qs = torch.cat(target_max_qs, dim=-1)        # [B, n_agents]
            q_tot_target = self.target_mixing_network(target_max_qs, next_state)

            # Team reward and done: share one reward across agents (cooperative).
            team_reward = rewards.mean(dim=-1, keepdim=True)          # [B, 1]
            team_done = dones.any(dim=-1, keepdim=True).float()       # [B, 1]
            y = team_reward + self.gamma * q_tot_target * (1.0 - team_done)

        # ── 5. QMIX loss ─────────────────────────────────────────────────────
        loss_qmix = F.mse_loss(q_tot, y)

        self.opt_qmix.zero_grad()
        loss_qmix.backward()
        qmix_params = (
            [p for qn in self.q_networks for p in qn.parameters()]
            + list(self.mixing_network.parameters())
        )
        torch.nn.utils.clip_grad_norm_(qmix_params, self.max_grad_norm)
        self.opt_qmix.step()

        # ── 6. Soft target updates ────────────────────────────────────────────
        for qn, tqn in zip(self.q_networks, self.target_q_networks):
            _polyak(qn, tqn, self.tau_polyak)
        _polyak(self.mixing_network, self.target_mixing_network, self.tau_polyak)

        return {
            "loss_cb": float(loss_cb.item()),
            "loss_qmix": float(loss_qmix.item()),
            "consensus_agreement": float(consensus_agreement),
            "consensus_entropy": float(consensus_entropy),
        }
