"""MAPPO update logic wired for COLA modules."""

from typing import Dict, List

import torch
import torch.nn.functional as F

from cola_framework.interfaces.updater import UpdateModule


class MAPPOUpdater(UpdateModule):
    """Coordinates one full PPO update for COLA + MAPPO.

    Update flow per call:
      1. Consensus builder forward + student optimizer step + EMA teacher update.
      2. Recompute discrete consensus embeddings (detached) with the updated student.
      3. For n_epochs: shuffle rollout timesteps into n_minibatches, then for
         each minibatch and each agent:
           - Evaluate new log-prob and entropy under the current actor.
           - PPO clipped surrogate actor loss + entropy bonus.
           - Value function MSE loss against TD-λ returns.
           - Gradient clipping + optimizer steps.

    All dependencies are injected so any module can be swapped independently.
    """

    def __init__(
        self,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        actors: List[torch.nn.Module],
        value_fns: List[torch.nn.Module],
        opt_cb: torch.optim.Optimizer,
        opt_actors: List[torch.optim.Optimizer],
        opt_values: List[torch.optim.Optimizer],
        n_epochs: int = 10,
        n_minibatches: int = 4,
        clip_eps: float = 0.2,
        c_value: float = 0.5,
        c_entropy: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        n_agents = len(actors)
        if n_agents == 0:
            raise ValueError("actors list cannot be empty.")
        if len(value_fns) != n_agents:
            raise ValueError("value_fns length must match actors length.")
        if len(opt_actors) != n_agents or len(opt_values) != n_agents:
            raise ValueError("optimizer lists must match number of agents.")

        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.actors = actors
        self.value_fns = value_fns
        self.opt_cb = opt_cb
        self.opt_actors = opt_actors
        self.opt_values = opt_values
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.clip_eps = clip_eps
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.max_grad_norm = max_grad_norm

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """Run one full PPO update cycle.

        batch keys (all tensors with leading dim T = n_rollout_steps):
            obs          : [T, n_agents, obs_dim]
            state        : [T, state_dim]
            actions      : [T, n_agents, action_dim]
            advantages   : [T, n_agents]
            returns      : [T, n_agents]
            old_log_probs: [T, n_agents]
        """
        obs = batch["obs"]
        state = batch["state"]
        actions = batch["actions"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_log_probs = batch["old_log_probs"]

        T, n_agents, _ = obs.shape
        # For very short partial rollouts reduce n_minibatches to avoid empty splits.
        n_mb = min(self.n_minibatches, T)

        # ── 1. Consensus builder update ──────────────────────────────────────
        loss_cb, consensus = self.consensus_builder(obs)
        self.opt_cb.zero_grad()
        loss_cb.backward()
        self.opt_cb.step()
        self.consensus_builder.ema_update_teacher()

        consensus_agreement = (
            (consensus == consensus[:, :1]).all(dim=1).float().mean().item()
        )

        # ── 2. Recompute embeddings with updated student (no RL grad) ────────
        with torch.no_grad():
            emb = self.embedding_layer(consensus)           # [T, n_agents, emb_dim]
        emb_dim = emb.shape[-1]
        all_emb = emb.reshape(T, n_agents * emb_dim)        # [T, n_agents * emb_dim]

        # ── 3. PPO epochs ────────────────────────────────────────────────────
        actor_losses_all: List[List[float]] = [[] for _ in range(n_agents)]
        value_losses_all: List[List[float]] = [[] for _ in range(n_agents)]

        for _ in range(self.n_epochs):
            perm = torch.randperm(T, device=obs.device)
            mb_size = T // n_mb

            for start in range(0, n_mb * mb_size, mb_size):
                idx = perm[start : start + mb_size]

                obs_mb = obs[idx]                       # [mb, n_agents, obs_dim]
                state_mb = state[idx]                   # [mb, state_dim]
                act_mb = actions[idx]                   # [mb, n_agents, action_dim]
                adv_mb = advantages[idx]                # [mb, n_agents]
                ret_mb = returns[idx]                   # [mb, n_agents]
                old_lp_mb = old_log_probs[idx]          # [mb, n_agents]
                emb_mb = emb[idx]                       # [mb, n_agents, emb_dim]
                all_emb_mb = all_emb[idx]               # [mb, n_agents * emb_dim]

                for a in range(n_agents):
                    # ── Actor update (PPO clip + entropy) ──────────────────
                    new_lp_a, entropy_a = self.actors[a].evaluate_actions(
                        obs_mb[:, a, :],
                        emb_mb[:, a, :],   # already detached (computed with no_grad)
                        act_mb[:, a, :],
                    )

                    # Per-minibatch advantage normalization for stability.
                    adv_a = adv_mb[:, a]
                    adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)

                    ratio = torch.exp(new_lp_a - old_lp_mb[:, a])
                    surr1 = ratio * adv_a
                    surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_a
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_bonus = -self.c_entropy * entropy_a.mean()
                    actor_total = actor_loss + entropy_bonus

                    self.opt_actors[a].zero_grad()
                    actor_total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actors[a].parameters(), self.max_grad_norm
                    )
                    self.opt_actors[a].step()
                    actor_losses_all[a].append(float(actor_total.item()))

                    # ── Value function update (MSE) ─────────────────────────
                    new_val_a = self.value_fns[a](state_mb, all_emb_mb)  # [mb, 1]
                    value_loss = F.mse_loss(new_val_a.squeeze(-1), ret_mb[:, a])

                    self.opt_values[a].zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.value_fns[a].parameters(), self.max_grad_norm
                    )
                    self.opt_values[a].step()
                    value_losses_all[a].append(float(value_loss.item()))

        def _mean(lst):
            return float(sum(lst) / len(lst)) if lst else 0.0

        return {
            "loss_cb": float(loss_cb.item()),
            "consensus_agreement": float(consensus_agreement),
            "actor_losses": [_mean(actor_losses_all[a]) for a in range(n_agents)],
            "value_losses": [_mean(value_losses_all[a]) for a in range(n_agents)],
        }
