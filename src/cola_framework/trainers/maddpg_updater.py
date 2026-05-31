"""Step 7: MADDPG update logic wired for COLA modules."""

from typing import Dict, List

import torch
import torch.nn.functional as F

from cola_framework.interfaces.updater import UpdateModule


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


class MADDPGUpdater(UpdateModule):
    """Coordinates one full training update for COLA + MADDPG.

    Dependencies are injected so each module can be swapped independently.
    """

    def __init__(
        self,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        actors: List[torch.nn.Module],
        critics: List[torch.nn.Module],
        target_actors: List[torch.nn.Module],
        target_critics: List[torch.nn.Module],
        opt_cb: torch.optim.Optimizer,
        opt_actors: List[torch.optim.Optimizer],
        opt_critics: List[torch.optim.Optimizer],
        gamma: float = 0.95,
        tau_polyak: float = 0.01,
    ) -> None:
        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.actors = actors
        self.critics = critics
        self.target_actors = target_actors
        self.target_critics = target_critics
        self.opt_cb = opt_cb
        self.opt_actors = opt_actors
        self.opt_critics = opt_critics
        self.gamma = gamma
        self.tau_polyak = tau_polyak

        n_agents = len(self.actors)
        if n_agents == 0:
            raise ValueError("actors list cannot be empty.")
        if len(self.critics) != n_agents:
            raise ValueError("critics length must match actors length.")
        if len(self.target_actors) != n_agents or len(self.target_critics) != n_agents:
            raise ValueError("target networks length must match actors length.")
        if len(self.opt_actors) != n_agents or len(self.opt_critics) != n_agents:
            raise ValueError("optimizer lists must match number of agents.")

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        obs = batch["obs"]
        state = batch["state"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        next_state = batch["next_state"]
        dones = batch["dones"]

        batch_size, n_agents, action_dim = actions.shape

        # 1) Consensus builder update (student gets gradients, teacher gets EMA).
        loss_cb, consensus = self.consensus_builder(obs)
        self.opt_cb.zero_grad()
        loss_cb.backward()
        self.opt_cb.step()
        self.consensus_builder.ema_update_teacher()

        # Fraction of batch timesteps where all agents picked the same class.
        consensus_agreement = (
            (consensus == consensus[:, :1]).all(dim=1).float().mean().item()
        )

        # Entropy of the marginal consensus label distribution over batch × agents.
        # Near log(K) → CB uses all classes uniformly; near 0 → collapsed to one class.
        with torch.no_grad():
            flat_c = consensus.reshape(-1)
            k = self.consensus_builder.k
            counts = torch.zeros(k, device=flat_c.device, dtype=torch.float32)
            for i in range(k):
                counts[i] = (flat_c == i).sum().float()
            probs = counts / counts.sum().clamp(min=1.0)
            consensus_entropy = -(probs * (probs + 1e-8).log()).sum().item()

        # 2) Build detached consensus embeddings for RL updates.
        emb_c = self.embedding_layer(consensus).detach()
        with torch.no_grad():
            # Use infer() instead of forward() to avoid updating CB center twice.
            next_consensus = torch.stack(
                [self.consensus_builder.infer(next_obs[b]) for b in range(batch_size)],
                dim=0,
            )
            next_emb_c = self.embedding_layer(next_consensus).detach()

        all_emb_c = emb_c.reshape(batch_size, -1)
        all_next_emb_c = next_emb_c.reshape(batch_size, -1)
        all_actions = actions.reshape(batch_size, -1)

        # 3) Target actions for next state.
        with torch.no_grad():
            next_actions = torch.stack(
                [
                    self.target_actors[a](next_obs[:, a, :], next_emb_c[:, a, :])
                    for a in range(n_agents)
                ],
                dim=1,
            )
            next_all_actions = next_actions.reshape(batch_size, -1)

        critic_losses = []
        actor_losses = []

        # 4) Critic updates.
        for a in range(n_agents):
            with torch.no_grad():
                q_target = self.target_critics[a](
                    next_state,
                    all_next_emb_c,
                    next_all_actions,
                )
                y = rewards[:, a : a + 1] + self.gamma * q_target * (1.0 - dones[:, a : a + 1])

            q_pred = self.critics[a](state, all_emb_c, all_actions)
            loss_c = F.mse_loss(q_pred, y)

            self.opt_critics[a].zero_grad()
            loss_c.backward()
            self.opt_critics[a].step()
            critic_losses.append(float(loss_c.item()))

        # 5) Actor updates (only actor parameters receive policy gradient).
        for a in range(n_agents):
            _set_requires_grad(self.critics[a], False)

            pred_action_a = self.actors[a](obs[:, a, :], emb_c[:, a, :].detach())

            # Keep other agents' actions from replay buffer; replace only agent a.
            mixed_action_parts = []
            for i in range(n_agents):
                if i == a:
                    mixed_action_parts.append(pred_action_a)
                else:
                    mixed_action_parts.append(actions[:, i, :].detach())
            mixed_actions = torch.cat(mixed_action_parts, dim=-1)

            loss_a = -self.critics[a](state, all_emb_c.detach(), mixed_actions).mean()

            self.opt_actors[a].zero_grad()
            loss_a.backward()
            self.opt_actors[a].step()
            actor_losses.append(float(loss_a.item()))

            _set_requires_grad(self.critics[a], True)

        # 6) Soft-update target networks.
        for a in range(n_agents):
            for param, target_param in zip(self.actors[a].parameters(), self.target_actors[a].parameters()):
                target_param.data.mul_(1.0 - self.tau_polyak).add_(self.tau_polyak * param.data)
            for param, target_param in zip(self.critics[a].parameters(), self.target_critics[a].parameters()):
                target_param.data.mul_(1.0 - self.tau_polyak).add_(self.tau_polyak * param.data)

        return {
            "loss_cb": float(loss_cb.item()),
            "consensus_agreement": float(consensus_agreement),
            "consensus_entropy": float(consensus_entropy),
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
        }
