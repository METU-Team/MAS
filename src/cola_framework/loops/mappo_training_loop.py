"""On-policy MAPPO training loop for COLA modules."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule


@dataclass
class MAPPOTrainingConfig:
    """Runtime configuration for the MAPPO training loop."""

    max_steps: int = 2_000_000
    n_rollout_steps: int = 2048      # env steps collected before each update
    log_interval: int = 10_000
    reward_window: int = 1_000


class MAPPOTrainingLoop(TrainingLoopModule):
    """Collects on-policy rollouts and drives the MAPPO update cycle.

    Data flow per update cycle:
      1. Collect n_rollout_steps environment transitions.
         For each step: CB.infer → embedding → GaussianActor.sample → env.step
         Values are computed by the value functions for GAE.
      2. Compute GAE advantages (via RolloutBuffer.compute_returns_and_advantages).
      3. Call MAPPOUpdater.update(batch) which runs n_epochs × n_minibatches of PPO.
      4. Clear rollout buffer and repeat.
    """

    def __init__(
        self,
        env,
        rollout_buffer,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        actors: List[torch.nn.Module],
        value_fns: List[torch.nn.Module],
        updater,
        config: MAPPOTrainingConfig,
        on_log: Optional[Callable[[Dict[str, object]], None]] = None,
        metrics_logger=None,
    ) -> None:
        self.env = env
        self.rollout_buffer = rollout_buffer
        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.actors = actors
        self.value_fns = value_fns
        self.updater = updater
        self.config = config
        self.on_log = on_log
        self.metrics_logger = metrics_logger

        if len(self.actors) != self.env.n_agents:
            raise ValueError("Number of actors must match env.n_agents.")
        if len(self.value_fns) != self.env.n_agents:
            raise ValueError("Number of value_fns must match env.n_agents.")

    @torch.no_grad()
    def _act_and_estimate(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
    ):
        """Sample actions and compute log-probs and state values.

        Returns:
            actions   : [n_agents, action_dim]
            log_probs : [n_agents]
            values    : [n_agents]
        """
        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)              # [n_agents, emb_dim]
        all_emb = cemb.reshape(1, -1)                       # [1, n_agents * emb_dim]
        state_b = state.unsqueeze(0)                        # [1, state_dim]

        actions_list, log_prob_list, value_list = [], [], []
        for a in range(self.env.n_agents):
            # Use sample() from GaussianActor (not forward() which gives mean).
            action_a, lp_a = self.actors[a].sample(obs[a], cemb[a])
            v_a = self.value_fns[a](state_b, all_emb).squeeze()
            actions_list.append(action_a)
            log_prob_list.append(lp_a)
            value_list.append(v_a)

        actions = torch.stack(actions_list, dim=0)          # [n_agents, action_dim]
        log_probs = torch.stack(log_prob_list, dim=0)       # [n_agents]
        values = torch.stack(value_list, dim=0)             # [n_agents]
        return actions, log_probs, values

    @torch.no_grad()
    def _bootstrap_values(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute V(s_T) for the state following the last rollout step."""
        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)
        all_emb = cemb.reshape(1, -1)
        state_b = state.unsqueeze(0)
        return torch.stack(
            [self.value_fns[a](state_b, all_emb).squeeze() for a in range(self.env.n_agents)],
            dim=0,
        )

    def run(self) -> Dict[str, object]:
        obs, state = self.env.reset()
        dones = torch.zeros(self.env.n_agents, device=obs.device)

        step = 0
        update_count = 0
        episode_rewards: List[float] = []
        logs = []
        last_update_metrics = None

        while step < self.config.max_steps:
            # ── Rollout collection ───────────────────────────────────────────
            self.rollout_buffer.clear()
            for _ in range(self.config.n_rollout_steps):
                actions, log_probs, values = self._act_and_estimate(obs, state)

                next_obs, next_state, rewards, dones = self.env.step(actions)

                self.rollout_buffer.push(
                    obs=obs,
                    state=state,
                    actions=actions,
                    rewards=rewards,
                    dones=dones.float(),
                    values=values,
                    log_probs=log_probs,
                )

                episode_rewards.append(float(rewards.mean().item()))
                obs = next_obs
                state = next_state
                step += 1

                if bool(dones.any().item()):
                    obs, state = self.env.reset()
                    dones = torch.zeros(self.env.n_agents, device=obs.device)

                if step >= self.config.max_steps:
                    break

            # ── GAE computation ──────────────────────────────────────────────
            last_values = self._bootstrap_values(obs, state)
            self.rollout_buffer.compute_returns_and_advantages(last_values, dones)

            # ── PPO update ───────────────────────────────────────────────────
            batch = self.rollout_buffer.get()
            last_update_metrics = self.updater.update(batch)
            update_count += 1

            # ── Logging ──────────────────────────────────────────────────────
            if step % self.config.log_interval < self.config.n_rollout_steps or \
               step >= self.config.max_steps:
                window = episode_rewards[-self.config.reward_window :]
                mean_reward = float(sum(window) / max(1, len(window)))
                record: Dict[str, object] = {
                    "step": step,
                    "mean_reward": mean_reward,
                    "update_count": update_count,
                }
                if last_update_metrics is not None:
                    record.update(last_update_metrics)

                logs.append(record)
                if self.on_log is not None:
                    self.on_log(record)
                if self.metrics_logger is not None:
                    self.metrics_logger.log_metrics(record, step=step)

        result = {
            "total_steps": step,
            "update_count": update_count,
            "logs": logs,
        }
        if self.metrics_logger is not None:
            self.metrics_logger.finish({"total_steps": step, "update_count": update_count})
        return result
