"""Off-policy QMIX training loop with epsilon-greedy exploration for COLA."""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule


@dataclass
class QMIXTrainingConfig:
    """Runtime configuration for the QMIX training loop."""

    max_steps: int = 2_000_000
    warmup_steps: int = 1_024         # random actions only before this step
    train_freq: int = 100             # update every N environment steps
    batch_size: int = 1_024
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay_steps: int = 500_000    # linear annealing over this many steps
    log_interval: int = 10_000
    reward_window: int = 1_000


class QMIXTrainingLoop(TrainingLoopModule):
    """Coordinates env interaction and QMIXUpdater for COLA + QMIX.

    Action selection:
      - Epsilon-greedy: random action with probability ε, otherwise
        argmax over per-agent Q-values.
      - ε decays linearly from eps_start to eps_min over eps_decay_steps.

    Storage:
      - Discrete actions stored as float indices with shape [n_agents, 1]
        for compatibility with the existing ReplayBuffer interface.
    """

    def __init__(
        self,
        env,
        replay_buffer,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        q_networks: List[torch.nn.Module],
        updater,
        config: QMIXTrainingConfig,
        on_log: Optional[Callable[[Dict[str, object]], None]] = None,
        metrics_logger=None,
    ) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.q_networks = q_networks
        self.updater = updater
        self.config = config
        self.on_log = on_log
        self.metrics_logger = metrics_logger

        if len(self.q_networks) != self.env.n_agents:
            raise ValueError("Number of q_networks must match env.n_agents.")

        self._n_actions = self.env.n_actions

    def _epsilon(self, step: int) -> float:
        """Linear epsilon annealing schedule."""
        cfg = self.config
        progress = min(1.0, step / max(1, cfg.eps_decay_steps))
        return cfg.eps_start + (cfg.eps_min - cfg.eps_start) * progress

    @torch.no_grad()
    def _act(self, obs: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Return discrete actions [n_agents] using epsilon-greedy policy."""
        if random.random() < epsilon:
            return torch.randint(0, self._n_actions, (self.env.n_agents,))

        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)
        return torch.stack(
            [
                self.q_networks[a](obs[a], cemb[a]).argmax()
                for a in range(self.env.n_agents)
            ],
            dim=0,
        )

    def run(self) -> Dict[str, object]:
        obs, state = self.env.reset()

        step = 0
        train_steps = 0
        episode_rewards: List[float] = []
        logs = []
        last_update_metrics = None

        while step < self.config.max_steps:
            epsilon = self._epsilon(step)
            actions = self._act(obs, epsilon)  # [n_agents] int64

            next_obs, next_state, rewards, dones = self.env.step(actions)

            # Store actions as float [n_agents, 1] for ReplayBuffer compatibility.
            self.replay_buffer.push(
                obs,
                state,
                actions.float().unsqueeze(-1),   # [n_agents, 1]
                rewards,
                next_obs,
                next_state,
                dones.float(),
            )

            obs = next_obs
            state = next_state
            step += 1
            episode_rewards.append(float(rewards.mean().item()))

            if bool(dones.any().item()):
                obs, state = self.env.reset()

            if (
                step >= self.config.warmup_steps
                and step % self.config.train_freq == 0
                and len(self.replay_buffer) >= self.config.batch_size
            ):
                batch = self.replay_buffer.sample(self.config.batch_size)
                last_update_metrics = self.updater.update(batch)
                train_steps += 1

            if step % self.config.log_interval == 0:
                window = episode_rewards[-self.config.reward_window :]
                mean_reward = float(sum(window) / max(1, len(window)))
                record: Dict[str, object] = {
                    "step": step,
                    "mean_reward": mean_reward,
                    "epsilon": round(epsilon, 4),
                    "buffer_size": int(len(self.replay_buffer)),
                    "train_steps": train_steps,
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
            "train_steps": train_steps,
            "logs": logs,
        }
        if self.metrics_logger is not None:
            self.metrics_logger.finish({"total_steps": step, "train_steps": train_steps})
        return result
