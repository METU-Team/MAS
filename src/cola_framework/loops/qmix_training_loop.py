"""Off-policy QMIX training loop with epsilon-greedy exploration for COLA."""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule
from cola_framework.loops.cola_training_loop import _merge_update_metrics


@dataclass
class QMIXTrainingConfig:
    """Runtime configuration for the QMIX training loop."""

    max_steps: int = 2_000_000
    warmup_steps: int = 1_024
    train_freq: int = 100
    batch_size: int = 1_024
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay_steps: int = 500_000    # linear annealing over this many steps
    log_interval: int = 10_000
    reward_window: int = 100          # number of recent *episodes* to average


class QMIXTrainingLoop(TrainingLoopModule):
    """Coordinates env interaction and QMIXUpdater for COLA + QMIX.

    Action selection:
      Epsilon-greedy with linear annealing: random action with probability ε,
      otherwise argmax over per-agent Q-values.

    Storage:
      Discrete actions stored as float indices with shape [n_agents, 1]
      for compatibility with the existing ReplayBuffer interface.

    Episode tracking:
      Rewards accumulate within each episode; episode_return in the log record
      is the mean of the last reward_window completed episodes.
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
        progress = min(1.0, step / max(1, self.config.eps_decay_steps))
        return self.config.eps_start + (self.config.eps_min - self.config.eps_start) * progress

    @torch.no_grad()
    def _act(self, obs: torch.Tensor, epsilon: float) -> torch.Tensor:
        if random.random() < epsilon:
            return torch.randint(0, self._n_actions, (self.env.n_agents,))

        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)
        return torch.stack(
            [self.q_networks[a](obs[a], cemb[a]).argmax() for a in range(self.env.n_agents)],
            dim=0,
        )

    def run(self) -> Dict[str, object]:
        obs, state = self.env.reset()

        step = 0
        train_steps = 0
        logs: List[Dict] = []
        last_update_metrics: Optional[Dict] = None

        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        _ep_reward = 0.0
        _ep_length = 0

        while step < self.config.max_steps:
            epsilon = self._epsilon(step)
            actions = self._act(obs, epsilon)

            next_obs, next_state, rewards, dones = self.env.step(actions)

            self.replay_buffer.push(
                obs, state,
                actions.float().unsqueeze(-1),   # [n_agents, 1] float
                rewards, next_obs, next_state, dones.float(),
            )

            obs = next_obs
            state = next_state
            step += 1
            _ep_reward += float(rewards.mean().item())
            _ep_length += 1

            if bool(dones.any().item()):
                episode_returns.append(_ep_reward)
                episode_lengths.append(_ep_length)
                _ep_reward = 0.0
                _ep_length = 0
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
                n_ep = len(episode_returns)
                if n_ep > 0:
                    w_r = episode_returns[-self.config.reward_window:]
                    w_l = episode_lengths[-self.config.reward_window:]
                    mean_ep_ret = float(sum(w_r) / len(w_r))
                    max_ep_ret  = float(max(w_r))
                    min_ep_ret  = float(min(w_r))
                    mean_ep_len = float(sum(w_l) / len(w_l))
                else:
                    mean_ep_ret = max_ep_ret = min_ep_ret = mean_ep_len = 0.0

                record: Dict[str, object] = {
                    "step": step,
                    "episode_return": mean_ep_ret,
                    "episode_return_max": max_ep_ret,
                    "episode_return_min": min_ep_ret,
                    "episode_count": n_ep,
                    "mean_episode_length": mean_ep_len,
                    "epsilon": round(epsilon, 4),
                    "buffer_size": int(len(self.replay_buffer)),
                    "train_steps": train_steps,
                }
                if last_update_metrics is not None:
                    _merge_update_metrics(record, last_update_metrics)

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
