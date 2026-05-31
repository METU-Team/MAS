"""Full training loop for COLA + MADDPG modules."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule


@dataclass
class COLATrainingConfig:
    """Runtime configuration for the training loop."""

    max_steps: int = 2_000_000
    warmup_steps: int = 1_024
    train_freq: int = 100
    batch_size: int = 1_024
    noise_std_init: float = 0.3
    noise_std_min: float = 0.05
    noise_decay: float = 0.9999
    log_interval: int = 10_000
    reward_window: int = 100    # number of recent *episodes* to average


def _merge_update_metrics(record: Dict, um: Dict) -> None:
    """Flatten updater output into the log record with clean scalar names.

    List-valued per-agent metrics are averaged to a single scalar so that
    WandB receives one clean line per metric rather than N indexed lines.
    """
    for src, dst in [
        ("actor_losses", "actor_loss"),
        ("critic_losses", "critic_loss"),
        ("value_losses", "value_loss"),
    ]:
        vals = um.get(src, [])
        if vals:
            record[dst] = float(sum(vals) / len(vals))

    for key in ("loss_cb", "loss_qmix", "consensus_agreement", "consensus_entropy"):
        if key in um:
            record[key] = float(um[key])


class COLATrainingLoop(TrainingLoopModule):
    """Orchestrates data flow among environment, buffer, policy, and updater.

    Data flow per step:
      env -> consensus infer -> embedding -> actor actions -> env.step
      transition -> replay buffer -> updater (on schedule)

    Episode tracking:
      rewards are accumulated within each episode; episode_return in the log
      record is the mean of the last reward_window completed episodes, not
      the mean of individual step rewards.
    """

    def __init__(
        self,
        env,
        replay_buffer,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        actors: List[torch.nn.Module],
        updater,
        config: COLATrainingConfig,
        on_log: Optional[Callable[[Dict[str, object]], None]] = None,
        metrics_logger=None,
    ) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.actors = actors
        self.updater = updater
        self.config = config
        self.on_log = on_log
        self.metrics_logger = metrics_logger

        if len(self.actors) != self.env.n_agents:
            raise ValueError("Number of actors must match env.n_agents.")

    @torch.no_grad()
    def _act(self, obs: torch.Tensor, noise_std: float) -> torch.Tensor:
        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)
        actions = torch.stack(
            [self.actors[a](obs[a], cemb[a]) for a in range(self.env.n_agents)],
            dim=0,
        )
        if noise_std > 0.0:
            actions = actions + noise_std * torch.randn_like(actions)
        return actions.clamp(-1.0, 1.0)

    def run(self) -> Dict[str, object]:
        obs, state = self.env.reset()
        noise_std = self.config.noise_std_init

        step = 0
        train_steps = 0
        logs: List[Dict] = []
        last_update_metrics: Optional[Dict] = None

        # Episode-level return tracking — accumulate within each episode,
        # record when an episode completes.
        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        _ep_reward = 0.0
        _ep_length = 0

        while step < self.config.max_steps:
            actions = self._act(obs, noise_std)
            next_obs, next_state, rewards, dones = self.env.step(actions)

            self.replay_buffer.push(obs, state, actions, rewards,
                                    next_obs, next_state, dones.float())

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
                noise_std = max(self.config.noise_std_min,
                                noise_std * self.config.noise_decay)

            if (
                step >= self.config.warmup_steps
                and step % self.config.train_freq == 0
                and len(self.replay_buffer) > 0
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
                    "noise_std": float(noise_std),
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
            "final_noise_std": float(noise_std),
            "logs": logs,
        }
        if self.metrics_logger is not None:
            self.metrics_logger.finish({
                "total_steps": step,
                "train_steps": train_steps,
                "final_noise_std": float(noise_std),
            })
        return result
