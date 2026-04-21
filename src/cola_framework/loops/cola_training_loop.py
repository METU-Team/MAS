"""Step 8: Full training loop for COLA + MADDPG modules."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule


@dataclass
class COLATrainingConfig:
    """Runtime configuration for the training loop."""

    max_steps: int = 2000000
    warmup_steps: int = 1024
    train_freq: int = 100
    batch_size: int = 1024
    noise_std_init: float = 0.3
    noise_std_min: float = 0.05
    noise_decay: float = 0.9999
    log_interval: int = 10000
    reward_window: int = 1000


class COLATrainingLoop(TrainingLoopModule):
    """Orchestrates data flow among environment, buffer, policy, and updater.

    Data flow per step:
      env -> consensus infer -> embedding -> actor actions -> env.step
      transition -> replay buffer -> updater (on schedule)
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
        """Generate joint actions from current observation."""
        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)

        actions = torch.stack(
            [
                self.actors[a](obs[a], cemb[a])
                for a in range(self.env.n_agents)
            ],
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
        episode_rewards = []
        logs = []
        last_update_metrics = None

        while step < self.config.max_steps:
            actions = self._act(obs, noise_std)

            next_obs, next_state, rewards, dones = self.env.step(actions)

            self.replay_buffer.push(
                obs,
                state,
                actions,
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
                noise_std = max(self.config.noise_std_min, noise_std * self.config.noise_decay)

            if (
                step >= self.config.warmup_steps
                and step % self.config.train_freq == 0
                and len(self.replay_buffer) > 0
            ):
                batch = self.replay_buffer.sample(self.config.batch_size)
                last_update_metrics = self.updater.update(batch)
                train_steps += 1

            if step % self.config.log_interval == 0:
                window = episode_rewards[-self.config.reward_window :]
                mean_reward = float(sum(window) / max(1, len(window)))
                record = {
                    "step": step,
                    "mean_reward": mean_reward,
                    "noise_std": float(noise_std),
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
            "final_noise_std": float(noise_std),
            "logs": logs,
        }
        if self.metrics_logger is not None:
            self.metrics_logger.finish(
                {
                    "total_steps": step,
                    "train_steps": train_steps,
                    "final_noise_std": float(noise_std),
                }
            )
        return result
