"""History-aware training loop for COLA + MADDPG modules."""

from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule
from cola_framework.loops.cola_training_loop import COLATrainingConfig


class HistoryAwareCOLATrainingLoop(TrainingLoopModule):
    """Orchestrates history-aware rollout and update data flow.

    Data flow per step:
      env obs -> ObservationWindowManager -> history-aware CB infer
      -> embedding -> actor actions -> env step
      -> SequenceReplayBuffer push (flat + sequence fields)
      -> updater sample/update on schedule
    """

    def __init__(
        self,
        env,
        replay_buffer,
        window_manager,
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
        self.window_manager = window_manager
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
    def _act(self, obs: torch.Tensor, obs_seq: torch.Tensor, noise_std: float) -> torch.Tensor:
        consensus = self.consensus_builder.infer(obs_seq)
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

        self.window_manager.reset()
        self.window_manager.push(obs)

        step = 0
        train_steps = 0
        episode_rewards = []
        logs = []
        last_update_metrics = None

        while step < self.config.max_steps:
            obs_seq = self.window_manager.get()
            actions = self._act(obs, obs_seq, noise_std)

            next_obs, next_state, rewards, dones = self.env.step(actions)

            self.window_manager.push(next_obs)
            next_obs_seq = self.window_manager.get()

            self.replay_buffer.push(
                obs,
                state,
                actions,
                rewards,
                next_obs,
                next_state,
                dones.float(),
                obs_seq=obs_seq,
                next_obs_seq=next_obs_seq,
            )

            obs = next_obs
            state = next_state
            step += 1
            episode_rewards.append(float(rewards.mean().item()))

            if bool(dones.any().item()):
                obs, state = self.env.reset()
                self.window_manager.reset()
                self.window_manager.push(obs)
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
