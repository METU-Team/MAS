"""On-policy MAPPO training loop for COLA modules."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from cola_framework.interfaces.training_loop import TrainingLoopModule
from cola_framework.loops.cola_training_loop import _merge_update_metrics


@dataclass
class MAPPOTrainingConfig:
    """Runtime configuration for the MAPPO training loop."""

    max_steps: int = 2_000_000
    n_rollout_steps: int = 2_048      # env steps collected before each update
    log_interval: int = 10_000
    reward_window: int = 100          # number of recent *episodes* to average
    eval_interval: int = 0            # >0 enables periodic greedy eval (env steps)
    eval_episodes: int = 20           # episodes per periodic eval


class MAPPOTrainingLoop(TrainingLoopModule):
    """Collects on-policy rollouts and drives the MAPPO update cycle.

    Data flow per update cycle:
      1. Collect n_rollout_steps environment transitions.
         For each step: CB.infer → embedding → GaussianActor.sample → env.step.
         Values are computed by the value functions for GAE.
      2. Compute GAE advantages (via RolloutBuffer.compute_returns_and_advantages).
      3. Call MAPPOUpdater.update(batch) which runs n_epochs × n_minibatches of PPO.
      4. Clear rollout buffer and repeat.

    Episode tracking: rewards accumulate within each episode; episode_return is the
    mean of the last reward_window completed episodes.
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
        evaluator=None,
        on_eval: Optional[Callable[[Dict[str, object], int], None]] = None,
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
        self.evaluator = evaluator
        self.on_eval = on_eval

        if len(self.actors) != self.env.n_agents:
            raise ValueError("Number of actors must match env.n_agents.")
        if len(self.value_fns) != self.env.n_agents:
            raise ValueError("Number of value_fns must match env.n_agents.")

        # Log every time we complete this many PPO updates.
        self._log_every_n_updates = max(
            1, self.config.log_interval // self.config.n_rollout_steps
        )

    @torch.no_grad()
    def _act_and_estimate(self, obs: torch.Tensor, state: torch.Tensor):
        consensus = self.consensus_builder.infer(obs)
        cemb = self.embedding_layer(consensus)
        all_emb = cemb.reshape(1, -1)
        state_b = state.unsqueeze(0)

        actions_list, log_prob_list, value_list = [], [], []
        for a in range(self.env.n_agents):
            action_a, lp_a = self.actors[a].sample(obs[a], cemb[a])
            v_a = self.value_fns[a](state_b, all_emb).squeeze()
            actions_list.append(action_a)
            log_prob_list.append(lp_a)
            value_list.append(v_a)

        return (
            torch.stack(actions_list, dim=0),
            torch.stack(log_prob_list, dim=0),
            torch.stack(value_list, dim=0),
        )

    @torch.no_grad()
    def _bootstrap_values(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
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
        logs: List[Dict] = []
        last_update_metrics: Optional[Dict] = None
        next_eval_at = self.config.eval_interval

        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        _ep_reward = 0.0
        _ep_length = 0

        while step < self.config.max_steps:
            # ── Rollout collection ───────────────────────────────────────────
            self.rollout_buffer.clear()
            for _ in range(self.config.n_rollout_steps):
                actions, log_probs, values = self._act_and_estimate(obs, state)
                next_obs, next_state, rewards, dones = self.env.step(actions)

                self.rollout_buffer.push(
                    obs=obs, state=state, actions=actions,
                    rewards=rewards, dones=dones.float(),
                    values=values, log_probs=log_probs,
                )

                _ep_reward += float(rewards.mean().item())
                _ep_length += 1
                obs = next_obs
                state = next_state
                step += 1

                if bool(dones.any().item()):
                    episode_returns.append(_ep_reward)
                    episode_lengths.append(_ep_length)
                    _ep_reward = 0.0
                    _ep_length = 0
                    obs, state = self.env.reset()
                    dones = torch.zeros(self.env.n_agents, device=obs.device)

                if step >= self.config.max_steps:
                    break

            # ── GAE and update ───────────────────────────────────────────────
            last_values = self._bootstrap_values(obs, state)
            self.rollout_buffer.compute_returns_and_advantages(last_values, dones)
            last_update_metrics = self.updater.update(self.rollout_buffer.get())
            update_count += 1

            # ── Logging ──────────────────────────────────────────────────────
            if update_count % self._log_every_n_updates == 0 or step >= self.config.max_steps:
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
                    "update_count": update_count,
                }
                _merge_update_metrics(record, last_update_metrics)

                logs.append(record)
                if self.on_log is not None:
                    self.on_log(record)
                if self.metrics_logger is not None:
                    self.metrics_logger.log_metrics(record, step=step)

                # ── Periodic greedy eval (learning curve + collapse metrics) ──
                # Mirrors the MADDPG / history-aware paths so eval_episode_return,
                # eval_distinct_classes and eval_consensus_entropy are logged as
                # curves rather than a single end-of-training number.
                if (
                    self.evaluator is not None
                    and self.config.eval_interval > 0
                    and step >= next_eval_at
                ):
                    eval_record = self.evaluator.evaluate(self.config.eval_episodes)
                    if self.on_eval is not None:
                        self.on_eval(eval_record, step)
                    if self.metrics_logger is not None:
                        self.metrics_logger.log_metrics(
                            {"eval/" + k: v for k, v in eval_record.items()}, step=step
                        )
                    next_eval_at += self.config.eval_interval
                    # Eval consumed the shared env; restart a clean rollout episode.
                    obs, state = self.env.reset()
                    dones = torch.zeros(self.env.n_agents, device=obs.device)
                    _ep_reward = 0.0
                    _ep_length = 0

        result = {
            "total_steps": step,
            "update_count": update_count,
            "logs": logs,
        }
        if self.metrics_logger is not None:
            self.metrics_logger.finish({"total_steps": step, "update_count": update_count})
        return result
