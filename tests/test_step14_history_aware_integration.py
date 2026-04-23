"""Step 14 gate test for history-aware end-to-end training path."""

import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.sequence_buffer import SequenceReplayBuffer
from cola_framework.consensus.history_aware_builder import HistoryAwareConsensusBuilder
from cola_framework.critics.centralized_critic import Critic
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.encoders.gru_encoder import GRUHistoryEncoder
from cola_framework.evaluation.history_aware_policy_evaluator import HistoryAwarePolicyEvaluator
from cola_framework.loops.cola_training_loop import COLATrainingConfig
from cola_framework.loops.history_aware_training_loop import HistoryAwareCOLATrainingLoop
from cola_framework.policies.actor import Actor
from cola_framework.trainers.history_aware_maddpg_updater import HistoryAwareMADDPGUpdater
from cola_framework.utils.window_manager import ObservationWindowManager


class DummyEnv:
    """Small deterministic-shape environment for integration gate tests."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, episode_len: int = 7) -> None:
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = n_agents * obs_dim
        self._episode_len = episode_len
        self._t = 0

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self._t = 0
        obs = torch.randn(self.n_agents, self.obs_dim)
        state = obs.reshape(-1)
        return obs, state

    def step(self, action_tensor: torch.Tensor):
        self._t += 1
        next_obs = torch.randn(self.n_agents, self.obs_dim)
        next_state = next_obs.reshape(-1)

        rewards = -action_tensor.pow(2).mean(dim=-1)

        done_flag = self._t >= self._episode_len
        dones = torch.full((self.n_agents,), done_flag, dtype=torch.bool)
        return next_obs, next_state, rewards, dones


def main() -> None:
    torch.manual_seed(303)

    n_agents = 3
    obs_dim = 6
    action_dim = 2
    emb_dim = 8
    state_dim = n_agents * obs_dim
    window = 4

    env = DummyEnv(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim, episode_len=7)

    buffer = SequenceReplayBuffer(
        capacity=400,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        window=window,
    )

    window_manager = ObservationWindowManager(
        n_agents=n_agents,
        obs_dim=obs_dim,
        window=window,
    )

    encoder = GRUHistoryEncoder(obs_dim=obs_dim, hidden_dim=32)
    cb = HistoryAwareConsensusBuilder(encoder=encoder, k=4, mlp_hidden=32)
    emb = ConsensusEmbedding(k=4, emb_dim=emb_dim)

    actors = [Actor(obs_dim=obs_dim, emb_dim=emb_dim, action_dim=action_dim, hidden_dim=32) for _ in range(n_agents)]
    critics = [
        Critic(state_dim=state_dim, n_agents=n_agents, emb_dim=emb_dim, action_dim=action_dim, hidden_dim=32)
        for _ in range(n_agents)
    ]
    target_actors = [copy.deepcopy(a) for a in actors]
    target_critics = [copy.deepcopy(c) for c in critics]

    opt_cb = torch.optim.Adam(
        list(cb.student_encoder.parameters()) + list(cb.student_head.parameters()),
        lr=3e-4,
    )
    opt_actors = [torch.optim.Adam(a.parameters(), lr=1e-3) for a in actors]
    opt_critics = [torch.optim.Adam(c.parameters(), lr=1e-3) for c in critics]

    updater = HistoryAwareMADDPGUpdater(
        consensus_builder=cb,
        embedding_layer=emb,
        actors=actors,
        critics=critics,
        target_actors=target_actors,
        target_critics=target_critics,
        opt_cb=opt_cb,
        opt_actors=opt_actors,
        opt_critics=opt_critics,
        gamma=0.95,
        tau_polyak=0.01,
    )

    config = COLATrainingConfig(
        max_steps=40,
        warmup_steps=8,
        train_freq=4,
        batch_size=8,
        noise_std_init=0.2,
        noise_std_min=0.05,
        noise_decay=0.95,
        log_interval=10,
        reward_window=10,
    )

    logs = []

    def on_log(record):
        logs.append(record)

    loop = HistoryAwareCOLATrainingLoop(
        env=env,
        replay_buffer=buffer,
        window_manager=window_manager,
        consensus_builder=cb,
        embedding_layer=emb,
        actors=actors,
        updater=updater,
        config=config,
        on_log=on_log,
    )

    result = loop.run()

    assert result["total_steps"] == 40
    assert result["train_steps"] > 0
    assert len(result["logs"]) == 4
    assert len(logs) == len(result["logs"])

    has_update_metrics = any("loss_cb" in r for r in result["logs"])
    assert has_update_metrics

    eval_window_manager = ObservationWindowManager(
        n_agents=n_agents,
        obs_dim=obs_dim,
        window=window,
    )
    evaluator = HistoryAwarePolicyEvaluator(
        env=env,
        window_manager=eval_window_manager,
        consensus_builder=cb,
        embedding_layer=emb,
        actors=actors,
    )
    eval_result = evaluator.evaluate(n_episodes=3)

    assert "mean_episode_return" in eval_result
    assert "consensus_agreement_eval" in eval_result

    print("Step 14 gate passed: History-aware integration OK")


if __name__ == "__main__":
    main()
