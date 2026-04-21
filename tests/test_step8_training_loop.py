"""Step 8 module gate test for full training loop orchestration."""

import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.replay_buffer import ReplayBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.critics.centralized_critic import Critic
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop
from cola_framework.policies.actor import Actor
from cola_framework.trainers.maddpg_updater import MADDPGUpdater


class DummyEnv:
    """Small deterministic-shape environment for training-loop gate tests."""

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

        # Reward depends on actions so policy outputs affect learning signals.
        rewards = -action_tensor.pow(2).mean(dim=-1)

        done_flag = self._t >= self._episode_len
        dones = torch.full((self.n_agents,), done_flag, dtype=torch.bool)
        return next_obs, next_state, rewards, dones


def main() -> None:
    torch.manual_seed(202)

    n_agents = 3
    obs_dim = 6
    action_dim = 2
    emb_dim = 8
    state_dim = n_agents * obs_dim

    env = DummyEnv(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim, episode_len=7)
    buffer = ReplayBuffer(
        capacity=400,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
    )

    cb = ConsensusBuilder(obs_dim=obs_dim, k=4, hidden_dim=32)
    emb = ConsensusEmbedding(k=4, emb_dim=emb_dim)

    actors = [Actor(obs_dim=obs_dim, emb_dim=emb_dim, action_dim=action_dim, hidden_dim=32) for _ in range(n_agents)]
    critics = [
        Critic(state_dim=state_dim, n_agents=n_agents, emb_dim=emb_dim, action_dim=action_dim, hidden_dim=32)
        for _ in range(n_agents)
    ]
    target_actors = [copy.deepcopy(a) for a in actors]
    target_critics = [copy.deepcopy(c) for c in critics]

    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_actors = [torch.optim.Adam(a.parameters(), lr=1e-3) for a in actors]
    opt_critics = [torch.optim.Adam(c.parameters(), lr=1e-3) for c in critics]

    updater = MADDPGUpdater(
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

    loop = COLATrainingLoop(
        env=env,
        replay_buffer=buffer,
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

    # Ensure update metrics were propagated into logs after first optimization.
    has_update_metrics = any("loss_cb" in r for r in result["logs"])
    assert has_update_metrics

    print("Step 8 gate passed: Training loop OK")


if __name__ == "__main__":
    main()
