"""Step 9 gate test for WandB logging and evaluation modules."""

import os
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.evaluation.policy_evaluator import PolicyEvaluator
from cola_framework.monitoring.wandb_logger import WandbLogger
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.policies.actor import Actor


class DummyEnv:
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, episode_len: int = 5) -> None:
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


class FakeWandbRun:
    def __init__(self) -> None:
        self.summary = {}
        self.finished = False

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self) -> None:
        self.login_key = None
        self.login_called = False
        self.init_called = False
        self.logged = []
        self.run = FakeWandbRun()

    def login(self, key, relogin=True):
        self.login_called = True
        self.login_key = key

    def init(self, **kwargs):
        self.init_called = True
        return self.run

    def log(self, payload, step=None):
        self.logged.append((payload, step))


def test_wandb_logger() -> None:
    fake_wandb = FakeWandb()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write("dummy_key_123\n")
        key_path = tmp.name

    try:
        logger = WandbLogger(
            project="cola-test",
            run_name="step9-gate",
            mode="offline",
            api_key_path=key_path,
            wandb_module=fake_wandb,
        )

        assert fake_wandb.login_called
        assert fake_wandb.login_key == "dummy_key_123"
        assert fake_wandb.init_called

        logger.log_metrics(
            {
                "mean_reward": 1.5,
                "actor_losses": [0.2, 0.3, 0.4],
                "note": "non-numeric values must be ignored",
            },
            step=10,
        )

        assert len(fake_wandb.logged) == 1
        payload, step = fake_wandb.logged[0]
        assert step == 10
        assert "mean_reward" in payload
        assert "actor_losses/0" in payload
        assert "actor_losses/mean" in payload
        assert "note" not in payload

        logger.finish({"final_noise_std": 0.05})
        assert fake_wandb.run.finished
        assert "final_noise_std" in fake_wandb.run.summary
    finally:
        os.remove(key_path)


def test_policy_evaluator() -> None:
    torch.manual_seed(303)

    n_agents = 3
    obs_dim = 6
    action_dim = 2
    emb_dim = 8

    env = DummyEnv(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim)
    cb = ConsensusBuilder(obs_dim=obs_dim, k=4, hidden_dim=32)
    emb = ConsensusEmbedding(k=4, emb_dim=emb_dim)
    actors = [Actor(obs_dim=obs_dim, emb_dim=emb_dim, action_dim=action_dim, hidden_dim=32) for _ in range(n_agents)]

    evaluator = PolicyEvaluator(
        env=env,
        consensus_builder=cb,
        embedding_layer=emb,
        actors=actors,
    )
    metrics = evaluator.evaluate(n_episodes=5)

    assert "mean_episode_return" in metrics
    assert "consensus_agreement_eval" in metrics
    assert isinstance(metrics["mean_episode_return"], float)
    assert 0.0 <= metrics["consensus_agreement_eval"] <= 1.0


if __name__ == "__main__":
    test_wandb_logger()
    test_policy_evaluator()
    print("Step 9 gate passed: Logging and Evaluation OK")
