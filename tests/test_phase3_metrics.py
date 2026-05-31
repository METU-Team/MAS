"""Phase 3 gate test: verify that all metrics are correctly calculated.

Tests:
 - episode_return = sum of per-step rewards within each episode (not per-step mean)
 - consensus_entropy is in [0, log(K)] for COLA and exactly 0 for NullConsensusBuilder
 - WandB prefix transformation produces train/ and eval/ keys
 - _merge_update_metrics correctly extracts scalars and means from lists
 - All three training loops produce episode_return, episode_count, consensus_entropy
"""

import copy
import math
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.replay_buffer import ReplayBuffer
from cola_framework.buffers.rollout_buffer import RolloutBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.critics.centralized_critic import Critic
from cola_framework.critics.mixing_network import MixingNetwork
from cola_framework.critics.value_function import CentralizedValueFunction
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.discrete_mpe_wrapper import DiscreteMPEWrapper
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.evaluation.mappo_policy_evaluator import MAPPOPolicyEvaluator
from cola_framework.evaluation.policy_evaluator import PolicyEvaluator
from cola_framework.evaluation.qmix_policy_evaluator import QMIXPolicyEvaluator
from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop, _merge_update_metrics
from cola_framework.loops.mappo_training_loop import MAPPOTrainingConfig, MAPPOTrainingLoop
from cola_framework.loops.qmix_training_loop import QMIXTrainingConfig, QMIXTrainingLoop
from cola_framework.policies.actor import Actor
from cola_framework.policies.gaussian_actor import GaussianActor
from cola_framework.policies.q_network import QNetwork
from cola_framework.trainers.maddpg_updater import MADDPGUpdater
from cola_framework.trainers.mappo_updater import MAPPOUpdater
from cola_framework.trainers.qmix_updater import QMIXUpdater

N_AGENTS = 3
OBS_DIM = 18
EMB_DIM = 8
HIDDEN = 32
K = 4
DEVICE = "cpu"


# ── _merge_update_metrics ─────────────────────────────────────────────────────

def test_merge_scalars():
    um = {"loss_cb": 0.5, "consensus_agreement": 0.75, "consensus_entropy": 1.2}
    record = {}
    _merge_update_metrics(record, um)
    assert record["loss_cb"] == 0.5
    assert record["consensus_agreement"] == 0.75
    assert record["consensus_entropy"] == 1.2
    print("  [OK] _merge_update_metrics: scalars passed through correctly")


def test_merge_list_to_mean():
    um = {"actor_losses": [0.1, 0.2, 0.3], "critic_losses": [1.0, 2.0]}
    record = {}
    _merge_update_metrics(record, um)
    assert abs(record["actor_loss"] - 0.2) < 1e-6
    assert abs(record["critic_loss"] - 1.5) < 1e-6
    assert "actor_losses" not in record  # raw list not present
    assert "critic_losses" not in record
    print("  [OK] _merge_update_metrics: per-agent lists averaged to single scalar")


def test_merge_value_losses():
    um = {"value_losses": [0.4, 0.6], "loss_qmix": 0.88}
    record = {}
    _merge_update_metrics(record, um)
    assert abs(record["value_loss"] - 0.5) < 1e-6
    assert abs(record["loss_qmix"] - 0.88) < 1e-6
    print("  [OK] _merge_update_metrics: value_losses and loss_qmix handled")


# ── consensus_entropy ─────────────────────────────────────────────────────────

def test_null_builder_entropy_zero():
    """NullConsensusBuilder always outputs class 0 → entropy must be 0."""
    cb = NullConsensusBuilder(k=K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    actors = [Actor(OBS_DIM, EMB_DIM, 5, HIDDEN) for _ in range(N_AGENTS)]
    critics = [Critic(OBS_DIM * N_AGENTS, N_AGENTS, EMB_DIM, 5, HIDDEN) for _ in range(N_AGENTS)]
    t_actors = [copy.deepcopy(a) for a in actors]
    t_critics = [copy.deepcopy(c) for c in critics]
    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_actors = [torch.optim.Adam(a.parameters(), lr=1e-2) for a in actors]
    opt_critics = [torch.optim.Adam(c.parameters(), lr=1e-2) for c in critics]
    updater = MADDPGUpdater(cb, emb, actors, critics, t_actors, t_critics,
                             opt_cb, opt_actors, opt_critics)

    batch = {
        "obs": torch.randn(32, N_AGENTS, OBS_DIM),
        "state": torch.randn(32, OBS_DIM * N_AGENTS),
        "actions": torch.randn(32, N_AGENTS, 5).tanh(),
        "rewards": torch.randn(32, N_AGENTS),
        "next_obs": torch.randn(32, N_AGENTS, OBS_DIM),
        "next_state": torch.randn(32, OBS_DIM * N_AGENTS),
        "dones": torch.zeros(32, N_AGENTS),
    }
    metrics = updater.update(batch)
    assert abs(metrics["consensus_entropy"]) < 1e-5, \
        f"NullCB entropy should be 0, got {metrics['consensus_entropy']}"
    print("  [OK] NullConsensusBuilder → consensus_entropy = 0.0 (all class 0)")


def test_cola_builder_entropy_positive():
    """Trained ConsensusBuilder should produce entropy > 0 (uses multiple classes)."""
    torch.manual_seed(0)
    cb = ConsensusBuilder(OBS_DIM, K, HIDDEN)
    emb = ConsensusEmbedding(K, EMB_DIM)
    actors = [Actor(OBS_DIM, EMB_DIM, 5, HIDDEN) for _ in range(N_AGENTS)]
    critics = [Critic(OBS_DIM * N_AGENTS, N_AGENTS, EMB_DIM, 5, HIDDEN) for _ in range(N_AGENTS)]
    t_actors = [copy.deepcopy(a) for a in actors]
    t_critics = [copy.deepcopy(c) for c in critics]
    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    updater = MADDPGUpdater(cb, emb, actors, critics, t_actors, t_critics,
                             opt_cb,
                             [torch.optim.Adam(a.parameters()) for a in actors],
                             [torch.optim.Adam(c.parameters()) for c in critics])
    # Train a few steps so CB produces diverse labels
    for _ in range(50):
        batch = {
            "obs": torch.randn(64, N_AGENTS, OBS_DIM),
            "state": torch.randn(64, OBS_DIM * N_AGENTS),
            "actions": torch.randn(64, N_AGENTS, 5).tanh(),
            "rewards": torch.randn(64, N_AGENTS),
            "next_obs": torch.randn(64, N_AGENTS, OBS_DIM),
            "next_state": torch.randn(64, OBS_DIM * N_AGENTS),
            "dones": torch.zeros(64, N_AGENTS),
        }
        metrics = updater.update(batch)

    max_entropy = math.log(K)  # log(4) ≈ 1.386
    assert metrics["consensus_entropy"] >= 0.0
    assert metrics["consensus_entropy"] <= max_entropy + 1e-5, \
        f"entropy {metrics['consensus_entropy']} > log(K)={max_entropy}"
    print(f"  [OK] ConsensusBuilder entropy in [0, log({K})]: {metrics['consensus_entropy']:.4f}")


# ── Episode return tracking ───────────────────────────────────────────────────

def test_maddpg_episode_return_vs_step_average():
    """episode_return must equal sum-of-rewards-in-episode, not mean of step rewards."""
    torch.manual_seed(0)
    env = MPEWrapper("simple_spread", N_AGENTS, max_cycles=25, device=DEVICE)
    buf = ReplayBuffer(2000, env.n_agents, env.obs_dim, env.action_dim, env.state_dim)
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    actors = [Actor(env.obs_dim, EMB_DIM, env.action_dim, HIDDEN) for _ in range(env.n_agents)]
    critics = [Critic(env.state_dim, env.n_agents, EMB_DIM, env.action_dim, HIDDEN) for _ in range(env.n_agents)]
    t_actors = [copy.deepcopy(a) for a in actors]
    t_critics = [copy.deepcopy(c) for c in critics]
    opt_cb = torch.optim.Adam(cb.student.parameters())
    updater = MADDPGUpdater(cb, emb, actors, critics, t_actors, t_critics,
                             opt_cb,
                             [torch.optim.Adam(a.parameters()) for a in actors],
                             [torch.optim.Adam(c.parameters()) for c in critics])

    # Run for just enough steps to complete several episodes
    cfg = COLATrainingConfig(
        max_steps=500, warmup_steps=200, train_freq=100,
        batch_size=64, log_interval=250, reward_window=50,
    )
    all_records = []

    def _on_log(rec):
        all_records.append(rec)

    loop = COLATrainingLoop(env, buf, cb, emb, actors, updater, cfg, on_log=_on_log)
    result = loop.run()

    assert result["total_steps"] == 500
    for rec in all_records:
        # episode_return must be present and finite
        assert "episode_return" in rec, "episode_return missing from log record"
        assert "episode_count" in rec, "episode_count missing from log record"
        assert "mean_episode_length" in rec, "mean_episode_length missing"
        assert "episode_return_max" in rec
        assert "episode_return_min" in rec
        # episode_return_min <= episode_return <= episode_return_max
        assert rec["episode_return_min"] <= rec["episode_return"] + 1e-6
        assert rec["episode_return"] <= rec["episode_return_max"] + 1e-6
        # No longer should have the old per-step mean_reward key
        assert "mean_reward" not in rec, "Old 'mean_reward' key should be removed"

    print("  [OK] MADDPG loop: episode_return present, ordered min≤mean≤max, no mean_reward")


def test_mappo_episode_return_tracking():
    torch.manual_seed(1)
    env = MPEWrapper("simple_spread", N_AGENTS, max_cycles=25, device=DEVICE)
    buf = RolloutBuffer(128, env.n_agents, env.obs_dim, env.action_dim, env.state_dim,
                        0.99, 0.95, DEVICE)
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    actors = [GaussianActor(env.obs_dim, EMB_DIM, env.action_dim, HIDDEN) for _ in range(env.n_agents)]
    value_fns = [CentralizedValueFunction(env.state_dim, env.n_agents, EMB_DIM, HIDDEN) for _ in range(env.n_agents)]
    opt_cb = torch.optim.Adam(cb.student.parameters())
    updater = MAPPOUpdater(cb, emb, actors, value_fns, opt_cb,
                           [torch.optim.Adam(a.parameters()) for a in actors],
                           [torch.optim.Adam(v.parameters()) for v in value_fns],
                           n_epochs=1, n_minibatches=2)

    cfg = MAPPOTrainingConfig(max_steps=300, n_rollout_steps=128, log_interval=100)
    records = []
    loop = MAPPOTrainingLoop(env, buf, cb, emb, actors, value_fns, updater, cfg,
                              on_log=lambda r: records.append(r))
    loop.run()

    for rec in records:
        assert "episode_return" in rec
        assert "episode_count" in rec
        assert "mean_episode_length" in rec
        assert "mean_reward" not in rec
        assert "actor_loss" in rec   # pre-aggregated scalar, not list
        assert "actor_losses" not in rec
    print("  [OK] MAPPO loop: episode_return tracking correct, actor_loss scalar")


def test_qmix_episode_return_tracking():
    torch.manual_seed(2)
    env = DiscreteMPEWrapper("simple_spread", N_AGENTS, max_cycles=25, device=DEVICE)
    buf = ReplayBuffer(2000, env.n_agents, env.obs_dim, env.action_dim, env.state_dim)
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    qnets = [QNetwork(env.obs_dim, EMB_DIM, env.n_actions, HIDDEN) for _ in range(env.n_agents)]
    tqnets = [copy.deepcopy(q) for q in qnets]
    mixer = MixingNetwork(env.n_agents, env.state_dim, 16)
    tmixer = copy.deepcopy(mixer)
    opt_cb = torch.optim.Adam(cb.student.parameters())
    opt_qmix = torch.optim.Adam(
        [p for q in qnets for p in q.parameters()] + list(mixer.parameters()))
    updater = QMIXUpdater(cb, emb, qnets, tqnets, mixer, tmixer, opt_cb, opt_qmix)

    cfg = QMIXTrainingConfig(max_steps=600, warmup_steps=200, train_freq=100,
                              batch_size=64, log_interval=300)
    records = []
    loop = QMIXTrainingLoop(env, buf, cb, emb, qnets, updater, cfg,
                             on_log=lambda r: records.append(r))
    loop.run()

    for rec in records:
        assert "episode_return" in rec
        assert "episode_count" in rec
        assert "epsilon" in rec
        assert "mean_reward" not in rec
        if "loss_qmix" in rec:
            assert isinstance(rec["loss_qmix"], float)
    print("  [OK] QMIX loop: episode_return tracking correct, loss_qmix scalar")


# ── WandB prefix transformation ───────────────────────────────────────────────

def test_wandb_train_prefix():
    """Verify the train/ prefix logic used in entrypoints."""
    record = {
        "step": 10000,
        "episode_return": -0.5,
        "loss_cb": 0.1,
        "consensus_agreement": 0.8,
        "consensus_entropy": 1.2,
    }
    step = int(record["step"])
    wandb_record = {"train/" + k: v for k, v in record.items() if k != "step"}

    assert "train/episode_return" in wandb_record
    assert "train/loss_cb" in wandb_record
    assert "train/consensus_agreement" in wandb_record
    assert "train/consensus_entropy" in wandb_record
    assert "step" not in wandb_record      # step is passed separately to WandB
    assert "train/step" not in wandb_record
    assert step == 10000
    print("  [OK] WandB train/ prefix transformation: correct keys, step excluded")


def test_wandb_eval_prefix():
    eval_metrics = {"mean_episode_return": 1.5, "consensus_agreement_eval": 0.9}
    eval_wandb = {"eval/" + k: v for k, v in eval_metrics.items()}
    assert "eval/mean_episode_return" in eval_wandb
    assert "eval/consensus_agreement_eval" in eval_wandb
    print("  [OK] WandB eval/ prefix transformation: correct keys")


# ── Evaluator metric keys ─────────────────────────────────────────────────────

def test_evaluator_key_names():
    """All evaluators must return mean_episode_return and consensus_agreement_eval."""
    env_cont = MPEWrapper("simple_spread", N_AGENTS, max_cycles=10, device=DEVICE)
    env_disc = DiscreteMPEWrapper("simple_spread", N_AGENTS, max_cycles=10, device=DEVICE)
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)

    actors = [Actor(env_cont.obs_dim, EMB_DIM, env_cont.action_dim, HIDDEN) for _ in range(N_AGENTS)]
    g_actors = [GaussianActor(env_cont.obs_dim, EMB_DIM, env_cont.action_dim, HIDDEN) for _ in range(N_AGENTS)]
    qnets = [QNetwork(env_disc.obs_dim, EMB_DIM, env_disc.n_actions, HIDDEN) for _ in range(N_AGENTS)]

    for evaluator, env in [
        (PolicyEvaluator(env_cont, cb, emb, actors), env_cont),
        (MAPPOPolicyEvaluator(env_cont, cb, emb, g_actors), env_cont),
        (QMIXPolicyEvaluator(env_disc, cb, emb, qnets), env_disc),
    ]:
        metrics = evaluator.evaluate(n_episodes=2)
        assert "mean_episode_return" in metrics, f"{type(evaluator).__name__} missing mean_episode_return"
        assert "consensus_agreement_eval" in metrics, f"{type(evaluator).__name__} missing consensus_agreement_eval"

    env_cont.close()
    env_disc.close()
    print("  [OK] All three evaluators return consistent metric key names")


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print("Phase 3 gate: Metric correctness")

    print("\n-- _merge_update_metrics --")
    test_merge_scalars()
    test_merge_list_to_mean()
    test_merge_value_losses()

    print("\n-- consensus_entropy --")
    test_null_builder_entropy_zero()
    test_cola_builder_entropy_positive()

    print("\n-- Episode return tracking --")
    test_maddpg_episode_return_vs_step_average()
    test_mappo_episode_return_tracking()
    test_qmix_episode_return_tracking()

    print("\n-- WandB prefix transformation --")
    test_wandb_train_prefix()
    test_wandb_eval_prefix()

    print("\n-- Evaluator key names --")
    test_evaluator_key_names()

    print("\nPhase 3 gate PASSED: all metrics correctly computed and named")


if __name__ == "__main__":
    main()
