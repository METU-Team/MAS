"""Phase 2 gate test: COLA-QMIX — all modules and the integrated training loop."""

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
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.critics.mixing_network import MixingNetwork
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.discrete_mpe_wrapper import DiscreteMPEWrapper
from cola_framework.evaluation.qmix_policy_evaluator import QMIXPolicyEvaluator
from cola_framework.interfaces.environment import MultiAgentEnvironment
from cola_framework.interfaces.evaluator import EvaluatorModule
from cola_framework.interfaces.q_network import QNetworkModule
from cola_framework.interfaces.training_loop import TrainingLoopModule
from cola_framework.interfaces.updater import UpdateModule
from cola_framework.loops.qmix_training_loop import QMIXTrainingConfig, QMIXTrainingLoop
from cola_framework.policies.q_network import QNetwork
from cola_framework.trainers.qmix_updater import QMIXUpdater

# ── constants ─────────────────────────────────────────────────────────────────

N_AGENTS = 3
N_ACTIONS = 5
OBS_DIM = 18
STATE_DIM = OBS_DIM * N_AGENTS
EMB_DIM = 8
HIDDEN = 32
MIX_HIDDEN = 16
K = 4
B = 32
DEVICE = "cpu"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_qnet():
    return QNetwork(OBS_DIM, EMB_DIM, N_ACTIONS, HIDDEN)

def _make_mixer():
    return MixingNetwork(N_AGENTS, STATE_DIM, MIX_HIDDEN)

def _make_env():
    return DiscreteMPEWrapper("simple_spread", N_AGENTS, max_cycles=25, device=DEVICE)

def _make_cb():
    return ConsensusBuilder(OBS_DIM, K, HIDDEN)

def _make_emb():
    return ConsensusEmbedding(K, EMB_DIM)

def _make_updater(env):
    cb = _make_cb()
    emb = _make_emb()
    qnets = [_make_qnet() for _ in range(env.n_agents)]
    tqnets = [copy.deepcopy(q) for q in qnets]
    mixer = _make_mixer()
    tmixer = copy.deepcopy(mixer)
    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_qmix = torch.optim.Adam(
        [p for q in qnets for p in q.parameters()] + list(mixer.parameters()),
        lr=5e-4,
    )
    updater = QMIXUpdater(cb, emb, qnets, tqnets, mixer, tmixer, opt_cb, opt_qmix)
    return updater, cb, emb, qnets


# ── DiscreteMPEWrapper ────────────────────────────────────────────────────────

def test_discrete_wrapper_interface():
    env = _make_env()
    assert isinstance(env, MultiAgentEnvironment)
    print("  [OK] DiscreteMPEWrapper implements MultiAgentEnvironment")


def test_discrete_wrapper_attributes():
    env = _make_env()
    assert env.n_agents == N_AGENTS
    assert env.n_actions == N_ACTIONS
    assert env.action_dim == 1
    assert env.obs_dim == OBS_DIM
    assert env.state_dim == STATE_DIM
    print("  [OK] DiscreteMPEWrapper attributes: n_agents, n_actions, action_dim, obs_dim, state_dim")


def test_discrete_wrapper_reset():
    env = _make_env()
    obs, state = env.reset()
    assert obs.shape == (N_AGENTS, OBS_DIM), f"obs shape: {obs.shape}"
    assert state.shape == (STATE_DIM,), f"state shape: {state.shape}"
    print("  [OK] DiscreteMPEWrapper.reset(): correct obs and state shapes")


def test_discrete_wrapper_step():
    env = _make_env()
    env.reset()
    actions = torch.randint(0, N_ACTIONS, (N_AGENTS,))
    next_obs, next_state, rewards, dones = env.step(actions)
    assert next_obs.shape == (N_AGENTS, OBS_DIM)
    assert next_state.shape == (STATE_DIM,)
    assert rewards.shape == (N_AGENTS,)
    assert dones.shape == (N_AGENTS,)
    assert dones.dtype == torch.bool
    print("  [OK] DiscreteMPEWrapper.step(): correct output shapes and dtypes")


def test_discrete_wrapper_actions_in_range():
    """All valid action indices [0, n_actions) must not raise."""
    env = _make_env()
    for action_idx in range(N_ACTIONS):
        env.reset()
        actions = torch.full((N_AGENTS,), action_idx, dtype=torch.int64)
        env.step(actions)   # must not raise
    print(f"  [OK] DiscreteMPEWrapper.step(): accepts all {N_ACTIONS} action indices")


# ── QNetwork ──────────────────────────────────────────────────────────────────

def test_qnetwork_interface():
    qnet = _make_qnet()
    assert isinstance(qnet, QNetworkModule)
    assert isinstance(qnet, torch.nn.Module)
    print("  [OK] QNetwork implements QNetworkModule and nn.Module")


def test_qnetwork_forward_batched():
    qnet = _make_qnet()
    obs = torch.randn(B, OBS_DIM)
    cemb = torch.randn(B, EMB_DIM)
    out = qnet(obs, cemb)
    assert out.shape == (B, N_ACTIONS), f"Expected ({B},{N_ACTIONS}), got {out.shape}"
    print("  [OK] QNetwork.forward(): [B, n_actions] output")


def test_qnetwork_forward_unbatched():
    qnet = _make_qnet()
    obs = torch.randn(OBS_DIM)
    cemb = torch.randn(EMB_DIM)
    out = qnet(obs, cemb)
    assert out.shape == (N_ACTIONS,), f"Expected ({N_ACTIONS},), got {out.shape}"
    print("  [OK] QNetwork.forward(): [n_actions] output for unbatched input")


def test_qnetwork_gradients():
    qnet = _make_qnet()
    obs = torch.randn(B, OBS_DIM)
    cemb = torch.randn(B, EMB_DIM)
    loss = qnet(obs, cemb).mean()
    loss.backward()
    grads = [p.grad for p in qnet.parameters() if p.grad is not None]
    assert len(grads) > 0
    print("  [OK] QNetwork: gradients flow correctly")


# ── MixingNetwork ─────────────────────────────────────────────────────────────

def test_mixing_network_output_shape():
    mixer = _make_mixer()
    qs = torch.randn(B, N_AGENTS)
    state = torch.randn(B, STATE_DIM)
    q_tot = mixer(qs, state)
    assert q_tot.shape == (B, 1), f"Expected ({B},1), got {q_tot.shape}"
    print("  [OK] MixingNetwork.forward(): [B,1] output")


def test_mixing_network_monotonicity():
    """Q_tot must be non-decreasing in each individual Q-value."""
    torch.manual_seed(99)
    mixer = _make_mixer()
    mixer.eval()
    state = torch.randn(1, STATE_DIM)
    base_qs = torch.zeros(1, N_AGENTS)

    q_base = mixer(base_qs, state).item()
    for a in range(N_AGENTS):
        qs_higher = base_qs.clone()
        qs_higher[0, a] = 5.0   # increase one agent's Q-value
        q_higher = mixer(qs_higher, state).item()
        assert q_higher >= q_base - 1e-5, \
            f"Monotonicity violated for agent {a}: {q_higher:.4f} < {q_base:.4f}"
    print("  [OK] MixingNetwork: monotonicity holds (Q_tot non-decreasing in each Q_a)")


def test_mixing_network_gradients():
    mixer = _make_mixer()
    qs = torch.randn(B, N_AGENTS, requires_grad=True)
    state = torch.randn(B, STATE_DIM)
    q_tot = mixer(qs, state)
    q_tot.sum().backward()
    assert qs.grad is not None
    print("  [OK] MixingNetwork: gradients flow through individual Q-values")


# ── QMIXUpdater ───────────────────────────────────────────────────────────────

def test_qmix_updater_interface():
    env = _make_env()
    updater, *_ = _make_updater(env)
    assert isinstance(updater, UpdateModule)
    print("  [OK] QMIXUpdater implements UpdateModule")


def _make_fake_batch(B=32, device=DEVICE):
    """Synthetic batch matching ReplayBuffer.sample() output for QMIX."""
    return {
        "obs": torch.randn(B, N_AGENTS, OBS_DIM, device=device),
        "state": torch.randn(B, STATE_DIM, device=device),
        "actions": torch.randint(0, N_ACTIONS, (B, N_AGENTS, 1)).float(),
        "rewards": torch.randn(B, N_AGENTS, device=device),
        "next_obs": torch.randn(B, N_AGENTS, OBS_DIM, device=device),
        "next_state": torch.randn(B, STATE_DIM, device=device),
        "dones": torch.zeros(B, N_AGENTS, device=device),
    }


def test_qmix_updater_runs():
    torch.manual_seed(42)
    env = _make_env()
    updater, *_ = _make_updater(env)
    batch = _make_fake_batch()
    metrics = updater.update(batch)
    assert "loss_cb" in metrics
    assert "loss_qmix" in metrics
    assert "consensus_agreement" in metrics
    assert isinstance(metrics["loss_qmix"], float)
    assert 0.0 <= metrics["consensus_agreement"] <= 1.0
    print("  [OK] QMIXUpdater.update(): runs and returns correct metric keys")


def test_qmix_updater_null_cola():
    cb = NullConsensusBuilder(K)
    emb = _make_emb()
    qnets = [_make_qnet() for _ in range(N_AGENTS)]
    tqnets = [copy.deepcopy(q) for q in qnets]
    mixer = _make_mixer()
    tmixer = copy.deepcopy(mixer)
    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_qmix = torch.optim.Adam(
        [p for q in qnets for p in q.parameters()] + list(mixer.parameters()), lr=5e-4
    )
    updater = QMIXUpdater(cb, emb, qnets, tqnets, mixer, tmixer, opt_cb, opt_qmix)
    metrics = updater.update(_make_fake_batch())
    assert abs(metrics["loss_cb"]) < 1e-7
    assert abs(metrics["consensus_agreement"] - 1.0) < 1e-6
    print("  [OK] QMIXUpdater with NullConsensusBuilder: loss_cb=0, agreement=1")


def test_qmix_updater_target_drift():
    """Target network params must change (Polyak) after update, but slowly."""
    torch.manual_seed(7)
    env = _make_env()
    updater, _, _, qnets = _make_updater(env)
    # Snapshot target params before update
    before = [p.clone() for p in updater.target_q_networks[0].parameters()]
    updater.update(_make_fake_batch())
    after = list(updater.target_q_networks[0].parameters())
    delta = sum((b - a).abs().sum().item() for b, a in zip(before, after))
    assert delta > 0, "Target network did not update at all"
    # With tau=0.005 (small), delta should be small but non-zero
    online_params = list(qnets[0].parameters())
    max_possible = sum(p.abs().sum().item() for p in online_params)
    assert delta < max_possible, "Target update seems too large"
    print("  [OK] QMIXUpdater: target networks updated (Polyak) — small but non-zero delta")


# ── QMIXTrainingLoop + QMIXPolicyEvaluator — integrated run ──────────────────

def test_qmix_evaluator_interface():
    env = _make_env()
    cb = NullConsensusBuilder(K)
    emb = _make_emb()
    qnets = [_make_qnet() for _ in range(env.n_agents)]
    evaluator = QMIXPolicyEvaluator(env, cb, emb, qnets)
    assert isinstance(evaluator, EvaluatorModule)
    print("  [OK] QMIXPolicyEvaluator implements EvaluatorModule")


def test_qmix_training_loop_interface():
    env = _make_env()
    buf = ReplayBuffer(200, env.n_agents, env.obs_dim, env.action_dim, env.state_dim)
    updater, cb, emb, qnets = _make_updater(env)
    cfg = QMIXTrainingConfig(max_steps=10, warmup_steps=5, train_freq=5, batch_size=8)
    loop = QMIXTrainingLoop(env, buf, cb, emb, qnets, updater, cfg)
    assert isinstance(loop, TrainingLoopModule)
    print("  [OK] QMIXTrainingLoop implements TrainingLoopModule")


def test_integrated_qmix_loop():
    """Full COLA-QMIX end-to-end training loop."""
    torch.manual_seed(0)
    env = _make_env()
    buf = ReplayBuffer(
        capacity=2000, n_agents=env.n_agents, obs_dim=env.obs_dim,
        action_dim=env.action_dim, state_dim=env.state_dim, device=DEVICE,
    )
    cb = _make_cb()
    emb = _make_emb()
    qnets = [_make_qnet() for _ in range(env.n_agents)]
    tqnets = [copy.deepcopy(q) for q in qnets]
    mixer = _make_mixer()
    tmixer = copy.deepcopy(mixer)
    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_qmix = torch.optim.Adam(
        [p for q in qnets for p in q.parameters()] + list(mixer.parameters()), lr=5e-4
    )
    updater = QMIXUpdater(cb, emb, qnets, tqnets, mixer, tmixer, opt_cb, opt_qmix)

    cfg = QMIXTrainingConfig(
        max_steps=600, warmup_steps=200, train_freq=100,
        batch_size=64, log_interval=300,
    )
    metrics_log = []

    def _on_log(rec):
        metrics_log.append(rec)

    loop = QMIXTrainingLoop(env, buf, cb, emb, qnets, updater, cfg, on_log=_on_log)
    result = loop.run()

    assert result["total_steps"] == 600
    assert result["train_steps"] >= 1
    assert len(metrics_log) >= 1

    for rec in metrics_log:
        if "loss_qmix" in rec:
            assert isinstance(rec["loss_qmix"], float)

    # Evaluator
    evaluator = QMIXPolicyEvaluator(env, cb, emb, qnets)
    eval_metrics = evaluator.evaluate(n_episodes=3)
    assert "mean_episode_return" in eval_metrics
    assert "consensus_agreement_eval" in eval_metrics
    print("  [OK] Integrated COLA-QMIX loop: runs, produces metrics, evaluator works")


def test_integrated_qmix_no_cola():
    """Vanilla QMIX (NullConsensusBuilder) must also run to completion."""
    torch.manual_seed(1)
    env = _make_env()
    buf = ReplayBuffer(2000, env.n_agents, env.obs_dim, env.action_dim, env.state_dim)
    cb = NullConsensusBuilder(K)
    emb = _make_emb()
    qnets = [_make_qnet() for _ in range(env.n_agents)]
    tqnets = [copy.deepcopy(q) for q in qnets]
    mixer = _make_mixer()
    tmixer = copy.deepcopy(mixer)
    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_qmix = torch.optim.Adam(
        [p for q in qnets for p in q.parameters()] + list(mixer.parameters()), lr=5e-4
    )
    updater = QMIXUpdater(cb, emb, qnets, tqnets, mixer, tmixer, opt_cb, opt_qmix)
    cfg = QMIXTrainingConfig(max_steps=400, warmup_steps=200, train_freq=100, batch_size=64)
    loop = QMIXTrainingLoop(env, buf, cb, emb, qnets, updater, cfg)
    result = loop.run()
    assert result["total_steps"] == 400
    print("  [OK] Vanilla QMIX (--no_cola) loop runs to completion")


def test_action_storage_roundtrip():
    """Action indices must survive the float storage round-trip without error."""
    actions_int = torch.randint(0, N_ACTIONS, (N_AGENTS,))
    stored = actions_int.float().unsqueeze(-1)           # [n_agents, 1]
    recovered = stored.squeeze(-1).long()                # [n_agents]
    assert (recovered == actions_int).all(), \
        f"Roundtrip failed: {actions_int} -> {recovered}"
    print("  [OK] Action index float storage round-trip: int64 → float → int64 exact")


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print("Phase 2 gate: COLA-QMIX modules")

    print("\n-- DiscreteMPEWrapper --")
    test_discrete_wrapper_interface()
    test_discrete_wrapper_attributes()
    test_discrete_wrapper_reset()
    test_discrete_wrapper_step()
    test_discrete_wrapper_actions_in_range()

    print("\n-- QNetwork --")
    test_qnetwork_interface()
    test_qnetwork_forward_batched()
    test_qnetwork_forward_unbatched()
    test_qnetwork_gradients()

    print("\n-- MixingNetwork --")
    test_mixing_network_output_shape()
    test_mixing_network_monotonicity()
    test_mixing_network_gradients()

    print("\n-- QMIXUpdater --")
    test_qmix_updater_interface()
    test_qmix_updater_runs()
    test_qmix_updater_null_cola()
    test_qmix_updater_target_drift()

    print("\n-- Integration --")
    test_qmix_evaluator_interface()
    test_qmix_training_loop_interface()
    test_integrated_qmix_loop()
    test_integrated_qmix_no_cola()
    test_action_storage_roundtrip()

    print("\nPhase 2 gate PASSED: COLA-QMIX OK")


if __name__ == "__main__":
    main()
