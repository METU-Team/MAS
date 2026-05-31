"""Phase 1 gate test: COLA-MAPPO — all modules and the integrated training loop."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.rollout_buffer import RolloutBuffer
from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.critics.value_function import CentralizedValueFunction
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.evaluation.mappo_policy_evaluator import MAPPOPolicyEvaluator
from cola_framework.interfaces.evaluator import EvaluatorModule
from cola_framework.interfaces.policy import PolicyModule
from cola_framework.interfaces.updater import UpdateModule
from cola_framework.interfaces.value import ValueModule
from cola_framework.loops.mappo_training_loop import MAPPOTrainingConfig, MAPPOTrainingLoop
from cola_framework.interfaces.training_loop import TrainingLoopModule
from cola_framework.policies.gaussian_actor import GaussianActor
from cola_framework.trainers.mappo_updater import MAPPOUpdater

# ── constants ─────────────────────────────────────────────────────────────────

N_AGENTS = 3
OBS_DIM = 18
ACTION_DIM = 5
STATE_DIM = OBS_DIM * N_AGENTS
EMB_DIM = 8
HIDDEN = 32
K = 4
T = 128     # rollout steps for tests
DEVICE = "cpu"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_obs(B=1):
    return torch.randn(B, N_AGENTS, OBS_DIM)

def _make_state(B=1):
    return torch.randn(B, STATE_DIM)

def _make_buffer(n_steps=T):
    return RolloutBuffer(n_steps, N_AGENTS, OBS_DIM, ACTION_DIM, STATE_DIM,
                         gamma=0.99, gae_lambda=0.95, device=DEVICE)

def _make_gaussian_actor():
    return GaussianActor(OBS_DIM, EMB_DIM, ACTION_DIM, HIDDEN)

def _make_value_fn():
    return CentralizedValueFunction(STATE_DIM, N_AGENTS, EMB_DIM, HIDDEN)


# ── ValueModule interface ─────────────────────────────────────────────────────

def test_value_module_interface():
    vf = _make_value_fn()
    assert isinstance(vf, ValueModule)
    assert isinstance(vf, torch.nn.Module)
    print("  [OK] CentralizedValueFunction implements ValueModule and nn.Module")


def test_value_function_forward():
    vf = _make_value_fn()
    state = torch.randn(16, STATE_DIM)
    all_emb = torch.randn(16, N_AGENTS * EMB_DIM)
    out = vf(state, all_emb)
    assert out.shape == (16, 1), f"Expected (16,1), got {out.shape}"
    print("  [OK] CentralizedValueFunction forward: [B,1] output")


def test_value_function_single():
    vf = _make_value_fn()
    state = torch.randn(STATE_DIM)
    all_emb = torch.randn(N_AGENTS * EMB_DIM)
    out = vf(state, all_emb)
    assert out.shape == (1,), f"Expected (1,), got {out.shape}"
    print("  [OK] CentralizedValueFunction forward: [1] output for unbatched input")


def test_value_function_wrong_dim_raises():
    vf = _make_value_fn()
    try:
        vf(torch.randn(3, STATE_DIM), torch.randn(N_AGENTS * EMB_DIM))
        assert False, "Should raise ValueError on rank mismatch"
    except ValueError:
        pass
    print("  [OK] CentralizedValueFunction raises on rank mismatch")


# ── GaussianActor ─────────────────────────────────────────────────────────────

def test_gaussian_actor_interface():
    actor = _make_gaussian_actor()
    assert isinstance(actor, PolicyModule)
    assert isinstance(actor, torch.nn.Module)
    print("  [OK] GaussianActor implements PolicyModule and nn.Module")


def test_gaussian_actor_forward_bounded():
    actor = _make_gaussian_actor()
    obs = torch.randn(16, OBS_DIM)
    cemb = torch.randn(16, EMB_DIM)
    action = actor(obs, cemb)
    assert action.shape == (16, ACTION_DIM), f"Expected (16,{ACTION_DIM}), got {action.shape}"
    assert action.abs().max().item() <= 1.0 + 1e-5, "forward() must be in [-1,1] (tanh)"
    print("  [OK] GaussianActor.forward(): bounded in [-1,1], correct shape")


def test_gaussian_actor_sample():
    actor = _make_gaussian_actor()
    obs = torch.randn(OBS_DIM)
    cemb = torch.randn(EMB_DIM)
    action, log_prob = actor.sample(obs, cemb)
    assert action.shape == (ACTION_DIM,), f"action shape: {action.shape}"
    assert log_prob.ndim == 0, f"log_prob should be scalar, got shape {log_prob.shape}"
    assert action.abs().max().item() <= 1.0 + 1e-5, "sample() must be in [-1,1]"
    print("  [OK] GaussianActor.sample(): correct shapes and bounded action")


def test_gaussian_actor_evaluate_actions_consistency():
    """log_prob from evaluate_actions must be close to log_prob from sample."""
    torch.manual_seed(7)
    actor = _make_gaussian_actor()
    obs = torch.randn(OBS_DIM)
    cemb = torch.randn(EMB_DIM)
    with torch.no_grad():
        action, lp_sample = actor.sample(obs, cemb)
        lp_eval, _ = actor.evaluate_actions(obs.unsqueeze(0), cemb.unsqueeze(0), action.unsqueeze(0))
    assert abs(float(lp_sample.item()) - float(lp_eval.squeeze().item())) < 1e-4, \
        f"sample log_prob {lp_sample.item()} vs evaluate_actions {lp_eval.item()}"
    print("  [OK] GaussianActor.evaluate_actions() consistent with sample()")


def test_gaussian_actor_evaluate_gradient_flows():
    """Gradients must flow from evaluate_actions() through actor params."""
    actor = _make_gaussian_actor()
    obs = torch.randn(8, OBS_DIM)
    cemb = torch.randn(8, EMB_DIM)
    actions = torch.randn(8, ACTION_DIM).tanh()
    log_prob, entropy = actor.evaluate_actions(obs, cemb, actions)
    loss = -log_prob.mean() - 0.01 * entropy.mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in actor.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed"
    assert all(g >= 0 for g in grad_norms), "Negative gradient norms"
    print("  [OK] GaussianActor.evaluate_actions(): gradients flow correctly")


# ── RolloutBuffer ─────────────────────────────────────────────────────────────

def test_rollout_buffer_push_and_len():
    buf = _make_buffer(n_steps=10)
    assert len(buf) == 0
    for _ in range(10):
        buf.push(
            torch.randn(N_AGENTS, OBS_DIM), torch.randn(STATE_DIM),
            torch.randn(N_AGENTS, ACTION_DIM).tanh(), torch.randn(N_AGENTS),
            torch.zeros(N_AGENTS), torch.randn(N_AGENTS), torch.randn(N_AGENTS),
        )
    assert len(buf) == 10
    assert buf.is_full()
    print("  [OK] RolloutBuffer.push(): length tracking and is_full() correct")


def test_rollout_buffer_overflow_raises():
    buf = _make_buffer(n_steps=5)
    for _ in range(5):
        buf.push(torch.zeros(N_AGENTS, OBS_DIM), torch.zeros(STATE_DIM),
                 torch.zeros(N_AGENTS, ACTION_DIM), torch.zeros(N_AGENTS),
                 torch.zeros(N_AGENTS), torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
    try:
        buf.push(torch.zeros(N_AGENTS, OBS_DIM), torch.zeros(STATE_DIM),
                 torch.zeros(N_AGENTS, ACTION_DIM), torch.zeros(N_AGENTS),
                 torch.zeros(N_AGENTS), torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("  [OK] RolloutBuffer.push(): raises RuntimeError when full")


def test_rollout_buffer_clear():
    buf = _make_buffer(n_steps=5)
    for _ in range(5):
        buf.push(torch.zeros(N_AGENTS, OBS_DIM), torch.zeros(STATE_DIM),
                 torch.zeros(N_AGENTS, ACTION_DIM), torch.zeros(N_AGENTS),
                 torch.zeros(N_AGENTS), torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_full()
    print("  [OK] RolloutBuffer.clear(): resets pointer correctly")


def _fill_buffer(buf, n_steps):
    for _ in range(n_steps):
        buf.push(
            torch.randn(N_AGENTS, OBS_DIM), torch.randn(STATE_DIM),
            torch.randn(N_AGENTS, ACTION_DIM).tanh(), torch.rand(N_AGENTS),
            torch.zeros(N_AGENTS), torch.randn(N_AGENTS), torch.randn(N_AGENTS),
        )


def test_rollout_buffer_gae():
    buf = _make_buffer(n_steps=T)
    _fill_buffer(buf, T)
    last_values = torch.randn(N_AGENTS)
    last_dones = torch.zeros(N_AGENTS)
    buf.compute_returns_and_advantages(last_values, last_dones)

    batch = buf.get()
    assert batch["advantages"].shape == (T, N_AGENTS)
    assert batch["returns"].shape == (T, N_AGENTS)
    # returns = advantages + values (by definition)
    diff = (batch["returns"] - batch["advantages"] - buf.values).abs().max()
    assert diff < 1e-5, f"returns != advantages + values, max diff: {diff}"
    print("  [OK] RolloutBuffer.compute_returns_and_advantages(): GAE shapes and identity correct")


def test_rollout_buffer_gae_empty_raises():
    buf = _make_buffer(n_steps=T)
    try:
        buf.compute_returns_and_advantages(torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
        assert False
    except RuntimeError:
        pass
    print("  [OK] RolloutBuffer.compute_returns_and_advantages(): raises when buffer is empty")


def test_rollout_buffer_gae_partial():
    """GAE must work on a partial buffer (ptr < n_steps)."""
    buf = _make_buffer(n_steps=T)
    _fill_buffer(buf, T // 2)
    buf.compute_returns_and_advantages(torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
    batch = buf.get()
    assert batch["advantages"].shape == (T // 2, N_AGENTS)
    print("  [OK] RolloutBuffer.compute_returns_and_advantages(): works on partial buffer")


def test_rollout_buffer_gae_terminal():
    """With all dones=1, advantages must equal delta = r - V (no bootstrapping)."""
    buf = _make_buffer(n_steps=2)
    rewards = torch.tensor([1.0, 1.0, 1.0])   # [n_agents]
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.ones(N_AGENTS)               # all terminal
    for _ in range(2):
        buf.push(torch.zeros(N_AGENTS, OBS_DIM), torch.zeros(STATE_DIM),
                 torch.zeros(N_AGENTS, ACTION_DIM), rewards, dones, values,
                 torch.zeros(N_AGENTS))
    last_values = torch.ones(N_AGENTS) * 999.0  # should NOT contribute
    buf.compute_returns_and_advantages(last_values, dones)
    # At the last step: delta = r - V = 0.5 (no bootstrapping because done=1)
    last_adv = buf.advantages[-1]
    expected = rewards - values  # delta with no bootstrap
    assert (last_adv - expected).abs().max() < 1e-5, \
        f"terminal GAE wrong: {last_adv} vs expected {expected}"
    print("  [OK] RolloutBuffer GAE: terminal episodes correctly zero-out bootstrap")


def test_rollout_buffer_minibatches():
    buf = _make_buffer(n_steps=T)
    _fill_buffer(buf, T)
    buf.compute_returns_and_advantages(torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
    n_mb = 4
    batches = list(buf.get_minibatches(n_mb))
    assert len(batches) == n_mb
    mb_size = T // n_mb
    for mb in batches:
        assert mb["obs"].shape == (mb_size, N_AGENTS, OBS_DIM)
        assert mb["state"].shape == (mb_size, STATE_DIM)
        assert mb["actions"].shape == (mb_size, N_AGENTS, ACTION_DIM)
        assert mb["advantages"].shape == (mb_size, N_AGENTS)
        assert mb["returns"].shape == (mb_size, N_AGENTS)
        assert mb["old_log_probs"].shape == (mb_size, N_AGENTS)
    print("  [OK] RolloutBuffer.get_minibatches(): correct number and shapes")


# ── MAPPOUpdater ──────────────────────────────────────────────────────────────

def test_mappo_updater_interface():
    actors = [_make_gaussian_actor() for _ in range(N_AGENTS)]
    value_fns = [_make_value_fn() for _ in range(N_AGENTS)]
    cb = ConsensusBuilder(OBS_DIM, K, HIDDEN)
    emb = ConsensusEmbedding(K, EMB_DIM)
    updater = MAPPOUpdater(
        cb, emb, actors, value_fns,
        torch.optim.Adam(cb.student.parameters(), lr=3e-4),
        [torch.optim.Adam(a.parameters(), lr=3e-4) for a in actors],
        [torch.optim.Adam(v.parameters(), lr=1e-3) for v in value_fns],
    )
    assert isinstance(updater, UpdateModule)
    print("  [OK] MAPPOUpdater implements UpdateModule")


def test_mappo_updater_update_runs():
    torch.manual_seed(42)
    actors = [_make_gaussian_actor() for _ in range(N_AGENTS)]
    value_fns = [_make_value_fn() for _ in range(N_AGENTS)]
    cb = ConsensusBuilder(OBS_DIM, K, HIDDEN)
    emb = ConsensusEmbedding(K, EMB_DIM)
    updater = MAPPOUpdater(
        cb, emb, actors, value_fns,
        torch.optim.Adam(cb.student.parameters(), lr=3e-4),
        [torch.optim.Adam(a.parameters(), lr=3e-4) for a in actors],
        [torch.optim.Adam(v.parameters(), lr=1e-3) for v in value_fns],
        n_epochs=2, n_minibatches=2,
    )
    buf = _make_buffer(n_steps=64)
    _fill_buffer(buf, 64)
    buf.compute_returns_and_advantages(torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
    batch = buf.get()
    metrics = updater.update(batch)

    assert "loss_cb" in metrics
    assert "consensus_agreement" in metrics
    assert "actor_losses" in metrics and len(metrics["actor_losses"]) == N_AGENTS
    assert "value_losses" in metrics and len(metrics["value_losses"]) == N_AGENTS
    assert 0.0 <= metrics["consensus_agreement"] <= 1.0
    print("  [OK] MAPPOUpdater.update(): runs and returns correct metric keys")


def test_mappo_updater_null_cola():
    """With NullConsensusBuilder, loss_cb must be exactly 0.0."""
    actors = [_make_gaussian_actor() for _ in range(N_AGENTS)]
    value_fns = [_make_value_fn() for _ in range(N_AGENTS)]
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    updater = MAPPOUpdater(
        cb, emb, actors, value_fns,
        torch.optim.Adam(cb.student.parameters(), lr=3e-4),
        [torch.optim.Adam(a.parameters(), lr=3e-4) for a in actors],
        [torch.optim.Adam(v.parameters(), lr=1e-3) for v in value_fns],
        n_epochs=1, n_minibatches=2,
    )
    buf = _make_buffer(n_steps=32)
    _fill_buffer(buf, 32)
    buf.compute_returns_and_advantages(torch.zeros(N_AGENTS), torch.zeros(N_AGENTS))
    metrics = updater.update(buf.get())
    assert abs(metrics["loss_cb"]) < 1e-7
    assert abs(metrics["consensus_agreement"] - 1.0) < 1e-6
    print("  [OK] MAPPOUpdater with NullConsensusBuilder: loss_cb=0, agreement=1")


# ── MAPPOTrainingLoop + MAPPOPolicyEvaluator — integrated run ─────────────────

def test_mappo_evaluator_interface():
    env = MPEWrapper("simple_spread", 3, max_cycles=25, device=DEVICE)
    actors = [GaussianActor(env.obs_dim, EMB_DIM, env.action_dim, HIDDEN) for _ in range(env.n_agents)]
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    evaluator = MAPPOPolicyEvaluator(env, cb, emb, actors)
    assert isinstance(evaluator, EvaluatorModule)
    print("  [OK] MAPPOPolicyEvaluator implements EvaluatorModule")


def test_integrated_mappo_loop():
    """Run a minimal COLA-MAPPO end-to-end training loop."""
    torch.manual_seed(0)

    env = MPEWrapper("simple_spread", n_agents=3, max_cycles=25, device=DEVICE)

    buf = RolloutBuffer(
        n_steps=128, n_agents=env.n_agents, obs_dim=env.obs_dim,
        action_dim=env.action_dim, state_dim=env.state_dim,
        gamma=0.99, gae_lambda=0.95, device=DEVICE,
    )

    cb = ConsensusBuilder(env.obs_dim, K, HIDDEN)
    emb = ConsensusEmbedding(K, EMB_DIM)
    actors = [GaussianActor(env.obs_dim, EMB_DIM, env.action_dim, HIDDEN) for _ in range(env.n_agents)]
    value_fns = [CentralizedValueFunction(env.state_dim, env.n_agents, EMB_DIM, HIDDEN) for _ in range(env.n_agents)]

    updater = MAPPOUpdater(
        cb, emb, actors, value_fns,
        torch.optim.Adam(cb.student.parameters(), lr=3e-4),
        [torch.optim.Adam(a.parameters(), lr=3e-4) for a in actors],
        [torch.optim.Adam(v.parameters(), lr=1e-3) for v in value_fns],
        n_epochs=2, n_minibatches=2,
    )

    cfg = MAPPOTrainingConfig(max_steps=300, n_rollout_steps=128, log_interval=200)
    metrics_log = []

    def _on_log(rec):
        metrics_log.append(rec)

    loop = MAPPOTrainingLoop(env, buf, cb, emb, actors, value_fns, updater, cfg, on_log=_on_log)
    assert isinstance(loop, TrainingLoopModule)

    result = loop.run()

    assert result["total_steps"] >= 256  # at least 2 full rollouts
    assert result["update_count"] >= 2
    assert len(metrics_log) >= 1

    # Evaluator
    evaluator = MAPPOPolicyEvaluator(env, cb, emb, actors)
    eval_metrics = evaluator.evaluate(n_episodes=3)
    assert "mean_episode_return" in eval_metrics
    assert "consensus_agreement_eval" in eval_metrics
    print("  [OK] Integrated COLA-MAPPO loop: runs, produces metrics, evaluator works")


def test_integrated_mappo_no_cola():
    """NullConsensusBuilder path must also run to completion."""
    torch.manual_seed(1)
    env = MPEWrapper("simple_spread", n_agents=3, max_cycles=25, device=DEVICE)
    buf = RolloutBuffer(128, env.n_agents, env.obs_dim, env.action_dim, env.state_dim,
                        0.99, 0.95, DEVICE)
    cb = NullConsensusBuilder(K)
    emb = ConsensusEmbedding(K, EMB_DIM)
    actors = [GaussianActor(env.obs_dim, EMB_DIM, env.action_dim, HIDDEN) for _ in range(env.n_agents)]
    value_fns = [CentralizedValueFunction(env.state_dim, env.n_agents, EMB_DIM, HIDDEN) for _ in range(env.n_agents)]

    updater = MAPPOUpdater(
        cb, emb, actors, value_fns,
        torch.optim.Adam(cb.student.parameters(), lr=3e-4),
        [torch.optim.Adam(a.parameters(), lr=3e-4) for a in actors],
        [torch.optim.Adam(v.parameters(), lr=1e-3) for v in value_fns],
        n_epochs=2, n_minibatches=2,
    )
    cfg = MAPPOTrainingConfig(max_steps=256, n_rollout_steps=128, log_interval=200)
    loop = MAPPOTrainingLoop(env, buf, cb, emb, actors, value_fns, updater, cfg)
    result = loop.run()
    assert result["total_steps"] >= 256
    print("  [OK] Vanilla MAPPO (--no_cola) loop runs to completion")


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print("Phase 1 gate: COLA-MAPPO modules")

    print("\n-- ValueModule / CentralizedValueFunction --")
    test_value_module_interface()
    test_value_function_forward()
    test_value_function_single()
    test_value_function_wrong_dim_raises()

    print("\n-- GaussianActor --")
    test_gaussian_actor_interface()
    test_gaussian_actor_forward_bounded()
    test_gaussian_actor_sample()
    test_gaussian_actor_evaluate_actions_consistency()
    test_gaussian_actor_evaluate_gradient_flows()

    print("\n-- RolloutBuffer --")
    test_rollout_buffer_push_and_len()
    test_rollout_buffer_overflow_raises()
    test_rollout_buffer_clear()
    test_rollout_buffer_gae()
    test_rollout_buffer_gae_empty_raises()
    test_rollout_buffer_gae_partial()
    test_rollout_buffer_gae_terminal()
    test_rollout_buffer_minibatches()

    print("\n-- MAPPOUpdater --")
    test_mappo_updater_interface()
    test_mappo_updater_update_runs()
    test_mappo_updater_null_cola()

    print("\n-- Integration --")
    test_mappo_evaluator_interface()
    test_integrated_mappo_loop()
    test_integrated_mappo_no_cola()

    print("\nPhase 1 gate PASSED: COLA-MAPPO OK")


if __name__ == "__main__":
    main()
