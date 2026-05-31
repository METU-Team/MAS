"""Phase 0 gate test: NullConsensusBuilder and --no_cola flag in train_cola.py."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.null_builder import NullConsensusBuilder
from cola_framework.consensus import NullConsensusBuilder as NullFromInit
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.interfaces.consensus import ConsensusModule


# ── helpers ──────────────────────────────────────────────────────────────────

B, N, OBS = 32, 3, 18
K = 4


def _make_obs():
    return torch.randn(B, N, OBS)


# ── test 1: interface compliance ──────────────────────────────────────────────

def test_implements_interface():
    builder = NullConsensusBuilder(k=K)
    assert isinstance(builder, ConsensusModule), "Must implement ConsensusModule"
    assert isinstance(builder, torch.nn.Module), "Must be an nn.Module"
    print("  [OK] implements ConsensusModule and nn.Module")


# ── test 2: forward output shapes and types ───────────────────────────────────

def test_forward_shapes():
    builder = NullConsensusBuilder(k=K)
    obs = _make_obs()
    loss_cb, consensus = builder(obs)

    assert loss_cb.ndim == 0, f"loss_cb must be scalar, got shape {loss_cb.shape}"
    assert float(loss_cb.item()) == 0.0, f"loss_cb must be 0.0, got {loss_cb.item()}"
    assert consensus.shape == (B, N), f"consensus shape wrong: {consensus.shape}"
    assert consensus.dtype == torch.int64, f"consensus dtype wrong: {consensus.dtype}"
    assert consensus.eq(0).all(), "All consensus labels must be zero"
    print("  [OK] forward() shapes and values correct")


# ── test 3: forward rejects wrong input shape ─────────────────────────────────

def test_forward_rejects_wrong_ndim():
    builder = NullConsensusBuilder(k=K)
    try:
        builder(torch.randn(B, OBS))   # 2-D, should raise
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [OK] forward() raises ValueError on wrong ndim")


# ── test 4: infer output shape and type ──────────────────────────────────────

def test_infer_shapes():
    builder = NullConsensusBuilder(k=K)
    obs = torch.randn(N, OBS)
    consensus = builder.infer(obs)

    assert consensus.shape == (N,), f"infer shape wrong: {consensus.shape}"
    assert consensus.dtype == torch.int64
    assert consensus.eq(0).all(), "All inferred labels must be zero"
    print("  [OK] infer() shapes and values correct")


# ── test 5: infer rejects wrong shape ────────────────────────────────────────

def test_infer_rejects_wrong_ndim():
    builder = NullConsensusBuilder(k=K)
    try:
        builder.infer(torch.randn(B, N, OBS))   # 3-D, should raise
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [OK] infer() raises ValueError on wrong ndim")


# ── test 6: ema_update_teacher is a no-op ────────────────────────────────────

def test_ema_update_noop():
    builder = NullConsensusBuilder(k=K)
    before = builder.student.weight.clone()
    builder.ema_update_teacher()
    assert torch.allclose(before, builder.student.weight), "ema_update_teacher should not change weights"
    print("  [OK] ema_update_teacher() is a no-op")


# ── test 7: backward() on loss_cb succeeds with zero gradient ─────────────────

def test_backward_succeeds():
    builder = NullConsensusBuilder(k=K)
    opt = torch.optim.Adam(builder.student.parameters(), lr=1e-3)
    obs = _make_obs()

    loss_cb, _ = builder(obs)
    opt.zero_grad()
    loss_cb.backward()

    # Gradient must be exactly zero (not None)
    grad = builder.student.weight.grad
    assert grad is not None, "Gradient should not be None"
    assert grad.eq(0.0).all(), f"Gradient must be zero, got: {grad}"

    opt.step()   # Must not raise
    print("  [OK] backward() and opt.step() succeed without error")


# ── test 8: Adam optimizer on student.parameters() never errors ───────────────

def test_optimizer_creation():
    builder = NullConsensusBuilder(k=K)
    # This is the exact pattern used in train_cola.py
    opt = torch.optim.Adam(builder.student.parameters(), lr=3e-4)
    assert opt is not None
    print("  [OK] Adam(null_builder.student.parameters()) created without error")


# ── test 9: works end-to-end with ConsensusEmbedding ─────────────────────────

def test_end_to_end_with_embedding():
    builder = NullConsensusBuilder(k=K)
    embedding = ConsensusEmbedding(k=K, emb_dim=16)

    obs = _make_obs()
    loss_cb, consensus = builder(obs)
    emb = embedding(consensus)

    assert emb.shape == (B, N, 16), f"Embedding shape wrong: {emb.shape}"
    # All agents in all batches receive the same embedding (class-0 row of the proj matrix).
    assert emb.eq(emb[:, :1, :]).all(), "All-zero consensus should produce identical embeddings"
    print("  [OK] end-to-end with ConsensusEmbedding produces consistent embeddings")


# ── test 10: exported from package __init__ ────────────────────────────────────

def test_package_export():
    assert NullFromInit is NullConsensusBuilder, "__init__.py must re-export NullConsensusBuilder"
    print("  [OK] NullConsensusBuilder exported from cola_framework.consensus")


# ── test 11: short integrated training loop with --no_cola ────────────────────

def test_integrated_no_cola_loop():
    """Run a minimal MADDPG loop with NullConsensusBuilder end-to-end."""
    import copy
    from cola_framework.buffers.replay_buffer import ReplayBuffer
    from cola_framework.critics.centralized_critic import Critic
    from cola_framework.envs.mpe_wrapper import MPEWrapper
    from cola_framework.evaluation.policy_evaluator import PolicyEvaluator
    from cola_framework.loops.cola_training_loop import COLATrainingConfig, COLATrainingLoop
    from cola_framework.policies.actor import Actor
    from cola_framework.trainers.maddpg_updater import MADDPGUpdater

    torch.manual_seed(0)
    device = "cpu"
    k, emb_dim, hidden_dim = 4, 8, 32

    env = MPEWrapper(scenario="simple_spread", n_agents=3, max_cycles=25, device=device)

    replay_buffer = ReplayBuffer(
        capacity=2000, n_agents=env.n_agents, obs_dim=env.obs_dim,
        action_dim=env.action_dim, state_dim=env.state_dim, device=device,
    )

    consensus_builder = NullConsensusBuilder(k=k).to(device)
    embedding_layer = ConsensusEmbedding(k=k, emb_dim=emb_dim).to(device)

    actors = [Actor(env.obs_dim, emb_dim, env.action_dim, hidden_dim).to(device) for _ in range(env.n_agents)]
    critics = [Critic(env.state_dim, env.n_agents, emb_dim, env.action_dim, hidden_dim).to(device) for _ in range(env.n_agents)]
    target_actors = [copy.deepcopy(a) for a in actors]
    target_critics = [copy.deepcopy(c) for c in critics]

    opt_cb = torch.optim.Adam(consensus_builder.student.parameters(), lr=3e-4)
    opt_actors = [torch.optim.Adam(a.parameters(), lr=1e-2) for a in actors]
    opt_critics = [torch.optim.Adam(c.parameters(), lr=1e-2) for c in critics]

    updater = MADDPGUpdater(
        consensus_builder=consensus_builder, embedding_layer=embedding_layer,
        actors=actors, critics=critics,
        target_actors=target_actors, target_critics=target_critics,
        opt_cb=opt_cb, opt_actors=opt_actors, opt_critics=opt_critics,
    )

    cfg = COLATrainingConfig(max_steps=500, warmup_steps=100, train_freq=50, batch_size=64, log_interval=250)
    metrics_log = []

    def _on_log(record):
        metrics_log.append(record)

    loop = COLATrainingLoop(env, replay_buffer, consensus_builder, embedding_layer, actors, updater, cfg, on_log=_on_log)
    result = loop.run()

    assert result["total_steps"] == 500

    # Confirm loss_cb is 0.0 in every logged update
    for record in metrics_log:
        if "loss_cb" in record:
            assert abs(record["loss_cb"]) < 1e-7, f"loss_cb should be 0.0, got {record['loss_cb']}"

    # Confirm evaluator works
    evaluator = PolicyEvaluator(env, consensus_builder, embedding_layer, actors)
    eval_metrics = evaluator.evaluate(n_episodes=3)
    assert "mean_episode_return" in eval_metrics
    assert "consensus_agreement_eval" in eval_metrics
    # All consensus labels are 0, so all agents always agree → 100 % agreement
    assert abs(eval_metrics["consensus_agreement_eval"] - 1.0) < 1e-6, \
        f"Null builder should give 100% agreement, got {eval_metrics['consensus_agreement_eval']}"

    print("  [OK] integrated no-cola loop runs, loss_cb=0.0, consensus_agreement=1.0")


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print("Phase 0 gate: NullConsensusBuilder")
    test_implements_interface()
    test_forward_shapes()
    test_forward_rejects_wrong_ndim()
    test_infer_shapes()
    test_infer_rejects_wrong_ndim()
    test_ema_update_noop()
    test_backward_succeeds()
    test_optimizer_creation()
    test_end_to_end_with_embedding()
    test_package_export()
    test_integrated_no_cola_loop()
    print("\nPhase 0 gate PASSED: NullConsensusBuilder OK")


if __name__ == "__main__":
    main()
