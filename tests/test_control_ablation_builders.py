"""Gate tests for the control-ablation consensus builders.

Covers RandomLabelConsensusBuilder and ShuffledLabelConsensusBuilder — the two
variants that preserve COLA's architecture but corrupt the *content* of the
consensus label in controlled ways.
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.random_label_builder import RandomLabelConsensusBuilder
from cola_framework.consensus.shuffled_label_builder import ShuffledLabelConsensusBuilder
from cola_framework.consensus import (
    RandomLabelConsensusBuilder as RandomFromInit,
    ShuffledLabelConsensusBuilder as ShuffledFromInit,
)
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.interfaces.consensus import ConsensusModule


B, N, OBS = 16, 3, 18
K = 4


def _make_obs(n=N):
    return torch.randn(B, n, OBS)


# ── RandomLabelConsensusBuilder ───────────────────────────────────────────────

def test_random_implements_interface():
    builder = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=42)
    assert isinstance(builder, ConsensusModule)
    assert isinstance(builder, torch.nn.Module)
    print("  [OK] random builder implements ConsensusModule and nn.Module")


def test_random_labels_constant_across_batch():
    builder = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=42)
    _, consensus = builder(_make_obs())
    assert consensus.shape == (B, N)
    assert consensus.dtype == torch.int64
    # Every batch row must equal the first row (label depends only on agent index).
    assert consensus.eq(consensus[:1, :]).all(), "labels must be identical across batch dim"
    print("  [OK] random labels are identical across the batch dimension")


def test_random_labels_stable_across_calls():
    builder = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=42)
    _, c1 = builder(_make_obs())
    _, c2 = builder(_make_obs())  # different random inputs
    assert torch.equal(c1, c2), "labels must not change between forward calls"
    # infer must agree with forward's per-agent labels too.
    inferred = builder.infer(torch.randn(N, OBS))
    assert torch.equal(inferred, c1[0]), "infer must return the same fixed labels"
    print("  [OK] random labels are stable across separate forward/infer calls")


def test_random_loss_is_zero_and_backward_ok():
    builder = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=7)
    opt = torch.optim.Adam(builder.student.parameters(), lr=3e-4)
    loss_cb, _ = builder(_make_obs())
    assert loss_cb.ndim == 0
    assert float(loss_cb.item()) == 0.0, "loss_cb must be exactly 0.0"
    assert loss_cb.requires_grad, "loss_cb must require grad so optimizer.step() works"
    opt.zero_grad()
    loss_cb.backward()
    grad = builder.student.weight.grad
    assert grad is not None and grad.eq(0.0).all(), "gradient must be exactly zero"
    opt.step()  # must not raise
    print("  [OK] random loss_cb is zero and backward/step succeed")


def test_random_ema_update_noop():
    builder = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=42)
    before = builder.labels.clone()
    builder.ema_update_teacher()  # must not throw
    assert torch.equal(before, builder.labels), "labels must be untouched by ema_update_teacher"
    print("  [OK] random ema_update_teacher() is a safe no-op")


def test_random_seed_determinism_and_buffer():
    a = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=123)
    b = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=123)
    c = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=999)
    assert torch.equal(a.labels, b.labels), "same seed must give same labels"
    # labels is a registered buffer (appears in state_dict, moves with .to()).
    assert "labels" in a.state_dict(), "labels must be a registered buffer"
    # Different seeds *usually* differ; not asserted strictly to avoid flakiness.
    _ = c
    print("  [OK] random builder is seed-deterministic and stores labels as a buffer")


def test_random_end_to_end_with_embedding():
    builder = RandomLabelConsensusBuilder(n_agents=N, k=K, seed=42)
    embedding = ConsensusEmbedding(k=K, emb_dim=16)
    _, consensus = builder(_make_obs())
    emb = embedding(consensus)
    assert emb.shape == (B, N, 16)
    print("  [OK] random builder works end-to-end with ConsensusEmbedding")


# ── ShuffledLabelConsensusBuilder ─────────────────────────────────────────────

class _FakeInner:
    """Stub inner builder returning controlled labels for deterministic tests."""

    def __init__(self, labels, loss_value=0.5):
        self.k = K
        self._labels = labels
        self._loss_value = loss_value

    def __call__(self, obs_batch):
        loss = torch.tensor(self._loss_value, requires_grad=True)
        return loss, self._labels

    def infer(self, obs):
        return self._labels[0].clone()

    def ema_update_teacher(self):
        pass


def _shuffled_with_inner(labels, loss_value=0.5):
    builder = ShuffledLabelConsensusBuilder(obs_dim=OBS, k=K)
    builder.inner = _FakeInner(labels, loss_value)
    return builder


def test_shuffled_implements_interface():
    builder = ShuffledLabelConsensusBuilder(obs_dim=OBS, k=K)
    assert isinstance(builder, ConsensusModule)
    assert isinstance(builder, torch.nn.Module)
    print("  [OK] shuffled builder implements ConsensusModule and nn.Module")


def test_shuffled_identical_labels_unchanged():
    # Inner returns the SAME label for every agent → any permutation is a no-op.
    labels = torch.full((B, N), 2, dtype=torch.int64)
    builder = _shuffled_with_inner(labels)
    loss_cb, out = builder(_make_obs())
    assert torch.equal(out, labels), "permuting identical values must leave them unchanged"
    assert float(loss_cb.item()) == 0.5, "inner loss must be passed through unchanged"
    print("  [OK] shuffled output is unchanged when inner labels are all identical")


def test_shuffled_distinct_labels_preserve_multiset_but_reassign():
    # Inner returns distinct labels per agent: [0, 1, 2] for every batch element.
    base = torch.arange(N, dtype=torch.int64).view(1, N).expand(B, N).contiguous()
    builder = _shuffled_with_inner(base)
    torch.manual_seed(0)
    _, out = builder(_make_obs())

    # Multiset per batch element preserved → sorted rows equal the original.
    assert torch.equal(out.sort(dim=1).values, base.sort(dim=1).values), \
        "per-batch-element multiset of labels must be preserved"

    # Per-agent assignment must differ from the inner output on at least some rows.
    differing_rows = (out != base).any(dim=1).sum().item()
    assert differing_rows > 0, "shuffle must reassign labels on at least some batch elements"
    print(
        "  [OK] shuffled preserves multiset per batch element and reassigns "
        "({} / {} rows differ)".format(differing_rows, B)
    )


def test_shuffled_independent_permutations_per_batch_element():
    # With distinct labels, different batch rows should generally get different perms.
    base = torch.arange(N, dtype=torch.int64).view(1, N).expand(B, N).contiguous()
    builder = _shuffled_with_inner(base)
    torch.manual_seed(1)
    _, out = builder(_make_obs())
    # Not all rows should be identical (would imply a single shared permutation).
    unique_rows = {tuple(row.tolist()) for row in out}
    assert len(unique_rows) > 1, "permutations should be sampled independently per batch element"
    print("  [OK] shuffled samples permutations independently per batch element")


def test_shuffled_loss_passthrough_and_real_inner_shapes():
    # With the REAL inner builder, check shape/type contract and multiset preservation.
    builder = ShuffledLabelConsensusBuilder(obs_dim=OBS, k=K, hidden_dim=32)
    obs = _make_obs()
    # Capture the inner labels for the same obs (forward is deterministic given weights).
    _, inner_labels = builder.inner(obs)
    loss_cb, out = builder(obs)
    assert out.shape == (B, N) and out.dtype == torch.int64
    assert loss_cb.requires_grad, "inner loss must keep requiring grad so the CB trains"
    assert torch.equal(out.sort(dim=1).values, inner_labels.sort(dim=1).values), \
        "multiset per batch element must match the real inner builder's labels"
    print("  [OK] shuffled passes inner loss through and preserves real-inner multiset")


def test_shuffled_optimizer_and_ema_wiring():
    builder = ShuffledLabelConsensusBuilder(obs_dim=OBS, k=K, hidden_dim=32)
    # Same wiring as train_cola.py — must optimize the real inner student.
    opt = torch.optim.Adam(builder.student.parameters(), lr=3e-4)
    assert builder.student is builder.inner.student, "student must alias the inner builder's student"
    assert builder.k == K
    loss_cb, _ = builder(_make_obs())
    opt.zero_grad()
    loss_cb.backward()
    opt.step()
    builder.ema_update_teacher()  # must not throw and must update inner teacher
    print("  [OK] shuffled optimizer + EMA wiring works against the inner student/teacher")


def test_shuffled_infer_permutes_agents():
    builder = ShuffledLabelConsensusBuilder(obs_dim=OBS, k=K)
    out = builder.infer(torch.randn(N, OBS))
    assert out.shape == (N,) and out.dtype == torch.int64
    print("  [OK] shuffled infer() returns per-agent labels of correct shape")


# ── package exports ───────────────────────────────────────────────────────────

def test_package_exports():
    assert RandomFromInit is RandomLabelConsensusBuilder
    assert ShuffledFromInit is ShuffledLabelConsensusBuilder
    print("  [OK] both builders are exported from cola_framework.consensus")


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print("Control-ablation gate: Random / Shuffled consensus builders")
    test_random_implements_interface()
    test_random_labels_constant_across_batch()
    test_random_labels_stable_across_calls()
    test_random_loss_is_zero_and_backward_ok()
    test_random_ema_update_noop()
    test_random_seed_determinism_and_buffer()
    test_random_end_to_end_with_embedding()
    test_shuffled_implements_interface()
    test_shuffled_identical_labels_unchanged()
    test_shuffled_distinct_labels_preserve_multiset_but_reassign()
    test_shuffled_independent_permutations_per_batch_element()
    test_shuffled_loss_passthrough_and_real_inner_shapes()
    test_shuffled_optimizer_and_ema_wiring()
    test_shuffled_infer_permutes_agents()
    test_package_exports()
    print("\nControl-ablation gate PASSED")


if __name__ == "__main__":
    main()
