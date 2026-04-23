"""Step 12 gate test for HistoryAwareConsensusBuilder concrete class."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.history_aware_builder import HistoryAwareConsensusBuilder
from cola_framework.encoders.gru_encoder import GRUHistoryEncoder
from cola_framework.interfaces.consensus import ConsensusModule


def main() -> None:
    torch.manual_seed(29)

    batch_size = 48
    n_agents = 3
    t_len = 10
    obs_dim = 18

    encoder = GRUHistoryEncoder(obs_dim=obs_dim, hidden_dim=64)
    cb = HistoryAwareConsensusBuilder(encoder=encoder, k=4, mlp_hidden=64)

    # Ensure we keep the same external consensus interface contract.
    assert isinstance(cb, ConsensusModule)

    # Build aligned sequence views to make agreement learning feasible.
    latent_seq = torch.randn(batch_size, t_len, obs_dim)
    obs_seq_batch = torch.stack(
        [latent_seq + 0.02 * torch.randn_like(latent_seq) for _ in range(n_agents)],
        dim=1,
    )

    initial_teacher = [p.detach().clone() for p in cb.teacher_encoder.parameters()]

    opt = torch.optim.Adam(
        list(cb.student_encoder.parameters()) + list(cb.student_head.parameters()),
        lr=3e-4,
    )

    for _ in range(250):
        loss, consensus = cb(obs_seq_batch)
        opt.zero_grad()
        loss.backward()

        # Teacher branch must not receive gradients.
        for p in cb.teacher_encoder.parameters():
            assert p.grad is None
        for p in cb.teacher_head.parameters():
            assert p.grad is None

        opt.step()
        cb.ema_update_teacher()

    assert loss.ndim == 0
    assert consensus.shape == (batch_size, n_agents)
    assert consensus.dtype == torch.int64

    # Teacher should move after EMA updates.
    teacher_delta = 0.0
    for old, new in zip(initial_teacher, cb.teacher_encoder.parameters()):
        teacher_delta += (old - new.detach()).abs().sum().item()
    assert teacher_delta > 0.0

    with torch.no_grad():
        inferred = cb.infer(obs_seq_batch[0])
        assert inferred.shape == (n_agents,)

        _, eval_consensus = cb(obs_seq_batch)
        fully_agree = (eval_consensus == eval_consensus[:, :1]).all(dim=1)
        agreement_rate = fully_agree.float().mean().item()

    print("History-aware agreement rate:", round(agreement_rate, 4))
    assert agreement_rate >= 0.75
    print("Step 12 gate passed: HistoryAwareConsensusBuilder OK")


if __name__ == "__main__":
    main()
