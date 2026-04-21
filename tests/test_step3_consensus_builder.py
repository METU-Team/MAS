"""Step 3 module gate test for consensus builder."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.builder import ConsensusBuilder


def main() -> None:
    torch.manual_seed(7)

    batch_size = 64
    n_agents = 3
    obs_dim = 18

    cb = ConsensusBuilder(obs_dim=obs_dim, k=4, hidden_dim=64)
    opt = torch.optim.Adam(cb.student.parameters(), lr=3e-4)

    # Build aligned multi-view data: each agent observes a noisy view of
    # the same latent state, which should encourage consensus agreement.
    latent = torch.randn(batch_size, obs_dim)
    obs = torch.stack(
        [latent + 0.02 * torch.randn_like(latent) for _ in range(n_agents)],
        dim=1,
    )

    initial_teacher = [p.detach().clone() for p in cb.teacher.parameters()]

    for _ in range(250):
        loss, consensus = cb(obs)
        opt.zero_grad()
        loss.backward()

        # Teacher parameters must not receive gradients.
        for p in cb.teacher.parameters():
            assert p.grad is None

        opt.step()
        cb.ema_update_teacher()

    # Check output contract.
    assert loss.ndim == 0
    assert consensus.shape == (batch_size, n_agents)
    assert consensus.dtype == torch.int64

    # Teacher should move via EMA after student updates.
    teacher_delta = 0.0
    for old, new in zip(initial_teacher, cb.teacher.parameters()):
        teacher_delta += (old - new.detach()).abs().sum().item()
    assert teacher_delta > 0.0

    # Evaluate per-sample full agreement across agents.
    with torch.no_grad():
        inferred = cb.infer(obs[0])
        assert inferred.shape == (n_agents,)

        _, consensus_eval = cb(obs)
        fully_agree = (consensus_eval == consensus_eval[:, :1]).all(dim=1)
        agreement_rate = fully_agree.float().mean().item()

    print("Agreement rate:", round(agreement_rate, 4))
    assert agreement_rate >= 0.75
    print("Step 3 gate passed: Consensus builder OK")


if __name__ == "__main__":
    main()
