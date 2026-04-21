"""Step 5 module gate test for actor network."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.policies.actor import Actor


def main() -> None:
    torch.manual_seed(23)

    actor = Actor(obs_dim=18, emb_dim=16, action_dim=5, hidden_dim=64)

    # Batch contract.
    obs = torch.randn(32, 18)
    cemb = torch.randn(32, 16)
    act = actor(obs, cemb)
    assert act.shape == (32, 5)
    assert act.abs().max().item() <= 1.0 + 1e-6

    # Single-sample contract.
    obs_single = torch.randn(18)
    cemb_single = torch.randn(16)
    act_single = actor(obs_single, cemb_single)
    assert act_single.shape == (5,)
    assert act_single.abs().max().item() <= 1.0 + 1e-6

    # Gradient flow: actor parameters should receive gradients.
    loss = act.pow(2).mean()
    actor.zero_grad()
    loss.backward()
    grad_norm = 0.0
    for p in actor.parameters():
        if p.grad is not None:
            grad_norm += p.grad.abs().sum().item()
    assert grad_norm > 0.0

    print("Step 5 gate passed: Actor OK")


if __name__ == "__main__":
    main()
