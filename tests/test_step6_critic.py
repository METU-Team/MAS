"""Step 6 module gate test for centralized critic network."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.critics.centralized_critic import Critic


def main() -> None:
    torch.manual_seed(31)

    critic = Critic(state_dim=54, n_agents=3, emb_dim=16, action_dim=5, hidden_dim=64)

    # Batch contract.
    state = torch.randn(32, 54)
    all_cemb = torch.randn(32, 3 * 16)
    all_actions = torch.randn(32, 3 * 5)
    q = critic(state, all_cemb, all_actions)
    assert q.shape == (32, 1)
    assert q.dtype == torch.float32

    # Single-sample contract.
    state_single = torch.randn(54)
    all_cemb_single = torch.randn(3 * 16)
    all_actions_single = torch.randn(3 * 5)
    q_single = critic(state_single, all_cemb_single, all_actions_single)
    assert q_single.shape == (1,)

    # Gradient flow check.
    loss = q.pow(2).mean()
    critic.zero_grad()
    loss.backward()
    grad_norm = 0.0
    for p in critic.parameters():
        if p.grad is not None:
            grad_norm += p.grad.abs().sum().item()
    assert grad_norm > 0.0

    print("Step 6 gate passed: Critic OK")


if __name__ == "__main__":
    main()
