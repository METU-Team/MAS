"""Step 13 gate test for ObservationWindowManager + SequenceReplayBuffer."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.sequence_buffer import SequenceReplayBuffer
from cola_framework.interfaces.replay_buffer import ExperienceReplay
from cola_framework.utils.window_manager import ObservationWindowManager


def main() -> None:
    torch.manual_seed(41)

    n_agents = 3
    obs_dim = 18
    action_dim = 5
    state_dim = n_agents * obs_dim
    window = 5

    # Window manager checks.
    wm = ObservationWindowManager(n_agents=n_agents, obs_dim=obs_dim, window=window)
    w0 = wm.get()
    assert w0.shape == (n_agents, window, obs_dim)
    assert float(w0.abs().sum().item()) == 0.0

    for step in range(3):
        obs = torch.full((n_agents, obs_dim), float(step + 1))
        wm.push(obs)

    w1 = wm.get()
    assert w1.shape == (n_agents, window, obs_dim)
    assert w1[0, 0, 0].item() == 0.0
    assert w1[0, 2, 0].item() == 1.0
    assert w1[0, 4, 0].item() == 3.0

    wm.reset()
    assert float(wm.get().abs().sum().item()) == 0.0

    # Sequence replay checks.
    buf = SequenceReplayBuffer(
        capacity=1000,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        window=window,
    )

    # Verify we keep the same replay-buffer interface contract.
    assert isinstance(buf, ExperienceReplay)

    # Legacy push signature still works (sequence fields auto-generated).
    for _ in range(10):
        obs = torch.zeros(n_agents, obs_dim)
        state = torch.zeros(state_dim)
        actions = torch.zeros(n_agents, action_dim)
        rewards = torch.zeros(n_agents)
        next_obs = torch.zeros(n_agents, obs_dim)
        next_state = torch.zeros(state_dim)
        dones = torch.zeros(n_agents)
        buf.push(obs, state, actions, rewards, next_obs, next_state, dones)

    # Extended push signature with explicit history windows.
    for _ in range(40):
        obs = torch.randn(n_agents, obs_dim)
        state = torch.randn(state_dim)
        actions = torch.randn(n_agents, action_dim)
        rewards = torch.randn(n_agents)
        next_obs = torch.randn(n_agents, obs_dim)
        next_state = torch.randn(state_dim)
        dones = torch.randint(0, 2, (n_agents,)).float()

        obs_seq = torch.randn(n_agents, window, obs_dim)
        next_obs_seq = torch.randn(n_agents, window, obs_dim)

        buf.push(
            obs,
            state,
            actions,
            rewards,
            next_obs,
            next_state,
            dones,
            obs_seq=obs_seq,
            next_obs_seq=next_obs_seq,
        )

    assert len(buf) == 50

    batch = buf.sample(32)
    assert batch["obs_seq"].shape == (32, n_agents, window, obs_dim)
    assert batch["next_obs_seq"].shape == (32, n_agents, window, obs_dim)
    assert batch["obs"].shape == (32, n_agents, obs_dim)
    assert batch["state"].shape == (32, state_dim)
    assert batch["actions"].shape == (32, n_agents, action_dim)
    assert batch["rewards"].shape == (32, n_agents)
    assert batch["next_obs"].shape == (32, n_agents, obs_dim)
    assert batch["next_state"].shape == (32, state_dim)
    assert batch["dones"].shape == (32, n_agents)

    print("Step 13 gate passed: WindowManager + SequenceReplayBuffer OK")


if __name__ == "__main__":
    main()
