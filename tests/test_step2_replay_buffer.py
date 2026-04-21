"""Step 2 module gate test for replay buffer."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.buffers.replay_buffer import ReplayBuffer


def main() -> None:
    buf = ReplayBuffer(
        capacity=1000,
        n_agents=3,
        obs_dim=18,
        action_dim=5,
        state_dim=54,
    )

    def dummy(*shape):
        return np.zeros(shape, dtype=np.float32)

    for _ in range(50):
        buf.push(
            dummy(3, 18),
            dummy(54),
            dummy(3, 5),
            dummy(3),
            dummy(3, 18),
            dummy(54),
            dummy(3),
        )

    assert len(buf) == 50

    batch = buf.sample(32)
    assert batch["obs"].shape == (32, 3, 18)
    assert batch["state"].shape == (32, 54)
    assert batch["actions"].shape == (32, 3, 5)
    assert batch["rewards"].shape == (32, 3)
    assert batch["next_obs"].shape == (32, 3, 18)
    assert batch["next_state"].shape == (32, 54)
    assert batch["dones"].shape == (32, 3)

    print("Step 2 gate passed: Replay buffer OK")


if __name__ == "__main__":
    main()
