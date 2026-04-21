"""Step 1 module gate test for the environment wrapper."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.envs.mpe_wrapper import MPEWrapper


def main() -> None:
    env = MPEWrapper(scenario="simple_spread", n_agents=3)
    obs, state = env.reset()

    assert obs.shape == (3, env.obs_dim)
    assert state.shape == (3 * env.obs_dim,)

    rand_actions = torch.zeros(3, env.action_dim)
    obs2, state2, rews, dones = env.step(rand_actions)
    assert obs2.shape == (3, env.obs_dim)
    assert state2.shape == (3 * env.obs_dim,)
    assert rews.shape == (3,)
    assert dones.shape == (3,)

    print("Step 1 gate passed: Environment wrapper OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Step 1 gate failed:", exc)
        raise
