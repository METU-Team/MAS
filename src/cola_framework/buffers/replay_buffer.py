"""Numpy-backed circular replay buffer for MADDPG-style training."""

from typing import Dict

import numpy as np
import torch

from cola_framework.interfaces.replay_buffer import ExperienceReplay


class ReplayBuffer(ExperienceReplay):
    """Stores multi-agent transitions and returns random minibatches."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)

    def push(
        self,
        obs,
        state,
        actions,
        rewards,
        next_obs,
        next_state,
        dones,
    ) -> None:
        def _to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        self.obs[self.ptr] = _to_numpy(obs)
        self.state[self.ptr] = _to_numpy(state)
        self.actions[self.ptr] = _to_numpy(actions)
        self.rewards[self.ptr] = _to_numpy(rewards)
        self.next_obs[self.ptr] = _to_numpy(next_obs)
        self.next_state[self.ptr] = _to_numpy(next_state)
        self.dones[self.ptr] = _to_numpy(dones)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        idx = np.random.randint(0, self.size, size=batch_size)

        def _to_tensor(array):
            return torch.tensor(array[idx], dtype=torch.float32, device=self.device)

        return {
            "obs": _to_tensor(self.obs),
            "state": _to_tensor(self.state),
            "actions": _to_tensor(self.actions),
            "rewards": _to_tensor(self.rewards),
            "next_obs": _to_tensor(self.next_obs),
            "next_state": _to_tensor(self.next_state),
            "dones": _to_tensor(self.dones),
        }

    def __len__(self) -> int:
        return self.size
