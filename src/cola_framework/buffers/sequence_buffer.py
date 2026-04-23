"""Replay buffer variant that stores observation sequences."""

from typing import Dict, Optional

import numpy as np
import torch

from cola_framework.interfaces.replay_buffer import ExperienceReplay


class SequenceReplayBuffer(ExperienceReplay):
    """Stores standard transitions plus sequence fields.

    Interface compatibility:
    - Keeps `push`, `sample`, `__len__` methods from ExperienceReplay.
    - Base push arguments match ReplayBuffer.
    - Optional sequence fields are accepted as extra keyword arguments.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        window: int,
        device: str = "cpu",
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive.")

        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.window = window
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

        self.obs_seq = np.zeros((capacity, n_agents, window, obs_dim), dtype=np.float32)
        self.next_obs_seq = np.zeros((capacity, n_agents, window, obs_dim), dtype=np.float32)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def _coerce_seq(self, seq, fallback_obs):
        if seq is None:
            obs = self._to_numpy(fallback_obs)
            if obs.shape != (self.n_agents, self.obs_dim):
                raise ValueError("fallback obs must have shape [n_agents, obs_dim].")
            return np.repeat(obs[:, None, :], self.window, axis=1)

        seq_np = self._to_numpy(seq)
        expected_shape = (self.n_agents, self.window, self.obs_dim)
        if seq_np.shape != expected_shape:
            raise ValueError("sequence must have shape {}, got {}".format(expected_shape, seq_np.shape))
        return seq_np

    def push(
        self,
        obs,
        state,
        actions,
        rewards,
        next_obs,
        next_state,
        dones,
        obs_seq: Optional[torch.Tensor] = None,
        next_obs_seq: Optional[torch.Tensor] = None,
    ) -> None:
        obs_np = self._to_numpy(obs)
        state_np = self._to_numpy(state)
        actions_np = self._to_numpy(actions)
        rewards_np = self._to_numpy(rewards)
        next_obs_np = self._to_numpy(next_obs)
        next_state_np = self._to_numpy(next_state)
        dones_np = self._to_numpy(dones)

        self.obs[self.ptr] = obs_np
        self.state[self.ptr] = state_np
        self.actions[self.ptr] = actions_np
        self.rewards[self.ptr] = rewards_np
        self.next_obs[self.ptr] = next_obs_np
        self.next_state[self.ptr] = next_state_np
        self.dones[self.ptr] = dones_np

        self.obs_seq[self.ptr] = self._coerce_seq(obs_seq, obs_np)
        self.next_obs_seq[self.ptr] = self._coerce_seq(next_obs_seq, next_obs_np)

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
            "obs_seq": _to_tensor(self.obs_seq),
            "next_obs_seq": _to_tensor(self.next_obs_seq),
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
