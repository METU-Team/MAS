"""Sliding window manager for per-agent observation histories."""

from collections import deque
from typing import Deque, List, Optional

import numpy as np
import torch


class ObservationWindowManager(object):
    """Maintains fixed-length observation histories for each agent.

    Returned window format: [n_agents, window, obs_dim].
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        window: int,
        device: Optional[str] = None,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if window <= 0:
            raise ValueError("window must be positive.")

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.window = window
        self.device = device
        self._buffers = None
        self.reset()

    def reset(self) -> None:
        zero = np.zeros(self.obs_dim, dtype=np.float32)
        buffers = []
        for _ in range(self.n_agents):
            buffers.append(deque([zero.copy() for _ in range(self.window)], maxlen=self.window))
        self._buffers = buffers
        # First push after reset warm-starts the window by repeating the first
        # observation instead of leaving zero-padding, so the temporal encoder
        # never sees spurious all-zero frames at episode starts.
        self._initialized = False

    def push(self, obs) -> None:
        """Append one timestep observation.

        obs shape: [n_agents, obs_dim]
        """
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        if obs_np.shape != (self.n_agents, self.obs_dim):
            raise ValueError(
                "obs must have shape ({}, {}), got {}".format(
                    self.n_agents, self.obs_dim, obs_np.shape
                )
            )

        for agent_idx in range(self.n_agents):
            ob = obs_np[agent_idx].copy()
            if not self._initialized:
                self._buffers[agent_idx].clear()
                for _ in range(self.window):
                    self._buffers[agent_idx].append(ob.copy())
            else:
                self._buffers[agent_idx].append(ob)
        self._initialized = True

    def get(self) -> torch.Tensor:
        stacked = np.stack(
            [np.asarray(self._buffers[a], dtype=np.float32) for a in range(self.n_agents)],
            axis=0,
        )
        out = torch.tensor(stacked, dtype=torch.float32)
        if self.device is not None:
            out = out.to(self.device)
        return out
