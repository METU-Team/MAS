"""PettingZoo MPE wrapper that matches the COLA environment interface."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from cola_framework.interfaces.environment import MultiAgentEnvironment


class MPEWrapper(MultiAgentEnvironment):
    """Wraps PettingZoo parallel MPE envs into tensor-first outputs.

    Supported scenarios:
    - simple_spread
    - simple_tag

    Note:
    `simple_tag` has heterogeneous observation sizes across roles.
    We pad each agent observation to `self.obs_dim = max(obs_dim_per_agent)` so
    downstream modules can operate on fixed-size tensors.
    """

    def __init__(
        self,
        scenario: str = "simple_spread",
        n_agents: int = 3,
        max_cycles: int = 100,
        device: str = "cpu",
        render_mode: Optional[str] = None,
    ) -> None:
        self.scenario = scenario
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.device = device
        self.render_mode = render_mode

        self.env = self._build_env()
        self.agents = list(self.env.possible_agents)

        obs_dict, _ = self.env.reset()
        self._obs_dim_by_agent = {
            agent: int(obs_dict[agent].shape[0])
            for agent in self.agents
        }
        self.obs_dim = max(self._obs_dim_by_agent.values())
        self.state_dim = self.obs_dim * self.n_agents
        self._action_dim_by_agent = {
            agent: int(self.env.action_space(agent).shape[0])
            for agent in self.agents
        }
        self.action_dim = max(self._action_dim_by_agent.values())

        # Continuous MPE actions are typically Box([0,1]); actor outputs are
        # in [-1,1], so we map them into the env range before stepping.
        self._action_low = {
            agent: self.env.action_space(agent).low.astype(np.float32)
            for agent in self.agents
        }
        self._action_high = {
            agent: self.env.action_space(agent).high.astype(np.float32)
            for agent in self.agents
        }

    def _build_env(self):
        if self.scenario == "simple_spread":
            from pettingzoo.mpe import simple_spread_v3

            return simple_spread_v3.parallel_env(
                N=self.n_agents,
                max_cycles=self.max_cycles,
                continuous_actions=True,
                render_mode=self.render_mode,
            )

        if self.scenario == "simple_tag":
            from pettingzoo.mpe import simple_tag_v3

            n_adversaries = max(1, self.n_agents - 1)
            n_good = 1
            return simple_tag_v3.parallel_env(
                num_good=n_good,
                num_adversaries=n_adversaries,
                num_obstacles=2,
                max_cycles=self.max_cycles,
                continuous_actions=True,
                render_mode=self.render_mode,
            )

        raise ValueError(
            "Unsupported scenario '{}'. Use 'simple_spread' or 'simple_tag'.".format(
                self.scenario
            )
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.env.reset(seed=seed)
        obs = self._dict_to_tensor(obs_dict)
        state = obs.reshape(-1)
        return obs.to(self.device), state.to(self.device)

    def step(self, action_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one environment step.

        action_tensor: [n_agents, action_dim]
        returns:
            next_obs: [n_agents, obs_dim]
            next_state: [state_dim]
            rewards: [n_agents]
            dones: [n_agents] (bool)
        """
        action_tensor = action_tensor.detach().to("cpu")
        action_dict = {}
        for i, agent in enumerate(self.agents):
            raw_action = np.asarray(action_tensor[i].numpy(), dtype=np.float32).reshape(-1)
            action_dim = self._action_dim_by_agent[agent]

            if raw_action.shape[0] < action_dim:
                pad = np.zeros(action_dim - raw_action.shape[0], dtype=np.float32)
                raw_action = np.concatenate([raw_action, pad], axis=0)
            elif raw_action.shape[0] > action_dim:
                raw_action = raw_action[:action_dim]

            low = self._action_low[agent]
            high = self._action_high[agent]

            # Affine map from [-1,1] to [low, high], then clip for safety.
            scaled = 0.5 * (raw_action + 1.0) * (high - low) + low
            action_dict[agent] = np.clip(scaled, low, high)

        obs_d, rew_d, term_d, trunc_d, _ = self.env.step(action_dict)

        next_obs = self._dict_to_tensor(obs_d)
        rewards = torch.tensor(
            [float(rew_d.get(agent, 0.0)) for agent in self.agents],
            dtype=torch.float32,
        )
        dones = torch.tensor(
            [bool(term_d.get(agent, False) or trunc_d.get(agent, False)) for agent in self.agents],
            dtype=torch.bool,
        )
        next_state = next_obs.reshape(-1)

        return (
            next_obs.to(self.device),
            next_state.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
        )

    def _dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        ordered = []
        for agent in self.agents:
            if agent in obs_dict:
                obs = np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1)
            else:
                obs = np.zeros(self._obs_dim_by_agent[agent], dtype=np.float32)

            # Pad per-agent observation to common width for fixed-shape tensors.
            if obs.shape[0] < self.obs_dim:
                pad = np.zeros(self.obs_dim - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, pad], axis=0)
            elif obs.shape[0] > self.obs_dim:
                obs = obs[: self.obs_dim]

            ordered.append(obs)

        return torch.tensor(np.stack(ordered, axis=0), dtype=torch.float32)

    def render_frame(self) -> Optional[np.ndarray]:
        """Return one RGB frame when render_mode is set to rgb_array."""
        if self.render_mode != "rgb_array":
            return None
        frame = self.env.render()
        if frame is None:
            return None
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self.env.close()
