"""PettingZoo MPE wrapper that matches the COLA environment interface."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from cola_framework.interfaces.environment import MultiAgentEnvironment


class MPEWrapper(MultiAgentEnvironment):
    """Wraps PettingZoo parallel MPE envs into tensor-first outputs.

    Supported scenarios:
    - simple_spread
    - simple_tag
    """

    def __init__(
        self,
        scenario: str = "simple_spread",
        n_agents: int = 3,
        max_cycles: int = 100,
        device: str = "cpu",
    ) -> None:
        self.scenario = scenario
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.device = device

        self.env = self._build_env()
        self.agents = list(self.env.possible_agents)

        obs_dict, _ = self.env.reset()
        self.obs_dim = int(obs_dict[self.agents[0]].shape[0])
        self.state_dim = self.obs_dim * self.n_agents
        self.action_dim = int(self.env.action_space(self.agents[0]).shape[0])

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
            raw_action = action_tensor[i].numpy()
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
                ordered.append(obs_dict[agent])
            else:
                ordered.append(np.zeros(self.obs_dim, dtype=np.float32))
        return torch.tensor(np.stack(ordered, axis=0), dtype=torch.float32)
