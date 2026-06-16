"""PettingZoo MPE wrapper that matches the COLA environment interface."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cola_framework.interfaces.environment import MultiAgentEnvironment


class MPEWrapper(MultiAgentEnvironment):
    """Wraps PettingZoo parallel MPE envs into tensor-first outputs.

    Supported scenarios:
    - simple_spread: fully cooperative, all ``n_agents`` are learner-controlled.
    - simple_tag: predator-prey. To match COLA's fully-cooperative assumption
      (and the original paper's setup), the learner controls a *cooperative team
      of ``n_agents`` predators* while the single prey is driven by a fixed
      heuristic flee policy. The prey is NOT learned and is excluded from the
      observation/consensus/critic tensors, so the consensus builder only ever
      sees the cooperative sub-team.

    Note:
    `simple_tag` predators share homogeneous observation sizes; we still pad each
    controlled observation to `self.obs_dim = max(obs_dim_per_controlled_agent)`
    so downstream modules can operate on fixed-size tensors.
    """

    def __init__(
        self,
        scenario: str = "simple_spread",
        n_agents: int = 3,
        max_cycles: int = 100,
        device: str = "cpu",
        render_mode: Optional[str] = None,
        heuristic_prey: bool = True,
        obs_mask: str = "none",
        disable_comm: bool = False,
    ) -> None:
        self.scenario = scenario
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.device = device
        self.render_mode = render_mode
        self.heuristic_prey = heuristic_prey
        self.obs_mask = obs_mask
        # Cooperative Pantomime: zero every agent's communication action so the
        # only channel left is movement/position (agents must infer each other's
        # hidden targets from behaviour). Continuous MPE actions are
        # [movement(5), comm(dim_c)], so comm occupies indices >= 5.
        self.disable_comm = disable_comm
        self._move_action_dim = 5

        self.env = self._build_env()
        self._all_agents = list(self.env.possible_agents)

        # Split env agents into learner-controlled and scripted (heuristic) sets.
        if self.scenario == "simple_tag":
            self.agents = [a for a in self._all_agents if a.startswith("adversary")]
            self.scripted_agents = [a for a in self._all_agents if not a.startswith("adversary")]
        else:
            self.agents = list(self._all_agents)
            self.scripted_agents = []

        # n_agents reported downstream is the number of *controlled* agents.
        self.n_agents = len(self.agents)

        # World handle (for scripted policies); positions update in place on reset.
        self._world = getattr(self.env.unwrapped, "world", None)

        obs_dict, _ = self.env.reset()
        self._obs_dim_by_agent = {
            agent: int(obs_dict[agent].shape[0])
            for agent in self._all_agents
        }
        # Dimensions are computed over the controlled agents only.
        self.obs_dim = max(self._obs_dim_by_agent[a] for a in self.agents)
        self.state_dim = self.obs_dim * self.n_agents
        self._action_dim_by_agent = {
            agent: int(self.env.action_space(agent).shape[0])
            for agent in self._all_agents
        }
        self.action_dim = max(self._action_dim_by_agent[a] for a in self.agents)

        # Continuous MPE actions are typically Box([0,1]); actor outputs are
        # in [-1,1], so we map them into the env range before stepping.
        self._action_low = {
            agent: self.env.action_space(agent).low.astype(np.float32)
            for agent in self._all_agents
        }
        self._action_high = {
            agent: self.env.action_space(agent).high.astype(np.float32)
            for agent in self._all_agents
        }

        # Partial-observability mask: zero out chosen slices of every controlled
        # observation so agents must coordinate through the consensus label rather
        # than reading neighbours directly. Applied in `_dict_to_tensor`, so both
        # actor/consensus inputs and the concatenated `state` reflect the masking.
        self._obs_keep = self._build_obs_keep_mask()

    def _build_obs_keep_mask(self) -> Optional[np.ndarray]:
        """Return a [obs_dim] float mask (1=keep, 0=hide), or None for no masking.

        Only ``simple_spread`` is supported; its per-agent layout for ``N`` agents
        (``N`` landmarks) is::

            [self_vel(2), self_pos(2), landmark_rel(2N), other_rel(2(N-1)), comm(2(N-1))]

        ``others`` hides the other-agents' relative positions; ``others_comm``
        also hides the communication channel; ``comm`` hides only communication.
        """
        if self.obs_mask in ("none", None):
            return None
        if self.scenario != "simple_spread":
            raise ValueError(
                "obs_mask='{}' is only supported for simple_spread (got scenario "
                "'{}').".format(self.obs_mask, self.scenario)
            )

        n = self.n_agents
        base = 4  # self_vel(2) + self_pos(2)
        others_start = base + 2 * n           # after landmark_rel(2N)
        comm_start = others_start + 2 * (n - 1)
        comm_end = comm_start + 2 * (n - 1)

        keep = np.ones(self.obs_dim, dtype=np.float32)
        if self.obs_mask == "others":
            keep[others_start:comm_start] = 0.0
        elif self.obs_mask == "comm":
            keep[comm_start:comm_end] = 0.0
        elif self.obs_mask == "others_comm":
            keep[others_start:comm_end] = 0.0
        elif self.obs_mask == "velocity":
            # Hide self-velocity (indices 0:2). Velocity = position deltas, so it
            # is RECOVERABLE from an observation *sequence* but not from a single
            # frame -> turns the Markov task into a POMDP where a history encoder
            # has something real to integrate (the right testbed for history).
            keep[0:2] = 0.0
        else:
            raise ValueError(
                "Unknown obs_mask '{}'. Use one of: none, others, comm, "
                "others_comm, velocity.".format(self.obs_mask)
            )
        return keep

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

            # Learner controls a cooperative team of `n_agents` predators; one
            # heuristic prey is added but not counted as a controlled agent.
            return simple_tag_v3.parallel_env(
                num_good=1,
                num_adversaries=self.n_agents,
                num_obstacles=2,
                max_cycles=self.max_cycles,
                continuous_actions=True,
                render_mode=self.render_mode,
            )

        if self.scenario == "simple_reference":
            from pettingzoo.mpe import simple_reference_v3

            # Cooperative Pantomime base: 2 agents, 3 landmarks. Each agent sees
            # the OTHER agent's target colour but not its own, so it must infer
            # its goal from the partner's behaviour. With ``disable_comm`` the
            # explicit message channel is removed (paper's Pantomime variant);
            # without it this is the Cooperative Communication / reference task.
            # n_agents is fixed at 2 by the env.
            return simple_reference_v3.parallel_env(
                local_ratio=0.5,
                max_cycles=self.max_cycles,
                continuous_actions=True,
                render_mode=self.render_mode,
            )

        raise ValueError(
            "Unsupported scenario '{}'. Use 'simple_spread', 'simple_tag' or "
            "'simple_reference'.".format(self.scenario)
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.env.reset(seed=seed)
        obs = self._dict_to_tensor(obs_dict)
        state = obs.reshape(-1)
        return obs.to(self.device), state.to(self.device)

    def step(self, action_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one environment step.

        action_tensor: [n_agents, action_dim] for the controlled agents only.
        returns:
            next_obs: [n_agents, obs_dim]
            next_state: [state_dim]
            rewards: [n_agents]
            dones: [n_agents] (bool)
        """
        action_tensor = action_tensor.detach().to("cpu")
        action_dict = {}

        # Controlled agents: map actor outputs [-1,1] -> env action range.
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
            scaled = np.clip(scaled, low, high)

            # Cooperative Pantomime: silence the communication channel so no
            # explicit messages are sent (indices >= movement dim).
            if self.disable_comm and scaled.shape[0] > self._move_action_dim:
                scaled[self._move_action_dim:] = low[self._move_action_dim:]

            action_dict[agent] = scaled

        # Scripted agents (e.g. prey): heuristic action already in env range.
        for agent in self.scripted_agents:
            action_dict[agent] = self._heuristic_prey_action(agent)

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

    def _heuristic_prey_action(self, agent_name: str) -> np.ndarray:
        """Fixed flee policy: move away from the nearest predator.

        Reads true positions from the world (not the learner) and emits a 5-dim
        continuous action [no_op, +x, -x, +y, +y] in [0,1] whose net force points
        away from the closest adversary, with a soft push back inside the arena
        so the prey does not simply run off-screen.
        """
        action_dim = self._action_dim_by_agent[agent_name]
        action = np.zeros(action_dim, dtype=np.float32)

        if self._world is None:
            return action  # No world access -> stay still (degenerate but safe).

        by_name = {ag.name: ag for ag in self._world.agents}
        prey = by_name.get(agent_name)
        if prey is None:
            return action
        prey_pos = np.asarray(prey.state.p_pos, dtype=np.float32)

        adv_positions = [
            np.asarray(ag.state.p_pos, dtype=np.float32)
            for ag in self._world.agents
            if getattr(ag, "adversary", False)
        ]
        if not adv_positions:
            return action

        # Direction away from the nearest predator.
        dists = [np.linalg.norm(prey_pos - p) for p in adv_positions]
        nearest = adv_positions[int(np.argmin(dists))]
        flee = prey_pos - nearest
        norm = float(np.linalg.norm(flee))
        direction = flee / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        # Soft boundary avoidance: if near/outside the arena edge, steer inward.
        for c in (0, 1):
            if prey_pos[c] > 1.0 and direction[c] > 0:
                direction[c] = -abs(direction[c])
            elif prey_pos[c] < -1.0 and direction[c] < 0:
                direction[c] = abs(direction[c])

        # Map (dx, dy) force into the [no_op, +x, -x, +y, +y] continuous slots.
        if action_dim >= 5:
            action[1] = max(float(direction[0]), 0.0)
            action[2] = max(float(-direction[0]), 0.0)
            action[3] = max(float(direction[1]), 0.0)
            action[4] = max(float(-direction[1]), 0.0)
        return np.clip(action, 0.0, 1.0)

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

            if self._obs_keep is not None:
                obs = obs * self._obs_keep

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
