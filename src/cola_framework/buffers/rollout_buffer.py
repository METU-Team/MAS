"""Fixed-length on-policy rollout buffer with GAE advantage computation."""

from typing import Dict, Iterator

import torch


class RolloutBuffer:
    """Stores a fixed-length rollout for on-policy PPO-style training.

    Unlike the circular ReplayBuffer, this buffer is filled once per update
    cycle and then cleared. It computes Generalized Advantage Estimation (GAE)
    in-place before the update step.

    Shape conventions:
        obs        : [n_steps, n_agents, obs_dim]
        state      : [n_steps, state_dim]
        actions    : [n_steps, n_agents, action_dim]
        rewards    : [n_steps, n_agents]
        dones      : [n_steps, n_agents]  — True when episode ended after step t
        values     : [n_steps, n_agents]  — V(s_t) from value function
        log_probs  : [n_steps, n_agents]  — log π(a_t | o_t, c_t)
        advantages : [n_steps, n_agents]  — filled by compute_returns_and_advantages()
        returns    : [n_steps, n_agents]  — advantages + values (target for V)
    """

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.ptr = 0

        self.obs = torch.zeros(n_steps, n_agents, obs_dim, device=device)
        self.state = torch.zeros(n_steps, state_dim, device=device)
        self.actions = torch.zeros(n_steps, n_agents, action_dim, device=device)
        self.rewards = torch.zeros(n_steps, n_agents, device=device)
        self.dones = torch.zeros(n_steps, n_agents, device=device)
        self.values = torch.zeros(n_steps, n_agents, device=device)
        self.log_probs = torch.zeros(n_steps, n_agents, device=device)
        self.advantages = torch.zeros(n_steps, n_agents, device=device)
        self.returns = torch.zeros(n_steps, n_agents, device=device)

    def push(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """Store one timestep of multi-agent data.

        obs      : [n_agents, obs_dim]
        state    : [state_dim]
        actions  : [n_agents, action_dim]
        rewards  : [n_agents]
        dones    : [n_agents]
        values   : [n_agents]
        log_probs: [n_agents]
        """
        if self.ptr >= self.n_steps:
            raise RuntimeError("RolloutBuffer is full. Call clear() before pushing more data.")

        def _store(arr, src):
            arr[self.ptr].copy_(src.detach() if isinstance(src, torch.Tensor) else torch.as_tensor(src, device=self.device))

        _store(self.obs, obs)
        _store(self.state, state)
        _store(self.actions, actions)
        _store(self.rewards, rewards)
        _store(self.dones, dones)
        _store(self.values, values)
        _store(self.log_probs, log_probs)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
    ) -> None:
        """Compute GAE advantages and TD(λ) returns in-place.

        last_values : [n_agents] — V(s_T), bootstrapped from state after last step
        last_dones  : [n_agents] — whether the episode ended on the last step

        Works for both full (ptr == n_steps) and partial (ptr < n_steps) buffers,
        which occurs when max_steps is reached mid-rollout.
        """
        if self.ptr == 0:
            raise RuntimeError("Buffer is empty. Push at least one transition first.")

        T = self.ptr  # actual number of stored steps (may be less than n_steps)
        last_values = last_values.detach().to(self.device)
        last_dones = last_dones.float().to(self.device)
        last_gae = torch.zeros(self.n_agents, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                # Bootstrap from the state after the last rollout step.
                # If the episode ended on that step, next value is 0.
                next_non_terminal = 1.0 - last_dones
                next_value = last_values
            else:
                # dones[t] is True if the episode ended AFTER step t, so the
                # value at t+1 belongs to a new episode and must not bootstrap.
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get(self) -> Dict[str, torch.Tensor]:
        """Return full rollout as a dict of tensors.

        All tensors have shape [n_steps, n_agents, ...].
        advantages and returns are only valid after compute_returns_and_advantages().
        """
        return {
            "obs": self.obs[:self.ptr],
            "state": self.state[:self.ptr],
            "actions": self.actions[:self.ptr],
            "advantages": self.advantages[:self.ptr],
            "returns": self.returns[:self.ptr],
            "old_log_probs": self.log_probs[:self.ptr],
            "values": self.values[:self.ptr],
        }

    def get_minibatches(self, n_minibatches: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield randomly shuffled minibatches of timestep slices.

        Each minibatch dict has the same keys as get() but with leading
        dimension = n_steps // n_minibatches instead of n_steps.
        Timestep ordering is shuffled so adjacent steps are decorrelated.
        """
        T = self.ptr
        if T == 0:
            raise RuntimeError("Buffer is empty. Push at least one transition first.")
        if T < n_minibatches:
            raise RuntimeError(
                f"Not enough steps ({T}) to create {n_minibatches} minibatches."
            )

        perm = torch.randperm(T, device=self.device)
        mb_size = T // n_minibatches

        for start in range(0, n_minibatches * mb_size, mb_size):
            idx = perm[start : start + mb_size]
            yield {
                "obs": self.obs[idx],
                "state": self.state[idx],
                "actions": self.actions[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
                "old_log_probs": self.log_probs[idx],
                "values": self.values[idx],
            }

    def clear(self) -> None:
        """Reset the write pointer. Existing data is overwritten on next push()."""
        self.ptr = 0

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def __len__(self) -> int:
        return self.ptr
