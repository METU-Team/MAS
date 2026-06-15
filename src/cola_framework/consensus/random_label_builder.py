"""Random-label consensus builder — control ablation for COLA.

Assigns each agent a fixed random label that never changes during training.
This preserves the COLA architecture (consensus labels are still embedded and
fed to actors/critics) while destroying any view-invariant content: the label
carries no information about the observation at all. Comparing this against the
real ConsensusBuilder isolates how much of COLA's gain comes from the *learned*
signal versus from simply enlarging the policy input.
"""

from typing import Tuple

import torch
import torch.nn as nn

from cola_framework.interfaces.consensus import ConsensusModule


class RandomLabelConsensusBuilder(nn.Module, ConsensusModule):
    """Emits a fixed, observation-independent random label per agent.

    The per-agent labels are sampled once at construction from the configured
    seed and stored in a registered buffer, so they survive ``.to(device)`` and
    checkpointing and stay identical for the whole run.

    The dummy student Linear keeps the same opt_cb = Adam(builder.student.parameters())
    wiring in train_cola.py working without any special-casing.
    """

    def __init__(self, n_agents: int, k: int = 4, seed: int = 42) -> None:
        super().__init__()
        self.k = k
        self.n_agents = n_agents

        # Sample one fixed label per agent using a private generator so the
        # global RNG (and therefore the rest of training) is left untouched.
        generator = torch.Generator()
        generator.manual_seed(seed)
        labels = torch.randint(0, k, (n_agents,), generator=generator, dtype=torch.int64)
        # Registered buffer → moves with .to(device) and is saved in checkpoints.
        self.register_buffer("labels", labels)

        # Dummy parameter so Adam(self.student.parameters()) never gets an empty
        # list. Its gradient is always exactly 0, so it never affects the RL loss.
        self.student = nn.Linear(1, 1, bias=False)

    def forward(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero loss and the fixed per-agent labels broadcast across batch.

        obs_batch: [B, n_agents, obs_dim]
        returns:
            loss_cb: scalar 0.0 (connected to dummy parameter so backward() succeeds)
            consensus: [B, n_agents] int64, consensus[b, a] == fixed label for agent a
        """
        if obs_batch.ndim != 3:
            raise ValueError("obs_batch must have shape [B, n_agents, obs_dim].")
        B, n, _ = obs_batch.shape
        if n != self.n_agents:
            raise ValueError(
                "obs_batch has {} agents but builder was built for {}.".format(n, self.n_agents)
            )
        # Multiply by 0 so the gradient through the dummy param is exactly 0,
        # but backward() on this tensor does not raise.
        loss_cb = self.student.weight.sum() * 0.0
        consensus = self.labels.view(1, n).expand(B, n).contiguous()
        return loss_cb, consensus

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the fixed per-agent labels for one timestep.

        obs: [n_agents, obs_dim]
        returns: [n_agents] int64
        """
        if obs.ndim != 2:
            raise ValueError("obs must have shape [n_agents, obs_dim].")
        if obs.shape[0] != self.n_agents:
            raise ValueError(
                "obs has {} agents but builder was built for {}.".format(obs.shape[0], self.n_agents)
            )
        return self.labels.clone()

    @torch.no_grad()
    def ema_update_teacher(self) -> None:
        """No-op — no teacher network exists in the random-label builder."""
