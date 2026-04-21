"""Step 7 module gate test for MADDPG update logic."""

import copy
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.consensus.builder import ConsensusBuilder
from cola_framework.critics.centralized_critic import Critic
from cola_framework.embedding.consensus_embedding import ConsensusEmbedding
from cola_framework.policies.actor import Actor
from cola_framework.trainers.maddpg_updater import MADDPGUpdater


def _clone_first_param(module: torch.nn.Module) -> torch.Tensor:
    return next(module.parameters()).detach().clone()


def main() -> None:
    torch.manual_seed(101)

    batch_size = 32
    n_agents = 3
    obs_dim = 18
    action_dim = 5
    emb_dim = 16
    state_dim = n_agents * obs_dim

    cb = ConsensusBuilder(obs_dim=obs_dim, k=4, hidden_dim=64)
    emb = ConsensusEmbedding(k=4, emb_dim=emb_dim)

    actors = [Actor(obs_dim=obs_dim, emb_dim=emb_dim, action_dim=action_dim) for _ in range(n_agents)]
    critics = [
        Critic(state_dim=state_dim, n_agents=n_agents, emb_dim=emb_dim, action_dim=action_dim)
        for _ in range(n_agents)
    ]

    target_actors = [copy.deepcopy(actor) for actor in actors]
    target_critics = [copy.deepcopy(critic) for critic in critics]

    opt_cb = torch.optim.Adam(cb.student.parameters(), lr=3e-4)
    opt_actors = [torch.optim.Adam(actor.parameters(), lr=1e-3) for actor in actors]
    opt_critics = [torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in critics]

    updater = MADDPGUpdater(
        consensus_builder=cb,
        embedding_layer=emb,
        actors=actors,
        critics=critics,
        target_actors=target_actors,
        target_critics=target_critics,
        opt_cb=opt_cb,
        opt_actors=opt_actors,
        opt_critics=opt_critics,
        gamma=0.95,
        tau_polyak=0.01,
    )

    batch = {
        "obs": torch.randn(batch_size, n_agents, obs_dim),
        "state": torch.randn(batch_size, state_dim),
        "actions": torch.tanh(torch.randn(batch_size, n_agents, action_dim)),
        "rewards": torch.randn(batch_size, n_agents),
        "next_obs": torch.randn(batch_size, n_agents, obs_dim),
        "next_state": torch.randn(batch_size, state_dim),
        "dones": torch.randint(0, 2, (batch_size, n_agents)).float(),
    }

    actor_before = _clone_first_param(actors[0])
    critic_before = _clone_first_param(critics[0])
    teacher_before = _clone_first_param(cb.teacher)
    target_actor_before = _clone_first_param(target_actors[0])

    metrics = updater.update(batch)

    assert "loss_cb" in metrics
    assert "critic_losses" in metrics
    assert "actor_losses" in metrics
    assert len(metrics["critic_losses"]) == n_agents
    assert len(metrics["actor_losses"]) == n_agents

    # Check that core modules were updated.
    actor_after = _clone_first_param(actors[0])
    critic_after = _clone_first_param(critics[0])
    teacher_after = _clone_first_param(cb.teacher)
    target_actor_after = _clone_first_param(target_actors[0])

    assert (actor_after - actor_before).abs().sum().item() > 0.0
    assert (critic_after - critic_before).abs().sum().item() > 0.0
    assert (teacher_after - teacher_before).abs().sum().item() > 0.0
    assert (target_actor_after - target_actor_before).abs().sum().item() > 0.0

    print("Step 7 gate passed: MADDPG update logic OK")


if __name__ == "__main__":
    main()
