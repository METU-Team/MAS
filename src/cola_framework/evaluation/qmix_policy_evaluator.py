"""Evaluation loop for QMIX policies (greedy argmax action selection)."""

from typing import Dict, List

import torch

from cola_framework.interfaces.evaluator import EvaluatorModule


class QMIXPolicyEvaluator(EvaluatorModule):
    """Greedy evaluation for QMIX Q-network policies.

    Selects actions by argmax over per-agent Q-values — no epsilon exploration.
    """

    def __init__(
        self,
        env,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        q_networks: List[torch.nn.Module],
    ) -> None:
        self.env = env
        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.q_networks = q_networks

        if len(self.q_networks) != self.env.n_agents:
            raise ValueError("Number of q_networks must match env.n_agents.")

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        returns = []
        total_steps = 0
        full_agreement_steps = 0.0

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_return = 0.0

            while not done:
                consensus = self.consensus_builder.infer(obs)
                cemb = self.embedding_layer(consensus)

                # Greedy argmax — no epsilon.
                actions = torch.stack(
                    [
                        self.q_networks[a](obs[a], cemb[a]).argmax()
                        for a in range(self.env.n_agents)
                    ],
                    dim=0,
                )

                obs, _, rewards, dones = self.env.step(actions)
                ep_return += float(rewards.mean().item())

                full_agreement_steps += float((consensus == consensus[0]).all().item())
                total_steps += 1
                done = bool(dones.all().item())

            returns.append(ep_return)

        mean_return = float(sum(returns) / max(1, len(returns)))
        eval_agreement = float(full_agreement_steps / max(1, total_steps))
        return {
            "mean_episode_return": mean_return,
            "consensus_agreement_eval": eval_agreement,
        }
