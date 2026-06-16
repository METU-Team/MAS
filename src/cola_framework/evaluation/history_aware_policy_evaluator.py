"""Evaluation loop for history-aware COLA policies."""

from typing import Dict, List

import torch

from cola_framework.interfaces.evaluator import EvaluatorModule


class HistoryAwarePolicyEvaluator(EvaluatorModule):
    """Runs greedy evaluation using observation history windows."""

    def __init__(
        self,
        env,
        window_manager,
        consensus_builder: torch.nn.Module,
        embedding_layer: torch.nn.Module,
        actors: List[torch.nn.Module],
    ) -> None:
        self.env = env
        self.window_manager = window_manager
        self.consensus_builder = consensus_builder
        self.embedding_layer = embedding_layer
        self.actors = actors

        if len(self.actors) != self.env.n_agents:
            raise ValueError("Number of actors must match env.n_agents.")

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        returns = []
        total_steps = 0
        full_agreement_steps = 0.0
        labels: List[torch.Tensor] = []
        distinct_per_ep: List[int] = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            self.window_manager.reset()
            self.window_manager.push(obs)

            done = False
            ep_return = 0.0
            ep_labels: List[torch.Tensor] = []

            while not done:
                obs_seq = self.window_manager.get()
                consensus = self.consensus_builder.infer(obs_seq)
                ep_labels.append(consensus)
                cemb = self.embedding_layer(consensus)

                actions = torch.stack(
                    [
                        self.actors[a](obs[a], cemb[a])
                        for a in range(self.env.n_agents)
                    ],
                    dim=0,
                ).clamp(-1.0, 1.0)

                obs, _, rewards, dones = self.env.step(actions)
                self.window_manager.push(obs)

                ep_return += float(rewards.mean().item())
                full_agreement_steps += float((consensus == consensus[0]).all().item())
                total_steps += 1
                done = bool(dones.all().item())

            returns.append(ep_return)
            ep_lab = torch.stack(ep_labels)
            labels.append(ep_lab.reshape(-1))
            distinct_per_ep.append(int(ep_lab.unique().numel()))

        # Consensus-label diversity, matching the per-timestep evaluator so the
        # collapse metrics (entropy, distinct classes) are comparable across the
        # vanilla and history-aware paths.
        flat = torch.cat(labels)
        k = int(getattr(self.consensus_builder, "k", int(flat.max().item()) + 1))
        counts = torch.bincount(flat, minlength=k).float()
        probs = counts / counts.sum().clamp(min=1.0)
        entropy = float(-(probs * (probs + 1e-8).log()).sum().item())

        mean_return = float(sum(returns) / max(1, len(returns)))
        eval_agreement = float(full_agreement_steps / max(1, total_steps))
        return {
            "mean_episode_return": mean_return,
            "eval_episode_return": mean_return,
            "eval_consensus_entropy": entropy,
            "eval_distinct_classes": float(sum(distinct_per_ep) / max(1, len(distinct_per_ep))),
            "consensus_agreement_eval": eval_agreement,
        }
