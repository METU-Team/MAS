"""Step 4 module gate test for consensus embedding."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.embedding.consensus_embedding import ConsensusEmbedding


def main() -> None:
    torch.manual_seed(11)

    emb_layer = ConsensusEmbedding(k=4, emb_dim=16)

    # Batch contract: [B, n_agents] -> [B, n_agents, emb_dim]
    cons_batch = torch.randint(0, 4, (32, 3), dtype=torch.int64)
    emb_batch = emb_layer(cons_batch)
    assert emb_batch.shape == (32, 3, 16)
    assert emb_batch.dtype == torch.float32

    # Single-step contract: [n_agents] -> [n_agents, emb_dim]
    cons_single = torch.randint(0, 4, (3,), dtype=torch.int64)
    emb_single = emb_layer(cons_single)
    assert emb_single.shape == (3, 16)

    # Projection parameters should still get gradients from RL losses.
    loss = emb_batch.pow(2).mean()
    emb_layer.zero_grad()
    loss.backward()
    assert emb_layer.proj.weight.grad is not None

    print("Step 4 gate passed: Consensus embedding OK")


if __name__ == "__main__":
    main()
