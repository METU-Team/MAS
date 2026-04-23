"""Step 11 gate test for concrete history encoder classes."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.encoders.gru_encoder import GRUHistoryEncoder
from cola_framework.encoders.identity_encoder import IdentityHistoryEncoder
from cola_framework.encoders.transformer_encoder import TransformerHistoryEncoder
from cola_framework.encoders.window_encoder import WindowConcatEncoder


def main() -> None:
    torch.manual_seed(19)

    batch_size = 16
    window = 10
    obs_dim = 18

    identity = IdentityHistoryEncoder(obs_dim=obs_dim)
    out_identity = identity(torch.randn(batch_size, 1, obs_dim))
    assert out_identity.shape == (batch_size, obs_dim)

    try:
        identity(torch.randn(batch_size, 2, obs_dim))
        raise AssertionError("IdentityHistoryEncoder should reject T != 1.")
    except ValueError:
        pass

    gru = GRUHistoryEncoder(obs_dim=obs_dim, hidden_dim=64)
    out_gru = gru(torch.randn(batch_size, window, obs_dim))
    assert out_gru.shape == (batch_size, 64)

    concat = WindowConcatEncoder(obs_dim=obs_dim, window=window, out_dim=64)
    out_concat = concat(torch.randn(batch_size, window, obs_dim))
    assert out_concat.shape == (batch_size, 64)

    transformer = TransformerHistoryEncoder(
        obs_dim=obs_dim,
        out_dim=64,
        nhead=4,
        num_layers=2,
        max_len=32,
    )
    out_transformer = transformer(torch.randn(batch_size, window, obs_dim))
    assert out_transformer.shape == (batch_size, 64)

    print("Step 11 gate passed: Concrete history encoders OK")


if __name__ == "__main__":
    main()
