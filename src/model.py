"""Neural network model for predicting card strength."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class CardStrengthPredictor(nn.Module):
    """Model combining text and structured card features."""

    def __init__(self, vocab_size: int, feature_dim: int, config: Dict | None = None) -> None:
        config = config or {}
        super().__init__()

        embed_dim = config.get("embed_dim", 32)
        dense_units = config.get("hidden_dim", 64)
        dropout = config.get("dropout_rate", 0.0)

        # Text embedding followed by a small Transformer encoder
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Linear projection for structured numeric/categorical features
        self.feature_proj = nn.Linear(feature_dim, dense_units)

        # Final regression network with two hidden layers
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim + dense_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
        )

    def forward(self, features: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Return a strength score prediction for a batch of cards."""

        # Text sequence to embedding vector using a Transformer encoder
        embedded = self.embedding(text_tokens)  # [batch, seq_len, embed_dim]
        encoded = self.transformer(embedded.transpose(0, 1))
        text_vec = encoded.mean(dim=0)

        # Project structured features into the same hidden space
        struct_vec = self.feature_proj(features)

        # Concatenate and map to a single regression output
        x = torch.cat([text_vec, struct_vec], dim=1)
        return self.regressor(x)

