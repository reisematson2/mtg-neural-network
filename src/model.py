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
        lstm_dim = config.get("lstm_dim", 32)
        dense_units = config.get("hidden_dim", 64)
        dropout = config.get("dropout_rate", 0.0)

        # Text embedding followed by a single LSTM layer
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_dim, batch_first=True)

        # Linear projection for structured numeric/categorical features
        self.feature_proj = nn.Linear(feature_dim, dense_units)

        # Final regression network with two hidden layers
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim + dense_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
        )

    def forward(self, features: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Return a strength score prediction for a batch of cards."""

        # Text sequence to embedding vector using the last LSTM hidden state
        embedded = self.embedding(text_tokens)
        _, (hidden, _) = self.lstm(embedded)
        text_vec = hidden[-1]

        # Project structured features into the same hidden space
        struct_vec = self.feature_proj(features)

        # Concatenate and map to a single regression output
        x = torch.cat([text_vec, struct_vec], dim=1)
        return self.regressor(x)

