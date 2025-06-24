import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CardStrengthPredictor(nn.Module):
    """LSTM-based model for card strength prediction."""

    def __init__(self, vocab_size: int, feature_dim: int, embed_dim: int = 32,
                 lstm_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_tokens: torch.Tensor, lengths: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        embedded = self.text_emb(text_tokens)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        text_vec = hidden[-1]
        x = torch.cat([text_vec, features], dim=1)
        return self.fc(x)
