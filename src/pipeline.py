# pipeline.py

"""
Neural network pipeline for evaluating Magic: The Gathering cards.

This version focuses on predicting a card's intrinsic strength based on its
oracle text and metadata rather than solely on tournament win rates.  The
implementation uses placeholder data to illustrate how an LSTM encoder can be
combined with structured features.

Real data should be collected from Scryfall and tournament websites.
"""

# ## Step 1: Define "strong" cards
# A strong card provides high intrinsic value based on its rules text and
# stats.  In a full system, we would correlate these features with deck win
# rates, but the model here learns directly from the card text and metadata.

# ## Step 2: Feature Specification
# Features used by the model:
# - oracle text (tokenized and fed into an LSTM)
# - mana_cost (numeric)
# - card_type (categorical)
# - power/toughness (numeric where applicable)

# ## Step 3: Data Collection Pipeline
# Placeholder functions to fetch card data from Scryfall and tournament
# results from external sites. Replace the sample data with real API calls
# and scraping logic.

import requests
import pandas as pd

SCRYFALL_API_URL = "https://api.scryfall.com/cards"


def fetch_card_data(card_name: str) -> dict:
    """Fetch card details from the Scryfall API (placeholder)."""
    # In real usage, make a GET request to Scryfall. Here we return
    # mock data for demonstration purposes.
    return {
        "name": card_name,
        "mana_cost": 2,
        "type_line": "Creature",
        "power": 2,
        "toughness": 2,
    }


def fetch_tournament_data() -> pd.DataFrame:
    """Collect decklists and win rates from online tournaments (placeholder)."""
    # For a real system, scrape tournament results or use an API that provides
    # deck lists and win percentages. These data could be used to supervise the
    # model. Here we return a minimal example for completeness but the network
    # below does not depend on it.
    return pd.DataFrame([
        {"deck": ["Card A", "Card B"], "win_rate": 0.55},
        {"deck": ["Card C", "Card A"], "win_rate": 0.60},
    ])


# ## Step 4: Preprocessing
# Convert card information into numerical features suitable for machine
# learning.  We build a small vocabulary from oracle text and map card types to
# integer IDs.  In practice this would be based on the entire MTG card set.

from typing import List, Dict
import numpy as np


def tokenize(text: str) -> List[str]:
    return text.lower().replace(",", "").replace(".", "").split()


def build_vocab(texts: List[str]) -> Dict[str, int]:
    tokens = sorted({tok for text in texts for tok in tokenize(text)})
    return {tok: idx + 1 for idx, tok in enumerate(tokens)}  # reserve 0 for PAD


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, 0) for tok in tokenize(text)]


def prepare_card_samples(cards: List[dict]):
    """Turn card dictionaries into tensors for training."""
    vocab = build_vocab([c["oracle_text"] for c in cards])
    types = sorted({c["type_line"] for c in cards})
    type_to_idx = {t: i for i, t in enumerate(types)}

    samples = []
    for card in cards:
        text_ids = encode_text(card["oracle_text"], vocab)
        struct = [
            card["mana_cost"],
            card.get("power", 0),
            card.get("toughness", 0),
            type_to_idx[card["type_line"]],
        ]
        samples.append((text_ids, struct, card["strength"]))

    return samples, vocab, type_to_idx


# ## Step 5: Neural Network Model
# PyTorch model that encodes oracle text with an LSTM, concatenates it with
# structured features, and outputs a card strength score.

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader


class CardStrengthNet(nn.Module):
    def __init__(self, vocab_size: int, type_vocab_size: int, embed_dim: int = 32, lstm_dim: int = 32):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_dim, batch_first=True)
        self.type_emb = nn.Embedding(type_vocab_size, 4)
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim + 4 + 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, text_tokens, lengths, struct_features):
        embedded = self.text_emb(text_tokens)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        text_vec = hidden[-1]

        type_idx = struct_features[:, -1].long()
        type_vec = self.type_emb(type_idx)
        numeric = struct_features[:, :3]

        combined = torch.cat([text_vec, type_vec, numeric], dim=1)
        return self.fc(combined)


def collate_fn(batch):
    texts, structs, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    padded_texts = pad_sequence([torch.tensor(t) for t in texts], batch_first=True)
    struct_tensor = torch.tensor(structs, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return padded_texts, lengths, struct_tensor, label_tensor


def train_model(samples, vocab_size, type_vocab_size):
    torch.manual_seed(0)
    loader = DataLoader(samples, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = CardStrengthNet(vocab_size, type_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        for text, lengths, struct, label in loader:
            optimizer.zero_grad()
            preds = model(text, lengths, struct)
            loss = loss_fn(preds, label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} loss: {loss.item():.4f}")
    return model


if __name__ == "__main__":
    # Example placeholder cards. Real data should include all relevant MTG
    # cards with accurate oracle text and stats.
    cards = [
        {
            "name": "Card A",
            "oracle_text": "When Card A enters the battlefield, draw a card",
            "mana_cost": 1,
            "type_line": "Creature",
            "power": 1,
            "toughness": 1,
            "strength": 0.8,
        },
        {
            "name": "Card B",
            "oracle_text": "Card B deals 2 damage to any target",
            "mana_cost": 2,
            "type_line": "Instant",
            "power": 0,
            "toughness": 0,
            "strength": 0.6,
        },
        {
            "name": "Card C",
            "oracle_text": "Creatures you control get +1/+1",
            "mana_cost": 3,
            "type_line": "Enchantment",
            "power": 0,
            "toughness": 0,
            "strength": 0.9,
        },
    ]

    samples, vocab, type_map = prepare_card_samples(cards)
    model = train_model(samples, len(vocab), len(type_map))
    # Placeholder: save the model or further evaluation here.
