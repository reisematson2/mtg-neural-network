"""Utilities for loading card data for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer used for oracle text."""
    return text.lower().replace("\n", " ").split()


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    """Create a vocabulary mapping token to index."""
    tokens = {tok for t in texts for tok in _tokenize(t)}
    # Reserve index 0 for padding
    return {tok: i + 1 for i, tok in enumerate(sorted(tokens))}


def _encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert a text string to a list of token ids."""
    return [vocab.get(tok, 0) for tok in _tokenize(text)]


class CardDataset(Dataset):
    """Dataset that loads card data and preprocessing state."""

    def __init__(self, config: dict) -> None:
        # Read the CSV containing card data
        paths_cfg = config.get("paths", {})
        data_path = Path(paths_cfg.get("data_csv", paths_cfg.get("data", "card_data.csv")))
        df = pd.read_csv(data_path)

        # Store raw dataframe for later inspection
        self.df = df.reset_index(drop=True)

        # Tokenize oracle_text and build vocabulary over entire dataset
        texts = self.df["oracle_text"].fillna("").astype(str).tolist()
        self.vocab = _build_vocab(texts)
        self.tokens = [_encode(t, self.vocab) for t in texts]

        # Numeric features to normalize
        num_cols = ["mana_value", "power", "toughness", "appearances"]
        # Ensure all required numeric columns exist
        for col in num_cols:
            if col not in self.df.columns:
                self.df[col] = 0

        # Fill missing values and robustly enforce float dtype for numeric columns
        for col in num_cols:
            # Coerce non-numeric values to NaN, then fill with 0 and convert to float
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(float)
            if (self.df[col] == 0).sum() > 0 and self.df[col].isna().sum() > 0:
                print(f"Warning: Non-numeric values found in column '{col}' and coerced to 0.")

        means = self.df[num_cols].mean()
        stds = self.df[num_cols].std().replace(0, 1)
        normed = (self.df[num_cols] - means) / stds

        # Categorical features to one-hot encode
        cat_cols = ["rarity", "type_line"]
        self.df[cat_cols] = self.df[cat_cols].fillna("unknown").astype(str)
        cat_maps = {
            col: {v: i for i, v in enumerate(sorted(self.df[col].unique()))}
            for col in cat_cols
        }
        cat_feats = []
        for col in cat_cols:
            idxs = self.df[col].map(cat_maps[col]).astype(int).values
            one_hot = torch.eye(len(cat_maps[col]))[idxs]
            cat_feats.append(one_hot)

        # Combine all structured features into a single tensor
        numeric_tensor = torch.tensor(normed.values, dtype=torch.float32)
        categorical_tensor = torch.cat(cat_feats, dim=1).float()
        self.features = torch.cat([numeric_tensor, categorical_tensor], dim=1)

        # Ensure regression targets are float
        self.df['strength_score'] = pd.to_numeric(self.df['strength_score'], errors='coerce').fillna(0).astype(float)
        self.targets = torch.tensor(
            self.df["strength_score"].values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = torch.tensor(self.tokens[idx], dtype=torch.long)
        feats = self.features[idx]
        target = self.targets[idx]
        return text, feats, target


def _pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Custom collate_fn to pad text sequences."""
    texts, feats, targets = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    feat_tensor = torch.stack(feats)
    targ_tensor = torch.stack(targets)
    return padded, lengths, feat_tensor, targ_tensor


def load_train_val_split(
    config: dict, test_size: float = 0.2, seed: int = 42
) -> Tuple[DataLoader, DataLoader, CardDataset]:
    """Create ``DataLoader`` objects for training and validation."""

    dataset = CardDataset(config)

    val_len = int(len(dataset) * test_size)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator)

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_pad_collate
    )

    return train_loader, val_loader, dataset

