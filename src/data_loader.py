"""Utilities for loading card data for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

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
        if "mana_value" not in self.df.columns:
            if "mana_cost" in self.df.columns:
                self.df["mana_value"] = self.df["mana_cost"]
            else:
                self.df["mana_value"] = 0
        if "appearances" not in self.df.columns:
            self.df["appearances"] = 0
        self.df["appearances"] = self.df["appearances"].fillna(0)
        self.df["log_appearances"] = np.log(self.df["appearances"].astype(float) + 1)
        num_cols = ["mana_value", "power", "toughness", "appearances", "log_appearances"]
        self.num_cols = num_cols
        self.df[num_cols] = (
            self.df[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        self.means = self.df[num_cols].mean()
        self.stds = self.df[num_cols].std().replace(0, 1)
        normed = (self.df[num_cols] - self.means) / self.stds

        # Categorical features to one-hot encode
        cat_cols = ["rarity"]
        self.df[cat_cols] = self.df[cat_cols].fillna("unknown").astype(str)
        cat_maps = {
            col: {v: i for i, v in enumerate(sorted(self.df[col].unique()))}
            for col in cat_cols
        }
        self.cat_maps = cat_maps
        self.rarity_eye = {
            col: torch.eye(len(cat_maps[col])) for col in cat_maps
        }
        cat_feats = []
        for col in cat_cols:
            idxs = self.df[col].map(cat_maps[col]).astype(int).values
            one_hot = self.rarity_eye[col][idxs]
            cat_feats.append(one_hot)

        # Type line one-hot for major card types
        self.type_categories = [
            "Creature",
            "Artifact",
            "Enchantment",
            "Planeswalker",
            "Land",
            "Instant",
            "Sorcery",
        ]
        type_features = []
        for tl in self.df["type_line"].fillna("").astype(str):
            type_features.append([1.0 if t in tl else 0.0 for t in self.type_categories])
        type_tensor = torch.tensor(type_features, dtype=torch.float32)

        # Color flags W,U,B,R,G plus Colorless(C)
        self.color_letters = ["W", "U", "B", "R", "G"]
        color_feats = []
        for cstr in self.df["colors"].fillna("").astype(str):
            letters = set(cstr)
            flags = [1.0 if l in letters else 0.0 for l in self.color_letters]
            if letters:
                flags.append(1.0 if "C" in letters else 0.0)
            else:
                flags.append(1.0)
            color_feats.append(flags)
        color_tensor = torch.tensor(color_feats, dtype=torch.float32)

        # Mana curve buckets
        cmc = self.df["mana_value"].fillna(0).astype(float)
        cmc_tensor = torch.stack(
            [
                torch.tensor((cmc <= 1).astype(float)),
                torch.tensor(((cmc >= 2) & (cmc <= 3)).astype(float)),
                torch.tensor((cmc >= 4).astype(float)),
            ],
            dim=1,
        )

        # Combine all structured features into a single tensor
        numeric_tensor = torch.tensor(normed.values, dtype=torch.float32)
        categorical_tensor = torch.cat(cat_feats, dim=1).float()
        self.features = torch.cat(
            [numeric_tensor, categorical_tensor, type_tensor, color_tensor, cmc_tensor],
            dim=1,
        )
        self.feature_dim = self.features.shape[1]

        # Regression targets
        self.targets = torch.tensor(
            self.df["strength_score"].fillna(0).values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = torch.tensor(self.tokens[idx], dtype=torch.long)
        feats = self.features[idx]
        target = self.targets[idx]
        return text, feats, target

    # ------------------------------------------------------------------
    def vectorize_row(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorize a single row using the dataset's preprocessing."""
        row = row.fillna(0)
        if "mana_value" not in row and "mana_cost" in row:
            row["mana_value"] = row["mana_cost"]
        row["appearances"] = row.get("appearances", 0)
        row["log_appearances"] = np.log(float(row["appearances"]) + 1)
        numeric = (row[self.num_cols] - self.means) / self.stds
        numeric_tensor = torch.tensor(numeric.values, dtype=torch.float32)

        # rarity
        rarity = row.get("rarity", "unknown")
        idx = self.cat_maps["rarity"].get(rarity, 0)
        rarity_tensor = self.rarity_eye["rarity"][idx]

        # type line features
        tl = str(row.get("type_line", ""))
        type_tensor = torch.tensor([1.0 if t in tl else 0.0 for t in self.type_categories], dtype=torch.float32)

        # color flags
        letters = set(str(row.get("colors", "")))
        flags = [1.0 if l in letters else 0.0 for l in self.color_letters]
        if letters:
            flags.append(1.0 if "C" in letters else 0.0)
        else:
            flags.append(1.0)
        color_tensor = torch.tensor(flags, dtype=torch.float32)

        cmc = float(row.get("mana_value", 0))
        cmc_tensor = torch.tensor([cmc <= 1, 2 <= cmc <= 3, cmc >= 4], dtype=torch.float32)

        feats = torch.cat([numeric_tensor, rarity_tensor.float(), type_tensor, color_tensor, cmc_tensor])

        text_tokens = torch.tensor(_encode(str(row.get("oracle_text", "")), self.vocab), dtype=torch.long)

        return text_tokens, feats


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

