"""Utilities for loading card data for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
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

        # -------------------------
        # Structured feature engineering
        # -------------------------
        # Base numeric columns
        num_cols = ["mana_value", "power", "toughness", "appearances"]
        for col in num_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # Fill and coerce numerics
        for col in num_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(float)

        # Add log transform of appearances
        self.df["log_appearances"] = np.log1p(self.df["appearances"]).astype(float)
        num_cols.append("log_appearances")

        # Compute normalization stats for numeric features
        self.means = self.df[num_cols].mean()
        self.stds = self.df[num_cols].std().replace(0, 1)
        normed = (self.df[num_cols] - self.means) / self.stds

        # ----- categorical: rarity -----
        self.df["rarity"] = self.df["rarity"].fillna("unknown").astype(str)
        self.rarity_map = {v: i for i, v in enumerate(sorted(self.df["rarity"].unique()))}
        rarity_idxs = self.df["rarity"].map(self.rarity_map).astype(int).values
        rarity_oh = torch.eye(len(self.rarity_map))[rarity_idxs]

        # ----- type line specific flags -----
        type_categories = ["Creature", "Artifact", "Enchantment", "Planeswalker", "Land", "Instant", "Sorcery"]
        for cat in type_categories:
            self.df[f"type_{cat.lower()}"] = self.df["type_line"].str.contains(cat, case=False, na=False).astype(int)
        type_flags = torch.tensor(self.df[[f"type_{c.lower()}" for c in type_categories]].values, dtype=torch.float32)

        # ----- color identity flags -----
        color_codes = ["W", "U", "B", "R", "G", "C"]
        if "colors" not in self.df.columns:
            self.df["colors"] = ""
        self.df["colors"] = self.df["colors"].fillna("").astype(str)
        for c in color_codes:
            self.df[f"color_{c}"] = self.df["colors"].str.contains(c, case=False, na=False).astype(int)
        color_flags = torch.tensor(self.df[[f"color_{c}" for c in color_codes]].values, dtype=torch.float32)

        # ----- mana curve buckets -----
        self.df["cmc_0_1"] = (self.df["mana_value"] <= 1).astype(int)
        self.df["cmc_2_3"] = ((self.df["mana_value"] >= 2) & (self.df["mana_value"] <= 3)).astype(int)
        self.df["cmc_4_plus"] = (self.df["mana_value"] >= 4).astype(int)
        cmc_flags = torch.tensor(self.df[["cmc_0_1", "cmc_2_3", "cmc_4_plus"]].values, dtype=torch.float32)

        # Combine structured features
        numeric_tensor = torch.tensor(normed.values, dtype=torch.float32)
        self.features = torch.cat([numeric_tensor, rarity_oh, type_flags, color_flags, cmc_flags], dim=1)

        self.feature_dim = self.features.shape[1]

        self.cat_maps = {"rarity": self.rarity_map}

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

    def featurize_row(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a single card row to tensors using stored preprocessing state."""
        tokens = torch.tensor(_encode(str(row.get("oracle_text", "")), self.vocab), dtype=torch.long)

        # numeric
        data = {
            "mana_value": float(row.get("mana_value", 0)),
            "power": float(row.get("power", 0)),
            "toughness": float(row.get("toughness", 0)),
            "appearances": float(row.get("appearances", 0)),
        }
        data["log_appearances"] = np.log1p(data["appearances"])
        num_vec = [(data[c] - self.means[c]) / self.stds[c] for c in self.means.index]
        num_tensor = torch.tensor(num_vec, dtype=torch.float32)

        # rarity one-hot
        rarity = str(row.get("rarity", "unknown"))
        r_idx = self.rarity_map.get(rarity, 0)
        rarity_tensor = torch.eye(len(self.rarity_map))[r_idx]

        # type flags
        type_categories = ["Creature", "Artifact", "Enchantment", "Planeswalker", "Land", "Instant", "Sorcery"]
        type_vals = [int(str(row.get("type_line", "")).lower().find(cat.lower()) != -1) for cat in type_categories]
        type_tensor = torch.tensor(type_vals, dtype=torch.float32)

        # color flags
        colors = str(row.get("colors", ""))
        color_codes = ["W", "U", "B", "R", "G", "C"]
        color_vals = [int(code in colors) for code in color_codes]
        color_tensor = torch.tensor(color_vals, dtype=torch.float32)

        # mana curve
        mana = data["mana_value"]
        cmc_vals = [int(mana <= 1), int(2 <= mana <= 3), int(mana >= 4)]
        cmc_tensor = torch.tensor(cmc_vals, dtype=torch.float32)

        feat = torch.cat([num_tensor, rarity_tensor, type_tensor, color_tensor, cmc_tensor])
        return tokens, feat


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

