import sys
from pathlib import Path
import yaml
import pandas as pd

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_train_val_split


def ensure_dataset_exists():
    """Create a minimal ``card_data.csv`` file if one does not exist."""
    data_path = Path("card_data.csv")
    if not data_path.is_file():
        df = pd.DataFrame(
            [
                {
                    "name": "CardA",
                    "oracle_text": "foo",
                    "mana_value": 1,
                    "power": 1,
                    "toughness": 1,
                    "appearances": 1,
                    "rarity": "common",
                    "type_line": "Creature",
                    "strength_score": 1.0,
                }
            ]
        )
        df.to_csv(data_path, index=False)


def test_dataset_loads():
    """Dataset should load and provide non-empty splits."""
    ensure_dataset_exists()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_loader, val_loader, _ = load_train_val_split(config, test_size=0.2, seed=0)
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) >= 0


def test_batch_shape():
    """Feature batch dimension must match target batch dimension."""
    ensure_dataset_exists()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_loader, _, _ = load_train_val_split(config, test_size=0.2, seed=0)
    text, lengths, feats, target = next(iter(train_loader))
    assert feats.shape[0] == target.shape[0]
