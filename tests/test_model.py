import sys
from pathlib import Path
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_loader import load_train_val_split
from model import CardStrengthPredictor


def test_forward_pass():
    """Ensure a forward pass returns the correct output shape."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config["model"]

    _, _, dataset = load_train_val_split(config, test_size=0.2, seed=0)

    model = CardStrengthPredictor(
        vocab_size=len(dataset.vocab),
        feature_dim=dataset.features.shape[1],
        config={
            "embed_dim": params["embed_dim"],
            "lstm_dim": params["lstm_dim"],
            "hidden_dim": params["hidden_dim"],
            "dropout_rate": params.get("dropout_rate", 0.0),
        },
    )

    dummy_tokens = torch.randint(0, len(dataset.vocab), (4, 10))
    dummy_feats = torch.randn(4, dataset.features.shape[1])
    out = model(dummy_feats, dummy_tokens)
    assert out.shape == (4, 1)
