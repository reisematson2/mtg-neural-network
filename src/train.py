import argparse
from pathlib import Path
import yaml

import torch
from torch import nn

from src.data_loader import load_train_val_split
from src.model import CardStrengthPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train card strength model")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def train_model(train_loader, val_loader, vocab_size, feature_dim, config, device, epochs=5, checkpoint_dir="checkpoints"):
    """Train ``CardStrengthPredictor`` and return the final validation loss."""
    model = CardStrengthPredictor(vocab_size, feature_dim, config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    loss_fn = nn.MSELoss()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for text, lengths, feats, labels in train_loader:
            text = text.to(device)
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(feats, text)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for text, lengths, feats, labels in val_loader:
                text = text.to(device)
                feats, labels = feats.to(device), labels.to(device)
                preds = model(feats, text)
                loss = loss_fn(preds, labels)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    return val_loss


def main():
    args = parse_args()
    config = {}
    if Path(args.config).is_file():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})

    torch.manual_seed(train_cfg.get("seed", 42))

    train_loader, val_loader, dataset = load_train_val_split(
        config, test_size=train_cfg.get("val_split", 0.2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loss = train_model(
        train_loader,
        val_loader,
        len(dataset.vocab),
        dataset.features.shape[1],
        model_cfg | train_cfg,
        device,
        epochs=train_cfg.get("epochs", 5),
        checkpoint_dir=paths_cfg.get("checkpoint_dir", "checkpoints"),
    )
    print(f"Validation loss: {val_loss:.4f}")



if __name__ == "__main__":
    main()
