import argparse
from pathlib import Path
import yaml

import torch
from torch import nn
import matplotlib.pyplot as plt  # For plotting loss curves

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

    # Initialize lists to store loss values for each epoch
    train_losses = []
    val_losses = []

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
        val_losses_epoch = []
        with torch.no_grad():
            for text, lengths, feats, labels in val_loader:
                text = text.to(device)
                feats, labels = feats.to(device), labels.to(device)
                preds = model(feats, text)
                loss = loss_fn(preds, labels)
                val_losses_epoch.append(loss.item())
        val_loss = sum(val_losses_epoch) / len(val_losses_epoch)

        # After computing train_loss and val_loss each epoch, append to lists
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    return val_loss, train_losses, val_losses


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

    # Diagnostic: Check for NaNs/Infs in features and targets
    feats = dataset.features
    targets = dataset.targets
    print(f"Feature shape: {feats.shape}, Target shape: {targets.shape}")
    print(f"Any NaN in features? {torch.isnan(feats).any().item()}")
    print(f"Any Inf in features? {torch.isinf(feats).any().item()}")
    print(f"Any NaN in targets? {torch.isnan(targets).any().item()}")
    print(f"Any Inf in targets? {torch.isinf(targets).any().item()}")

    val_loss, train_losses, val_losses = train_model(
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

    # After training, print the full loss curves
    print("Final train losses:", train_losses)
    print("Final val losses:  ", val_losses)

    # Plot and save the loss curves using matplotlib
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")


if __name__ == "__main__":
    main()
