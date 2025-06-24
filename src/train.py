import argparse
from pathlib import Path
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader import load_train_val_split, collate_fn
from model import CardStrengthPredictor
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train card strength model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--lstm-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def train_model(train_loader, val_loader, vocab_size, feature_dim, embed_dim, lr, lstm_dim, hidden_dim, device, epochs=5, checkpoint_dir="checkpoints"):
    """Train a model with the given hyperparameters and return final val loss."""
    model = CardStrengthPredictor(vocab_size, feature_dim,
                                  embed_dim=embed_dim,
                                  lstm_dim=lstm_dim,
                                  hidden_dim=hidden_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for text, lengths, feats, labels in train_loader:
            text, lengths = text.to(device), lengths.to(device)
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(text, lengths, feats)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for text, lengths, feats, labels in val_loader:
                text, lengths = text.to(device), lengths.to(device)
                feats, labels = feats.to(device), labels.to(device)
                preds = model(text, lengths, feats)
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

    args.epochs = args.epochs or train_cfg.get("epochs", 5)
    args.batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    args.lr = args.lr or train_cfg.get("learning_rate", 1e-3)
    args.embed_dim = args.embed_dim or model_cfg.get("embed_dim", 32)
    args.lstm_dim = args.lstm_dim or model_cfg.get("lstm_dim", 32)
    args.hidden_dim = args.hidden_dim or model_cfg.get("hidden_dim", 64)
    args.seed = args.seed or train_cfg.get("seed", 42)
    args.checkpoint_dir = args.checkpoint_dir or paths_cfg.get("checkpoint_dir", "checkpoints")
    data_path = paths_cfg.get("data", "card_data.csv")

    torch.manual_seed(args.seed)
    train_ds, val_ds = load_train_val_split(path=data_path, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lrs = [1e-3, 1e-4]
    embeds = [64, 128]

    results_path = Path("sweep_results.csv")
    with results_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "embed_dim", "val_loss"])
        for lr in lrs:
            for embed in embeds:
                val_loss = train_model(
                    train_loader,
                    val_loader,
                    len(train_ds.vocab),
                    train_ds.feature_dim,
                    embed,
                    lr,
                    args.lstm_dim,
                    args.hidden_dim,
                    device,
                    epochs=5,
                    checkpoint_dir=args.checkpoint_dir,
                )
                writer.writerow([lr, embed, f"{val_loss:.4f}"])
                print(f"lr={lr} embed_dim={embed} val_loss={val_loss:.4f}")



if __name__ == "__main__":
    main()
