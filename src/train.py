import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader import load_train_val_split, collate_fn
from model import CardStrengthPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train card strength model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lstm-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds, val_ds = load_train_val_split(seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    model = CardStrengthPredictor(len(train_ds.vocab), train_ds.feature_dim,
                                  embed_dim=args.embed_dim,
                                  lstm_dim=args.lstm_dim,
                                  hidden_dim=args.hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for text, lengths, feats, labels in train_loader:
            text, lengths = text.to(device), lengths.to(device)
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(text, lengths, feats)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for text, lengths, feats, labels in val_loader:
                text, lengths = text.to(device), lengths.to(device)
                feats, labels = feats.to(device), labels.to(device)
                preds = model(text, lengths, feats)
                loss = loss_fn(preds, labels)
                val_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    print(f"Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
