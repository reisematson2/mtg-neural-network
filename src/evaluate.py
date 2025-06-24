import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import CardDataset, collate_fn, load_train_val_split
from model import CardStrengthPredictor


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    # Build preprocessing using the original training split
    train_ds, _ = load_train_val_split()
    full_df = pd.read_csv("card_data.csv")
    dataset = CardDataset(full_df, vocab=train_ds.vocab, cat_maps=train_ds.cat_maps, scaler=train_ds.scaler)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = CardStrengthPredictor(len(train_ds.vocab), dataset.feature_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for text, lengths, feats, labs in loader:
            out = model(text, lengths, feats).squeeze(1)
            preds.extend(out.numpy())
            labels.extend(labs.squeeze(1).numpy())
    preds = np.array(preds)
    labels = np.array(labels)

    corr = float(np.corrcoef(labels, preds)[0, 1])
    mae = float(np.mean(np.abs(labels - preds)))
    print(f"Pearson correlation: {corr:.4f}")
    print(f"MAE: {mae:.4f}")

    plt.scatter(labels, preds, alpha=0.5)
    plt.xlabel("Actual Strength")
    plt.ylabel("Predicted Strength")
    plt.savefig("pred_vs_actual.png")

    diff = preds - labels
    order = np.argsort(diff)
    over_idx = order[-10:][::-1]
    under_idx = order[:10]
    df = dataset.df
    print("Top 10 over-performing predictions:")
    print(df.iloc[over_idx][["name", "strength_score"]])
    print("Top 10 under-performing predictions:")
    print(df.iloc[under_idx][["name", "strength_score"]])


if __name__ == "__main__":
    main()
