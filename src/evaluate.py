import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_loader import CardDataset, collate_fn, load_train_val_split
from model import CardStrengthPredictor


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = {}
    if Path(args.config).is_file():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    data_path = config.get("paths", {}).get("data", "card_data.csv")
    checkpoint = args.checkpoint or Path(config.get("paths", {}).get("checkpoint_dir", "checkpoints")) / "best_model.pt"

    # Build preprocessing using the original training split
    train_ds, _ = load_train_val_split(path=data_path)
    full_df = pd.read_csv(data_path)
    dataset = CardDataset(full_df, vocab=train_ds.vocab, cat_maps=train_ds.cat_maps, scaler=train_ds.scaler)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = CardStrengthPredictor(len(train_ds.vocab), dataset.feature_dim)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
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

    abs_err = np.abs(preds - labels)
    top_idx = np.argsort(abs_err)[-10:][::-1]
    df = dataset.df
    print("Top 10 prediction errors:")
    print(df.iloc[top_idx][["name", "strength_score"]].assign(predicted=preds[top_idx], error=abs_err[top_idx]))


if __name__ == "__main__":
    main()
