import argparse
from pathlib import Path
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_loader import load_train_val_split
from src.model import CardStrengthPredictor


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = {}
    if Path(args.config).is_file():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint = args.checkpoint or checkpoint_dir / "best_model.pt"

    # Load the full dataset for evaluation
    loader, _, dataset = load_train_val_split(config, test_size=0)

    state = torch.load(checkpoint, map_location="cpu")

    model = CardStrengthPredictor(
        len(dataset.vocab),
        dataset.features.shape[1],
        config.get("model", {}),
    )
    model.load_state_dict(state)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for text, lengths, feats, labs in loader:
            out = model(feats, text).squeeze(1)
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
    top_idx = np.argsort(-abs_err)[:10]
    df = dataset.df
    # Fix: Only print 'name' if it exists in the DataFrame
    cols = [c for c in ["name", "strength_score"] if c in df.columns]
    print("Top 10 prediction errors:")
    print(df.iloc[top_idx][cols].assign(predicted=preds[top_idx], error=abs_err[top_idx]))


if __name__ == "__main__":
    main()
