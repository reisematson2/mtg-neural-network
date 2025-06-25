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
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save per-card predictions")
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
    # Always show the card name in the error output
    if "card_name" in df.columns:
        name_col = "card_name"
    elif "name" in df.columns:
        name_col = "name"
    else:
        name_col = None

    print("Top 10 prediction errors:")
    for i in top_idx:
        if name_col:
            card_name = df.iloc[i][name_col]
        else:
            card_name = f"index {i}"
        print(f"{card_name}: actual={labels[i]:.3f}, predicted={preds[i]:.3f}, error={abs_err[i]:.3f}")

    # Save per-card predictions for error analysis
    if name_col:
        out_df = df[[name_col, "strength_score"]].copy()
    else:
        out_df = df.copy()
        out_df.insert(0, "row", range(len(out_df)))
        name_col = "row"
    out_df["predicted"] = preds
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
