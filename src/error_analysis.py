import pandas as pd
from pathlib import Path


def main():
    """Compute and save top prediction errors."""
    try:
        card_df = pd.read_csv("card_data.csv")
        pred_df = pd.read_csv("predictions.csv")
    except FileNotFoundError as e:
        raise SystemExit(f"Missing required file: {e.filename}")

    df = pred_df.merge(card_df[["name", "strength_score"]], left_on="card_name", right_on="name", how="left")
    if df["strength_score"].isna().any():
        df = df.dropna(subset=["strength_score"])

    df["error"] = df["predicted"] - df["strength_score"]
    df["abs_error"] = df["error"].abs()

    over = df.sort_values("error", ascending=False).head(20)
    under = df.sort_values("error", ascending=True).head(20)

    print("Top over-predictions:")
    print(over[["card_name", "strength_score", "predicted", "error"]])

    print("\nTop under-predictions:")
    print(under[["card_name", "strength_score", "predicted", "error"]])

    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    out_df = pd.concat([over.assign(kind="over"), under.assign(kind="under")])
    out_df.to_csv(logs / "top_errors.csv", index=False)


if __name__ == "__main__":
    main()
