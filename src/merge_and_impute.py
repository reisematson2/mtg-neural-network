"""Merge tournament win rates with card features and impute missing data."""

import pandas as pd
from pathlib import Path


CARD_PATH = Path("card_data.csv")
STATS_PATH = Path("data/card_win_rates.csv")
OUTPUT_PATH = CARD_PATH


def main() -> None:
    """Load card data and stats, merge them, and fill in missing values."""
    # Load the Scryfall features with placeholder strength_score
    cards = pd.read_csv(CARD_PATH)

    # card_data.csv produced by ingest_cards.py uses the column 'name'.
    # Rename it to 'card_name' for consistency with the win-rate stats.
    if "card_name" not in cards.columns and "name" in cards.columns:
        cards = cards.rename(columns={"name": "card_name"})

    # Load real tournament statistics with columns card_name, appearances, wins, win_rate
    if not STATS_PATH.is_file():
        raise FileNotFoundError(
            f"{STATS_PATH} not found. Run compute_win_rates.py first"
        )
    stats = pd.read_csv(STATS_PATH)

    # Perform a left join so that all TDM cards remain even if unseen in tournaments
    merged = cards.merge(stats, on="card_name", how="left")

    # Determine whether each card was ever seen in the tournament stats
    merged["seen"] = merged["appearances"].notna()

    # Compute the global mean win rate from cards that have data
    mean_wr = stats["win_rate"].mean()

    # Impute unseen cards with the mean win rate
    merged.loc[~merged["seen"], "win_rate"] = mean_wr

    # Normalize win_rate to the 0-1 range and store it as strength_score
    win = merged["win_rate"].astype(float)
    if win.max() > 1:
        win = win / 100.0
    merged["strength_score"] = win.clip(0, 1)

    # Optional feature for models: flag cards never seen in a tournament
    merged["never_seen_flag"] = (~merged["seen"]).astype(int)

    # Save the updated dataset
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote merged data with {len(merged)} cards to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
