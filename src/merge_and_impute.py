"""Merge tournament win rates with card features and impute missing data."""

import pandas as pd
import numpy as np
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

    # Compute the overall mean win rate weighted by appearances
    total_wins = stats["wins"].sum()
    total_games = stats["appearances"].sum()
    mean_wr = total_wins / total_games if total_games else 0.0

    # Fill missing numeric stats for unseen cards
    merged[["wins", "appearances"]] = merged[["wins", "appearances"]].fillna(0)

    # Wilson lower bound of the win proportion gives a conservative estimate
    # especially for cards with few appearances. This helps avoid overrating
    # cards that happened to win a small sample of games.
    z = 1.96  # 95% confidence interval
    n = merged["appearances"].astype(float)
    phat = merged["wins"].astype(float) / n.replace(0, np.nan)
    wilson = (
        phat + z * z / (2 * n)
        - z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    ) / (1 + z * z / n)

    merged["strength_score"] = wilson.fillna(mean_wr).clip(0, 1)
    merged["log_appearances"] = np.log1p(n)

    # Optional feature for models: flag cards never seen in a tournament
    merged["never_seen_flag"] = (~merged["seen"]).astype(int)

    # Save the updated dataset
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote merged data with {len(merged)} cards to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
