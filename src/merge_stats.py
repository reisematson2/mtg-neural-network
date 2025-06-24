"""Merge real performance statistics into card_data.csv."""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MTGGoldfish stats")
    parser.add_argument("--cards", type=Path, default=Path("card_data.csv"), help="Card CSV produced by ingest_cards.py")
    parser.add_argument("--stats", type=Path, default=Path("stats.csv"), help="CSV with columns name,win_rate,play_rate")
    parser.add_argument("--output", type=Path, default=Path("card_data.csv"), help="Output CSV path")
    return parser.parse_args()


def normalize_win_rate(series: pd.Series) -> pd.Series:
    win = series.fillna(0).astype(float)
    # If values look like percentages (>1), scale down to 0-1
    if win.max() > 1:
        win = win / 100.0
    return win.clip(0, 1)


def merge_stats(cards_path: Path, stats_path: Path, out_path: Path) -> None:
    cards = pd.read_csv(cards_path)
    stats = pd.read_csv(stats_path)
    merged = cards.merge(stats, on="name", how="left")
    merged["win_rate"] = merged.get("win_rate")
    merged["play_rate"] = merged.get("play_rate")
    merged["win_rate"] = merged["win_rate"].fillna(0)
    merged["play_rate"] = merged["play_rate"].fillna(0)
    merged["strength_score"] = normalize_win_rate(merged["win_rate"])
    merged.to_csv(out_path, index=False)
    print(f"Wrote merged data with {len(merged)} cards to {out_path}")


def main() -> None:
    args = parse_args()
    merge_stats(args.cards, args.stats, args.output)


if __name__ == "__main__":
    main()
