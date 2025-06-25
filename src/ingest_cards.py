# This script ingests only cards from the Tarkir – Dragonstorm (TDS) set.
"""Ingest Magic: The Gathering card data from Scryfall.

The script downloads basic information for each card from the Tarkir –
Dragonstorm set and stores it in ``card_data.csv``. A placeholder
``strength_score`` column is added based on a simple rarity heuristic.
This dataset can later be enhanced with usage or win-rate statistics from
sites like MTGGoldfish.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

SCRYFALL_API_URL = "https://api.scryfall.com/cards/search"
# Ingest only cards from the Tarkir – Dragonstorm set
SET_CODE = "TDM"

# Map rarities to a numeric baseline strength score
RARITY_BASE = {
    "common": 0.2,
    "uncommon": 0.4,
    "rare": 0.7,
    "mythic": 0.9,
}


def fetch_cards() -> list[dict]:
    """Fetch all current standard-legal cards from Scryfall bulk data (oracle_cards)."""
    bulk_url = "https://api.scryfall.com/bulk-data"
    resp = requests.get(bulk_url)
    resp.raise_for_status()
    bulk_data = resp.json()["data"]
    # Use 'oracle_cards' for up-to-date, unique cards
    oracle_entry = next((b for b in bulk_data if b["type"] == "oracle_cards"), None)
    if not oracle_entry:
        raise RuntimeError("Could not find oracle_cards bulk data on Scryfall")
    print(f"Downloading oracle cards from {oracle_entry['download_uri']}")
    cards_resp = requests.get(oracle_entry["download_uri"])
    cards_resp.raise_for_status()
    all_cards = cards_resp.json()
    # Filter for standard-legal cards
    std_cards = [c for c in all_cards if c.get("legalities", {}).get("standard") == "legal"]
    print(f"Fetched {len(std_cards)} standard-legal cards from oracle_cards bulk data.")
    return std_cards


def cards_to_dataframe(cards: List[Dict]) -> pd.DataFrame:
    # Prepare rows for the DataFrame
    rows = []
    for card in cards:
        type_line = card.get("type_line", "")
        # Exclude Basic Land cards
        if "Basic Land" in type_line:
            continue
        rows.append(
            {
                "name": card.get("name"),
                "mana_cost": card.get("cmc"),
                "type_line": type_line,
                "power": card.get("power"),
                "toughness": card.get("toughness"),
                "colors": ",".join(card.get("colors") or []),
                "oracle_text": card.get("oracle_text", ""),
                "rarity": card.get("rarity"),
            }
        )

    columns = [
        "name",
        "mana_cost",
        "type_line",
        "power",
        "toughness",
        "colors",
        "oracle_text",
        "rarity",
    ]
    # Ensure the DataFrame always has the expected columns
    df = pd.DataFrame(rows, columns=columns)
    return df


def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def compute_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    def score_row(row: pd.Series) -> float:
        rarity_bonus = RARITY_BASE.get(str(row["rarity"]).lower(), 0.3)
        power = safe_float(row.get("power"))
        toughness = safe_float(row.get("toughness"))
        cmc = safe_float(row.get("mana_cost")) or 1
        body_efficiency = (power + toughness) / max(cmc, 1)
        return rarity_bonus + body_efficiency

    df["strength_score"] = df.apply(score_row, axis=1)
    return df


def main() -> None:
    print("Fetching card data from Scryfall...")
    # Retrieve all cards from the specified set
    cards = fetch_cards()
    df = cards_to_dataframe(cards)
    df = compute_strength_score(df)
    out_path = Path("card_data.csv")
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved {len(df)} cards to {out_path}")


if __name__ == "__main__":
    main()
