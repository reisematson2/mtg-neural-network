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


def fetch_cards() -> List[Dict]:
    """Fetch all cards from the configured set code.

    The function follows pagination via ``next_page`` until the set's
    cards are exhausted.
    """
    url = f"{SCRYFALL_API_URL}?q=set%3A{SET_CODE}"
    cards: List[Dict] = []
    while url:
        resp = requests.get(url)
        # Handle the case where the set code does not exist
        if resp.status_code == 404:
            print(f"Set {SET_CODE} not found on Scryfall")
            break
        resp.raise_for_status()
        data = resp.json()
        cards.extend(data["data"])
        if not data.get("has_more"):
            break
        url = data["next_page"]
        time.sleep(0.1)
    return cards


def cards_to_dataframe(cards: List[Dict]) -> pd.DataFrame:
    # Prepare rows for the DataFrame
    rows = []
    for card in cards:
        rows.append(
            {
                "name": card.get("name"),
                "mana_cost": card.get("cmc"),
                "type_line": card.get("type_line"),
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
