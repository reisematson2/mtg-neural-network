"""Scrape Pro Tour match results from Magic.gg.

The script fetches each round's results page, parses the match table, and writes
all matches to ``data/tournament_matches.json``. Basic error handling is used so
missing or malformed pages don't stop the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TOURNAMENT_SLUG = "pro-tour-magic-the-gathering-final-fantasy"
NUM_ROUNDS = 16
BASE_URL = f"https://magic.gg/news/{TOURNAMENT_SLUG}-round-{{round}}-results"
OUTPUT_PATH = Path("data/tournament_matches.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_round(round_num: int) -> list[dict]:
    """Fetch and parse match results for a single round."""
    url = BASE_URL.format(round=round_num)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            logging.warning("Round %s not found (%s)", round_num, url)
            return []
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network
        logging.warning("Failed to fetch %s: %s", url, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        logging.warning("No results table found for round %s", round_num)
        return []

    matches: list[dict] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        player_a = cells[0].get_text(strip=True)
        player_b = cells[2].get_text(strip=True)
        result_text = cells[3].get_text(strip=True)
        if not player_a or not player_b:
            continue

        winner = None
        if player_a in result_text:
            winner = player_a
        elif player_b in result_text:
            winner = player_b

        links = row.find_all("a")
        decklist_a_url = links[0].get("href") if len(links) >= 1 else None
        decklist_b_url = links[1].get("href") if len(links) >= 2 else None

        record = {
            "round": round_num,
            "player_a": player_a,
            "player_b": player_b,
            "winner": winner,
        }
        if decklist_a_url:
            record["decklist_a_url"] = decklist_a_url
        if decklist_b_url:
            record["decklist_b_url"] = decklist_b_url

        matches.append(record)
    return matches


def main() -> None:
    """Scrape results for all rounds and write them to JSON."""
    all_matches: list[dict] = []
    for r in tqdm(range(1, NUM_ROUNDS + 1), desc="Rounds"):
        all_matches.extend(fetch_round(r))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2)

    print(f"Saved {len(all_matches)} matches to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
