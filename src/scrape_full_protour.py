# -*- coding: utf-8 -*-
"""Augment Pro Tour match records with full Standard decks."""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Input match results without decklists
INPUT_PATH = Path("data/tournament_matches.json")
# Output with 60-card lists for each player
OUTPUT_PATH = Path("data/tournament_matches_full.json")

# Base URL for decklist index pages broken up by last-name ranges
INDEX_TEMPLATE = (
    "https://magic.gg/decklists/"
    "pro-tour-magic-the-gathering-final-fantasy-standard-decklists-{}"
)
# Ranges used by Magic.gg
DECKLIST_RANGES = [("a", "c"), ("d", "g"), ("h", "k"), ("l", "n"), ("o", "s"), ("t", "z")]

# Caches to avoid re-downloading pages
_index_cache: dict[str, BeautifulSoup] = {}
_deck_cache: dict[str, list[str] | None] = {}


def range_for_letter(letter: str) -> str:
    """Return the index range key matching ``letter``."""
    letter = letter.lower()
    for start, end in DECKLIST_RANGES:
        if start <= letter <= end:
            return f"{start}-{end}"
    return f"{DECKLIST_RANGES[-1][0]}-{DECKLIST_RANGES[-1][1]}"


def fetch_index(range_key: str) -> BeautifulSoup:
    """Fetch and parse a decklist index page, using the cache when possible."""
    if range_key not in _index_cache:
        url = INDEX_TEMPLATE.format(range_key)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        _index_cache[range_key] = BeautifulSoup(resp.text, "html.parser")
    return _index_cache[range_key]


def parse_mainboard(tag: BeautifulSoup) -> list[str]:
    """Extract the first 60 card names from a ``<main-deck>`` tag."""
    main = tag.find("main-deck")
    if not main:
        return []
    cards: list[str] = []
    for line in main.get_text("\n").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\s+(.+)", line)
        if m:
            count = int(m.group(1))
            card = m.group(2)
            cards.extend([card] * count)
        else:
            cards.append(line)
        if len(cards) >= 60:
            break
    return cards[:60]


def fetch_player_deck(name: str) -> list[str] | None:
    """Return a 60-card list for ``name`` or ``None`` if not found."""
    if name in _deck_cache:
        return _deck_cache[name]

    # Split "Last, First" format
    parts = [p.strip() for p in name.split(",")]
    if len(parts) == 2:
        last, first = parts
    else:  # fallback for already "First Last" inputs
        first = parts[0]
        last = parts[1] if len(parts) > 1 else ""
    canonical = f"{first} {last}".strip()

    range_key = range_for_letter(last[0] if last else canonical[0])
    soup = fetch_index(range_key)

    deck_tag = None
    for dl in soup.find_all("deck-list"):
        title = dl.get("deck-title", "").strip().lower()
        if title == canonical.lower():
            deck_tag = dl
            break

    # If not found in this index page, give up
    if not deck_tag:
        _deck_cache[name] = None
        return None

    # Personal deck URL if available
    link = deck_tag.find("a", href=True)
    deck_html = None
    if link and link["href"].startswith("/"):
        deck_url = f"https://magic.gg{link['href']}"
        try:
            resp = requests.get(deck_url, timeout=10)
            resp.raise_for_status()
            deck_html = resp.text
        except Exception:
            deck_html = None
    
    if deck_html:
        deck_soup = BeautifulSoup(deck_html, "html.parser")
        cards = parse_mainboard(deck_soup)
    else:
        cards = parse_mainboard(deck_tag)

    _deck_cache[name] = cards
    return cards


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"{INPUT_PATH} not found")

    with INPUT_PATH.open() as f:
        matches = json.load(f)

    # Gather all unique player names
    players: set[str] = set()
    for m in matches:
        players.add(m["player_a"])
        players.add(m["player_b"])

    # Download decklists
    for p in tqdm(sorted(players), desc="Players"):
        fetch_player_deck(p)

    # Attach decks to each match
    for match in matches:
        match["player_a_deck"] = _deck_cache.get(match["player_a"])
        match["player_b_deck"] = _deck_cache.get(match["player_b"])

    # Quick verification that every deck has exactly 60 cards
    for rec in matches:
        assert len(rec["player_a_deck"]) == 60
        assert len(rec["player_b_deck"]) == 60

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)

    print(f"Wrote {len(matches)} matches with decks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
