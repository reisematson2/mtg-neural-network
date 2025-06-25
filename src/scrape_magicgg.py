"""Scrape match results and decklists from Magic.gg.

The script downloads match results for multiple rounds and Standard decklists
from Magic.gg, then merges them together so each match record includes the two
players' mainboard lists. The aggregated records are written to JSON.

Network requests may fail or pages may not have the expected structure, so the
implementation includes basic error handling and liberal parsing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Return a lowercased "last, first" format for matching."""
    name = name.strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{last.lower()}, {first.lower()}"
    parts = name.split()
    if len(parts) > 1:
        return f"{parts[-1].lower()}, {' '.join(parts[:-1]).lower()}"
    return name.lower()

def fetch_html(url: str) -> str | None:
    """Return the HTML contents of ``url`` or ``None`` on failure."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # pragma: no cover - network errors
        print(f"Failed to fetch {url}: {exc}")
        return None


def parse_round_from_url(url: str) -> str:
    """Try to infer the round label from the URL."""
    m = re.search(r"round-(\d+)", url)
    return m.group(1) if m else url


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_result_page(html: str, url: str) -> List[Dict]:
    """Parse match results from a single HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    round_label = parse_round_from_url(url)

    table = soup.find("table")
    if not table:
        print(f"No match table found on {url}")
        return []

    matches = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        player_a = cells[0].get_text(strip=True)
        player_b = cells[2].get_text(strip=True)
        result_cell = cells[3].get_text(strip=True)
        if not player_a or not player_b:
            continue

        winner = None
        m = re.search(r"(.+)\s+won", result_cell)
        if m:
            winner = m.group(1).strip()
        elif "draw" in result_cell.lower():
            winner = None
        matches.append({
            "round": round_label,
            "player_a": player_a,
            "player_b": player_b,
            "winner": winner,
        })
    return matches


# ---------------------------------------------------------------------------
# Decklist parsing
# ---------------------------------------------------------------------------

def parse_decklist_index(html: str, base_url: str) -> Dict[str, List[str]]:
    """Return a mapping of player name to mainboard list from a decklist index."""
    soup = BeautifulSoup(html, "html.parser")
    decklists: Dict[str, List[str]] = {}

    for dl in soup.find_all("deck-list"):
        name = dl.get("deck-title")
        main = dl.find("main-deck")
        if not name or not main:
            continue
        cards: List[str] = []
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
        decklists[normalize_name(name)] = cards[:60]

    return decklists




# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape Magic.gg tournament data")
    parser.add_argument("--result-urls", required=True, help="Comma-separated result URLs")
    parser.add_argument("--decklist-urls", required=True, help="Comma-separated decklist index URLs")
    parser.add_argument("--output", type=str, default="data/tournament_matches.json")
    args = parser.parse_args(argv)

    result_urls = [u.strip() for u in args.result_urls.split(",") if u.strip()]
    decklist_urls = [u.strip() for u in args.decklist_urls.split(",") if u.strip()]

    all_matches: List[Dict] = []
    for url in result_urls:
        html = fetch_html(url)
        if not html:
            continue
        all_matches.extend(parse_result_page(html, url))

    decklists: Dict[str, List[str]] = {}
    for index_url in decklist_urls:
        html = fetch_html(index_url)
        if not html:
            continue
        decklists.update(parse_decklist_index(html, index_url))

    for match in all_matches:
        a_key = normalize_name(match["player_a"])
        b_key = normalize_name(match["player_b"])
        match["player_a_deck"] = decklists.get(a_key)
        if not match["player_a_deck"]:
            print(f"Decklist not found for {match['player_a']}")
        match["player_b_deck"] = decklists.get(b_key)
        if not match["player_b_deck"]:
            print(f"Decklist not found for {match['player_b']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2)
    print(f"Saved {len(all_matches)} matches to {out_path}")


if __name__ == "__main__":
    main()
