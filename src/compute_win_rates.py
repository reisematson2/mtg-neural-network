import json
from collections import defaultdict
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/test_matches.json")
OUTPUT_PATH = Path("data/card_win_rates.csv")


def main() -> None:
    """Compute per-card win rates from a list of match records."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"{INPUT_PATH} not found")

    with INPUT_PATH.open() as f:
        matches = json.load(f)

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"appearances": 0, "wins": 0})

    for match in matches:
        deck_a = match.get("player_a_deck") or []
        deck_b = match.get("player_b_deck") or []
        winner = match.get("winner")
        player_a = match.get("player_a")
        player_b = match.get("player_b")

        for card in deck_a:
            stats[card]["appearances"] += 1
            if winner == player_a:
                stats[card]["wins"] += 1

        for card in deck_b:
            stats[card]["appearances"] += 1
            if winner == player_b:
                stats[card]["wins"] += 1

    records = []
    for card, s in stats.items():
        appearances = s["appearances"]
        wins = s["wins"]
        win_rate = wins / appearances if appearances else 0.0
        records.append(
            {
                "card_name": card,
                "appearances": appearances,
                "wins": wins,
                "win_rate": win_rate,
            }
        )

    df = pd.DataFrame(records)
    df.sort_values("card_name", inplace=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved win rates for {len(df)} cards to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
