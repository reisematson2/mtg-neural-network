import pandas as pd
from pathlib import Path

CARD_PATH = Path('card_data.csv')
PRED_PATH = Path('predictions.csv')
LOG_PATH = Path('logs/top_errors.csv')


def main() -> None:
    if not CARD_PATH.is_file():
        raise FileNotFoundError(f"{CARD_PATH} not found")
    if not PRED_PATH.is_file():
        raise FileNotFoundError(f"{PRED_PATH} not found. Run evaluate.py first")

    cards = pd.read_csv(CARD_PATH)
    preds = pd.read_csv(PRED_PATH)

    name_col = 'card_name' if 'card_name' in cards.columns else 'name'
    if name_col not in preds.columns:
        preds_name_col = 'card_name' if 'card_name' in preds.columns else 'name'
        preds = preds.rename(columns={preds_name_col: name_col})

    merged = cards[[name_col, 'strength_score']].merge(preds[[name_col, 'predicted']], on=name_col, how='inner')
    merged['error'] = merged['predicted'] - merged['strength_score']
    merged['abs_error'] = merged['error'].abs()

    over = merged[merged['error'] > 0].sort_values('abs_error', ascending=False).head(20)
    under = merged[merged['error'] < 0].sort_values('abs_error', ascending=False).head(20)

    print('Top over-predictions:')
    for _, row in over.iterrows():
        print(f"{row[name_col]}: actual={row['strength_score']:.3f} predicted={row['predicted']:.3f} error={row['error']:.3f}")

    print('\nTop under-predictions:')
    for _, row in under.iterrows():
        print(f"{row[name_col]}: actual={row['strength_score']:.3f} predicted={row['predicted']:.3f} error={row['error']:.3f}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat({'over': over, 'under': under}).to_csv(LOG_PATH)
    print(f"Saved error details to {LOG_PATH}")


if __name__ == '__main__':
    main()
