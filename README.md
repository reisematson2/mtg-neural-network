# mtg-neural-network

This project explores using a neural network to evaluate **Magic: The Gathering** cards for Standard deck building. The current prototype predicts card strength from intrinsic card features such as oracle text and mana cost.

## Steps

1. **Define "strong" cards** – cards that provide high intrinsic value based on their rules text and stats.
2. **Collect features** – oracle text, mana cost, card type, power/toughness, etc.
3. **Gather data** – use the [Scryfall](https://scryfall.com/docs/api) API for card details and scrape tournament sites for complementary win-rate data.
4. **Preprocess data** – tokenize oracle text and encode card metadata.
5. **Merge tournament stats** – combine win rates with card features and impute unseen cards.
6. **Train a neural network** – an LSTM encoder processes text while structured features are concatenated and fed through dense layers.

`src/ingest_cards.py` downloads card details from the Scryfall API and saves
them to `card_data.csv` with a placeholder `strength_score`.
`src/compute_win_rates.py` parses `data/tournament_matches_full.json` and generates
`data/card_win_rates.csv`.
`src/merge_and_impute.py` then merges those win-rate statistics, imputes unseen
cards with the mean win rate, and overwrites the `strength_score` column. The
merged data is written back to `card_data.csv`.
`src/train.py` trains the neural network on this dataset and stores the best
model checkpoint in the `checkpoints/` folder. Data loading and model
definitions live in `src/data_loader.py` and `src/model.py` so they can be
reused across scripts. `src/evaluate.py` loads the saved model and reports
prediction metrics on the full dataset.

## Running

```bash
pip install -r requirements.txt
python src/ingest_cards.py          # fetch real card data
python src/scrape_full_protour.py   # build tournament_matches_full.json
python src/compute_win_rates.py     # derive win rates from real matches
python src/merge_and_impute.py      # merge win rates into card_data.csv
python src/train.py                 # train the model using config.yaml
python src/evaluate.py              # evaluate the checkpoint
```
The training script saves its best weights to `checkpoints/best_model.pt` and
`evaluate.py` will report basic metrics against the full dataset. Parameters can
be adjusted in `config.yaml` or overridden on the command line.

## Next Steps

- Expand the win-rate dataset beyond the provided sample to improve training.
- Explore additional card features such as synergies or deck archetypes.
- Tune hyperparameters and perform cross validation for better performance.
