# mtg-neural-network

This project explores using a neural network to evaluate **Magic: The Gathering** cards for Standard deck building. The current prototype predicts card strength from intrinsic card features such as oracle text and mana cost.

## Steps

1. **Define "strong" cards** – cards that provide high intrinsic value based on their rules text and stats.
2. **Collect features** – oracle text, mana cost, card type, power/toughness, etc.
3. **Gather data** – use the [Scryfall](https://scryfall.com/docs/api) API for card details and scrape tournament sites for complementary win-rate data.
4. **Preprocess data** – tokenize oracle text and encode card metadata.
5. **Train a neural network** – an LSTM encoder processes text while structured features are concatenated and fed through dense layers.

See `src/pipeline.py` for a placeholder implementation with mock card data.
`src/ingest_cards.py` demonstrates how to download real card details from
the Scryfall API and save them to `card_data.csv` with a basic
`strength_score` column. Future iterations can merge this data with
tournament win rates.

## Running

```bash
pip install -r requirements.txt
python src/pipeline.py  # train on mock data
python src/ingest_cards.py  # fetch real card data
```

The script will run a tiny training loop on example data.
