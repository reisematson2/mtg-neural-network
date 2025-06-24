# mtg-neural-network

This project explores using a neural network to evaluate **Magic: The Gathering** cards for Standard deck building. The current prototype predicts card strength from intrinsic card features such as oracle text and mana cost.

## Steps

1. **Define "strong" cards** – cards that provide high intrinsic value based on their rules text and stats.
2. **Collect features** – oracle text, mana cost, card type, power/toughness, etc.
3. **Gather data** – use the [Scryfall](https://scryfall.com/docs/api) API for card details and scrape tournament sites for complementary win-rate data.
4. **Preprocess data** – tokenize oracle text and encode card metadata.
5. **Train a neural network** – an LSTM encoder processes text while structured features are concatenated and fed through dense layers.

`src/ingest_cards.py` demonstrates how to download real card details from
the Scryfall API and save them to `card_data.csv` with a basic
`strength_score` column. `src/train.py` trains the neural network on this
dataset and stores the best model checkpoint in the `checkpoints/` folder.
`src/evaluate.py` loads the saved model and reports prediction metrics on the
full dataset. Future iterations can merge this data with tournament win
rates.

## Running

```bash
pip install -r requirements.txt
python src/ingest_cards.py          # fetch real card data
python src/train.py                 # train the model using config.yaml
python src/evaluate.py              # evaluate the checkpoint
```
The training script saves its best weights to `checkpoints/best_model.pt` and
`evaluate.py` will report basic metrics against the full dataset. Parameters can
be adjusted in `config.yaml` or overridden on the command line.
