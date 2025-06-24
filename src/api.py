"""FastAPI endpoint for predicting card strength."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import CardDataset, collate_fn
from model import CardStrengthPredictor

app = FastAPI()

DATA_PATH = Path("card_data.csv")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

# Load preprocessing artifacts and model on startup
if DATA_PATH.is_file() and CHECKPOINT_PATH.is_file():
    base_df = pd.read_csv(DATA_PATH)
    base_ds = CardDataset(base_df)
    vocab = base_ds.vocab
    cat_maps = base_ds.cat_maps
    scaler = base_ds.scaler
    model = CardStrengthPredictor(len(vocab), base_ds.feature_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
else:
    raise RuntimeError("Required data/model files not found")


class CardRequest(BaseModel):
    name: str | None = None
    mana_cost: float | None = 0
    type_line: str | None = "Creature"
    power: float | None = 0
    toughness: float | None = 0
    colors: str | None = ""
    oracle_text: str | None = ""
    rarity: str | None = "common"


@app.post("/predict")
def predict(card: CardRequest):
    df = pd.DataFrame([card.dict()])
    ds = CardDataset(df, vocab=vocab, cat_maps=cat_maps, scaler=scaler)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    text, lengths, feats, _ = next(iter(loader))
    with torch.no_grad():
        pred = model(text, lengths, feats).item()
    return {"strength_score": float(pred)}

