"""FastAPI service for predicting MTG card strength."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Path as FastAPIPath
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import torch

from src.data_loader import CardDataset
from src.model import CardStrengthPredictor

CONFIG_PATH = Path("config.yaml")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

app = FastAPI()

dataset: CardDataset | None = None
model: CardStrengthPredictor | None = None


class CardRequest(BaseModel):
    card_name: str | None = None
    oracle_text: str = ""
    mana_value: float = 0
    type_line: str = ""
    colors: str = ""
    rarity: str = "common"
    power: float = 0
    toughness: float = 0
    appearances: float = 0


@app.on_event("startup")
def load_artifacts() -> None:
    """Load dataset preprocessing artifacts and model weights."""
    global dataset, model
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError("Model checkpoint not found")
    config = {}
    if CONFIG_PATH.is_file():
        import yaml

        with CONFIG_PATH.open() as f:
            config = yaml.safe_load(f) or {}
    dataset = CardDataset(config)
    model = CardStrengthPredictor(len(dataset.vocab), dataset.feature_dim, config.get("model", {}))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()


@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>MTG Neural Network API is running.</h2><p>Use the <code>/predict</code> endpoint to get card strength predictions.</p>"


@app.api_route("/predict", methods=["GET", "POST"])
def predict(card: CardRequest) -> dict[str, Any]:
    if dataset is None or model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    try:
        row = pd.Series(card.dict())
        tokens, feats = dataset.featurize_row(row)
        text_batch = torch.nn.utils.rnn.pad_sequence([tokens], batch_first=True)
        feat_batch = feats.unsqueeze(0)
        with torch.no_grad():
            pred = model(feat_batch, text_batch).item()
    except Exception as exc:  # pragma: no cover - runtime error reporting
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"strength_score": float(pred)}


@app.get("/card/{card_name}")
def predict_by_name(card_name: str):
    if dataset is None or model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    # Try to find the card in the loaded dataset (case-insensitive match)
    df = dataset.df
    mask = df["card_name"].str.lower() == card_name.lower()
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Card '{card_name}' not found in dataset")
    row = df[mask].iloc[0]
    try:
        tokens, feats = dataset.featurize_row(row)
        text_batch = torch.nn.utils.rnn.pad_sequence([tokens], batch_first=True)
        feat_batch = feats.unsqueeze(0)
        with torch.no_grad():
            pred = model(feat_batch, text_batch).item()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"card_name": row["card_name"], "strength_score": float(pred)}


if __name__ == "__main__":  # pragma: no cover - manual launch
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
