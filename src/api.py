"""FastAPI API for serving card strength predictions."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml

from src.data_loader import CardDataset
from src.model import CardStrengthPredictor

app = FastAPI()


class CardRequest(BaseModel):
    card_name: Optional[str] = None
    oracle_text: str = ""
    mana_value: float = 0.0
    type_line: str = ""
    colors: str = ""
    appearances: float = 0.0
    power: float = 0.0
    toughness: float = 0.0
    rarity: str = "common"


@app.on_event("startup")
def _startup():
    """Load preprocessing artifacts and trained model."""
    global DATASET, MODEL
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}

    DATASET = CardDataset(config)
    ckpt_dir = Path(config.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    state = torch.load(ckpt_dir / "best_model.pt", map_location="cpu")
    MODEL = CardStrengthPredictor(len(DATASET.vocab), DATASET.feature_dim, config.get("model", {}))
    MODEL.load_state_dict(state)
    MODEL.eval()


def _prepare_features(req: CardRequest) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert request data into model-ready tensors."""
    data = req.dict()

    if req.card_name:
        match = DATASET.df[DATASET.df["name"].str.lower() == req.card_name.lower()]
        if not match.empty:
            base = match.iloc[0]
            for key in [
                "oracle_text",
                "mana_value",
                "type_line",
                "colors",
                "appearances",
                "power",
                "toughness",
                "rarity",
            ]:
                if not data.get(key):
                    data[key] = base.get(key, data.get(key))

    row = pd.Series(data)
    text, feats = DATASET.vectorize_row(row)
    return text.unsqueeze(0), feats.unsqueeze(0)


@app.post("/predict")
def predict(card: CardRequest):
    try:
        text, feats = _prepare_features(card)
    except Exception as exc:  # pragma: no cover - robust against bad input
        raise HTTPException(status_code=400, detail=str(exc))

    with torch.no_grad():
        pred = MODEL(feats, text).item()
    return {"strength_score": float(pred)}


if __name__ == "__main__":  # pragma: no cover - manual run
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
