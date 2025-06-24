import subprocess
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import CardStrengthPredictor


def test_ingest_creates_csv(tmp_path):
    subprocess.run(["python", "src/ingest_cards.py"], check=True)
    assert Path("card_data.csv").is_file()
    df_size = Path("card_data.csv").stat().st_size
    assert df_size > 0


def test_model_forward():
    model = CardStrengthPredictor(vocab_size=10, feature_dim=5)
    text = torch.randint(0, 10, (2, 4))
    lengths = torch.tensor([4, 4])
    feats = torch.randn(2, 5)
    out = model(text, lengths, feats)
    assert out.shape == (2, 1)


