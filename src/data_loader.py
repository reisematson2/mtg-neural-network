import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset
import torch


def tokenize(text: str) -> List[str]:
    return text.lower().replace(',', ' ').replace('.', ' ').split()


def build_vocab(texts: List[str]) -> Dict[str, int]:
    tokens = sorted({tok for text in texts for tok in tokenize(text)})
    return {tok: idx + 1 for idx, tok in enumerate(tokens)}  # reserve 0 for PAD


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, 0) for tok in tokenize(text)]


class CardDataset(Dataset):
    """PyTorch dataset for MTG card strength prediction."""

    def __init__(self, df: pd.DataFrame, vocab: Optional[Dict[str, int]] = None,
                 cat_maps: Optional[Dict[str, Dict[str, int]]] = None,
                 scaler: Optional[Dict[str, pd.Series]] = None):
        self.df = df.reset_index(drop=True)
        self.texts = self.df.get('oracle_text', '').fillna('').astype(str).tolist()

        # Build vocab from provided texts if not given
        self.vocab = vocab or build_vocab(self.texts)
        tokens = [encode_text(t, self.vocab) for t in self.texts]
        self.tokenized = [tok if len(tok) > 0 else [0] for tok in tokens]

        # Numeric columns
        num_cols = ['mana_cost', 'power', 'toughness']
        self.df[num_cols] = self.df[num_cols].fillna(0).astype(float)
        if scaler is None:
            means = self.df[num_cols].mean()
            stds = self.df[num_cols].std().replace(0, 1)
            self.scaler = {'mean': means, 'std': stds}
        else:
            self.scaler = scaler
        self.df[num_cols] = (self.df[num_cols] - self.scaler['mean']) / self.scaler['std']
        self.numeric = self.df[num_cols].values.astype('float32')

        # Categorical columns
        cat_cols = ['rarity', 'type_line']
        self.df[cat_cols] = self.df[cat_cols].fillna('unknown').astype(str)
        if cat_maps is None:
            self.cat_maps = {
                col: {v: i for i, v in enumerate(sorted(self.df[col].unique()))}
                for col in cat_cols
            }
        else:
            self.cat_maps = cat_maps
        cat_feats = []
        for col in cat_cols:
            mapping = self.cat_maps[col]
            idxs = self.df[col].map(lambda x: mapping.get(x, 0)).astype(int).values
            one_hot = np.eye(len(mapping))[idxs]
            cat_feats.append(one_hot)
        self.categorical = np.concatenate(cat_feats, axis=1).astype('float32')

        self.labels = self.df['strength_score'].fillna(0).astype('float32').values
        self.feature_dim = self.numeric.shape[1] + self.categorical.shape[1]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        text_ids = self.tokenized[idx]
        struct = np.concatenate([self.numeric[idx], self.categorical[idx]])
        label = self.labels[idx]
        return text_ids, struct, label


def collate_fn(batch):
    texts, structs, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    padded_texts = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in texts], batch_first=True
    )
    struct_tensor = torch.tensor(structs, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return padded_texts, lengths, struct_tensor, label_tensor


def load_train_val_split(test_size: float = 0.2, seed: int = 42) -> Tuple[CardDataset, CardDataset]:
    """Load ``card_data.csv`` and return train/validation datasets."""
    path = Path('card_data.csv')
    df = pd.read_csv(path)
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    split = int(len(df) * (1 - test_size))
    train_idx, val_idx = indices[:split], indices[split:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_ds = CardDataset(train_df)
    val_ds = CardDataset(val_df, vocab=train_ds.vocab, cat_maps=train_ds.cat_maps, scaler=train_ds.scaler)
    return train_ds, val_ds
