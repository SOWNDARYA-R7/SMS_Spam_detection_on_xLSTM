
"""Vocabulary, dataset class, and access to the shared train/test split."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from text_utils import clean_text, ROOT, MAX_LEN


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(clean_text(text).split())

        for word in counter:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        ids = [
            self.word2idx.get(word, 1)
            for word in clean_text(text).split()
        ]

        return (ids + [0] * MAX_LEN)[:MAX_LEN]


class SMSDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.data = dataframe.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = self.vocab.encode(row["message"])
        y = int(row["label"])

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


def load_data(refresh=False, data_path=None, processed_dir=None):
    """Return train_df, test_df, vocabulary from the same saved split."""

    # Import training dependencies only when this function is called.
    import pandas as pd

    from process import (
        DEFAULT_DATA,
        DEFAULT_OUT,
        process_dataset
    )

    processed_dir = Path(processed_dir or DEFAULT_OUT)

    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if refresh or not (train_path.exists() and test_path.exists()):
        process_dataset(data_path or DEFAULT_DATA, processed_dir)

    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")

    for df in (train_df, test_df):
        df["label"] = df["label"].astype(int)
        df["message"] = df["message"].fillna("").astype(str)

    vocab = Vocabulary()
    vocab.build(train_df["message"])

    return train_df, test_df, vocab
