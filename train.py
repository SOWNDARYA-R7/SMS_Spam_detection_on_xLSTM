"""Train the proposed xLSTM SMS spam classifier."""
from __future__ import annotations
import argparse, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import load_data, SMSDataset
from process import ROOT

DEVICE = torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)


def train(epochs=5, batch_size=32, learning_rate=0.001):
    try:
        from model import SpamClassifier
    except Exception as exc:
        raise RuntimeError(
            "xLSTM could not be imported. Activate your Python environment where `import xlstm` works, "
            "then try again. Error: " + str(exc)
        ) from exc

    set_seed(42)
    train_df, test_df, vocab = load_data()
    loader_gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(SMSDataset(train_df, vocab), batch_size=batch_size,
                              shuffle=True, generator=loader_gen)
    model = SpamClassifier(len(vocab.word2idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training messages: {len(train_df)} | Test messages: {len(test_df)}")
    print(f"Vocabulary size: {len(vocab.word2idx):,}")
    print(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")
    history = []
    for epoch in range(epochs):
        model.train(); running_loss = 0.0; correct = 0; seen = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
        avg_loss = running_loss / max(1, seen)
        accuracy = 100.0 * correct / max(1, seen)
        history.append({"Epoch": epoch+1, "Training Loss": avg_loss, "Training Accuracy (%)": accuracy})
        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | accuracy={accuracy:.2f}%")

    out = ROOT / "saved_model"; out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "xlstm_model.pth")
    with open(out / "vocab.pkl", "wb") as f: pickle.dump(vocab, f)
    pd.DataFrame(history).to_csv(ROOT / "results" / "xlstm_training_history.csv", index=False)
    plt.figure(figsize=(7,4)); plt.plot([h["Epoch"] for h in history], [h["Training Loss"] for h in history], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("xLSTM Training Loss"); plt.grid(True); plt.tight_layout()
    plt.savefig(ROOT / "results" / "xlstm_training_loss.png", dpi=300); plt.close()
    plt.figure(figsize=(7,4)); plt.plot([h["Epoch"] for h in history], [h["Training Accuracy (%)"] for h in history], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("xLSTM Training Accuracy"); plt.grid(True); plt.tight_layout()
    plt.savefig(ROOT / "results" / "xlstm_training_accuracy.png", dpi=300); plt.close()
    print("\nTraining complete. Model and vocabulary saved in saved_model/.")
    print("Training curves saved in results/.")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.learning_rate)

if __name__ == "__main__": main()
