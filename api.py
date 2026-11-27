import pickle
import torch

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel

from model import SpamClassifier
from text_utils import clean_text, ROOT, MAX_LEN

DEVICE = torch.device("cpu")

model = None
vocab = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocab

    model_path = ROOT / "saved_model" / "xlstm_model.pth"
    vocab_path = ROOT / "saved_model" / "vocab.pkl"

    print("Model path:", model_path)
    print("Vocab path:", vocab_path)

    print("Model exists:", model_path.exists())
    print("Vocab exists:", vocab_path.exists())

    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        print("Vocabulary loaded successfully!")
        print("Vocabulary size:", len(vocab.word2idx))

        model = SpamClassifier(len(vocab.word2idx)).to(DEVICE)

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(checkpoint)
        model.eval()

        print("xLSTM model loaded successfully!")

    except Exception as e:
        print("MODEL LOADING ERROR:", repr(e))
        raise

    yield

    model = None
    vocab = None


app = FastAPI(
    title="SMS Spam Detection API",
    description="SMS classification using a trained xLSTM model",
    version="1.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SMSRequest(BaseModel):
    message: str


class BatchSMSRequest(BaseModel):
    messages: List[str]


@app.get("/")
def home():
    return {"message": "SMS Spam Detection API is running"}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "vocab_size": len(vocab.word2idx) if vocab is not None else 0,
        "device": str(DEVICE),
        "architecture": "xLSTM (3x mLSTM + 1x sLSTM, 4 Heads)",
        "embedding_dim": 128,
        "max_seq_len": MAX_LEN
    }


@app.get("/api/metrics")
def get_metrics():
    return {
        "model": "xLSTM (proposed)",
        "accuracy": 94.70,
        "precision": 0.7419,
        "recall": 0.9262,
        "f1_score": 0.8239,
        "confusion_matrix": {
            "true_negative": 917,
            "false_positive": 48,
            "false_negative": 11,
            "true_positive": 138,
            "total_samples": 1114
        },
        "baselines": [
            {"model": "TF-IDF + SVM", "accuracy": 98.38, "precision": 0.9712, "recall": 0.9060, "f1_score": 0.9375},
            {"model": "LSTM", "accuracy": 97.85, "precision": 0.9771, "recall": 0.8591, "f1_score": 0.9143},
            {"model": "GRU", "accuracy": 97.67, "precision": 0.9767, "recall": 0.8456, "f1_score": 0.9065},
            {"model": "Bi-LSTM", "accuracy": 97.31, "precision": 0.9542, "recall": 0.8389, "f1_score": 0.8929},
            {"model": "XGBoost", "accuracy": 96.23, "precision": 1.0000, "recall": 0.7181, "f1_score": 0.8359},
            {"model": "xLSTM (Proposed)", "accuracy": 94.70, "precision": 0.7419, "recall": 0.9262, "f1_score": 0.8239}
        ]
    }


@app.post("/predict")
def predict_sms(data: SMSRequest):

    if model is None or vocab is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter an SMS message"
        )

    # Apply the same preprocessing used during training
    cleaned_message = clean_text(message)

    ids = [
        vocab.word2idx.get(word, 1)
        for word in cleaned_message.split()
    ]

    ids = (ids + [0] * MAX_LEN)[:MAX_LEN]

    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    prediction = int(torch.argmax(probs).item())

    return {
        "prediction": "SPAM" if prediction == 1 else "HAM",
        "confidence": round(probs[prediction].item() * 100, 2),
        "ham_probability": round(probs[0].item() * 100, 2),
        "spam_probability": round(probs[1].item() * 100, 2)
    }


@app.post("/api/predict/batch")
def predict_batch(data: BatchSMSRequest):
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    results = []
    batch_tensors = []
    valid_indices = []

    for idx, raw_msg in enumerate(data.messages):
        clean_msg = clean_text(raw_msg.strip())
        if not clean_msg:
            continue
        ids = [vocab.word2idx.get(word, 1) for word in clean_msg.split()]
        ids = (ids + [0] * MAX_LEN)[:MAX_LEN]
        batch_tensors.append(ids)
        valid_indices.append((idx, raw_msg))

    if not batch_tensors:
        return {"results": [], "total": 0}

    x = torch.tensor(batch_tensors, dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    for i, (orig_idx, orig_msg) in enumerate(valid_indices):
        p = probs[i]
        pred = int(torch.argmax(p).item())
        results.append({
            "index": orig_idx,
            "message": orig_msg,
            "prediction": "SPAM" if pred == 1 else "HAM",
            "confidence": round(p[pred].item() * 100, 2),
            "ham_probability": round(p[0].item() * 100, 2),
            "spam_probability": round(p[1].item() * 100, 2)
        })

    return {
        "results": results,
        "total": len(results),
        "spam_count": sum(1 for r in results if r["prediction"] == "SPAM"),
        "ham_count": sum(1 for r in results if r["prediction"] == "HAM")
    }
