import pickle
import torch

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    allow_origins=["http://localhost:5173",
    "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SMSRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "SMS Spam Detection API is running"}


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
