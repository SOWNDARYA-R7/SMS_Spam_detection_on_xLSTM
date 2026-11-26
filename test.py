"""Evaluate the saved xLSTM model and create a confusion matrix."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset import load_data, SMSDataset
from process import ROOT

DEVICE = torch.device("cpu")

def evaluate():
    try:
        from model import SpamClassifier
    except Exception as exc:
        raise RuntimeError("xLSTM import failed. Activate the environment where `import xlstm` works. " + str(exc)) from exc
    model_path = ROOT / "saved_model" / "xlstm_model.pth"
    vocab_path = ROOT / "saved_model" / "vocab.pkl"
    if not model_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Model files missing. Run `python main.py train` first.")
    train_df, test_df, fresh_vocab = load_data()
    with open(vocab_path, "rb") as f: vocab = pickle.load(f)
    if len(vocab.word2idx) != len(fresh_vocab.word2idx):
        print("Warning: saved vocabulary size differs from the currently processed split. Retrain the model to avoid mismatch.")
    loader = DataLoader(SMSDataset(test_df, vocab), batch_size=32, shuffle=False)
    model = SpamClassifier(len(vocab.word2idx)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(DEVICE)); y_pred.extend(logits.argmax(dim=1).cpu().tolist()); y_true.extend(y.tolist())
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,pos_label=1,zero_division=0)
    recall=recall_score(y_true,y_pred,pos_label=1,zero_division=0)
    f1=f1_score(y_true,y_pred,pos_label=1,zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    print("\nTEST RESULTS (Spam is the positive class)")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:\n",classification_report(y_true,y_pred,labels=[0,1],target_names=["Ham","Spam"],zero_division=0))
    print("Confusion matrix (rows=true [Ham, Spam], columns=predicted [Ham, Spam]):\n",cm)
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    pd.DataFrame([{"Model":"xLSTM (proposed)","Accuracy (%)":round(accuracy*100,2),"Precision (spam)":round(precision,4),"Recall (spam)":round(recall,4),"F1-score (spam)":round(f1,4)}]).to_csv(out/"xlstm_metrics.csv",index=False)
    plt.figure(figsize=(5,4)); plt.imshow(cm, interpolation="nearest"); plt.title("xLSTM Confusion Matrix"); plt.colorbar()
    plt.xticks([0,1],["Ham","Spam"]); plt.yticks([0,1],["Ham","Spam"])
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    for i in range(2):
        for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(out/"xlstm_confusion_matrix.png",dpi=300); plt.close()
    print(f"Saved results to {out}")
    return accuracy,precision,recall,f1,cm

if __name__ == "__main__": evaluate()
