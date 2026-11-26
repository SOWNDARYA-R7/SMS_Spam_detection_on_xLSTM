"""Train and compare xLSTM, LSTM, Bi-LSTM, GRU, TF-IDF+SVM, and XGBoost.
All models use the same fixed stratified 80/20 train/test split from process.py.
Precision/recall/F1 are reported for Spam (label 1).
"""
from __future__ import annotations
import argparse, random, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from dataset import load_data, Vocabulary, SMSDataset
from process import ROOT

SEED=42; DEVICE=torch.device("cpu"); MAX_EPOCHS=5; BATCH_SIZE=32; LR=0.001; EMBED=128; HIDDEN=128

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)

class RecurrentClassifier(nn.Module):
    def __init__(self,vocab_size,kind):
        super().__init__(); self.kind=kind
        self.embedding=nn.Embedding(vocab_size,EMBED,padding_idx=0)
        if kind in ("LSTM","Bi-LSTM"):
            self.rnn=nn.LSTM(EMBED,HIDDEN,batch_first=True,bidirectional=(kind=="Bi-LSTM"))
        else: self.rnn=nn.GRU(EMBED,HIDDEN,batch_first=True)
        self.fc=nn.Linear(HIDDEN*2 if kind=="Bi-LSTM" else HIDDEN,2)
    def forward(self,ids):
        lengths=ids.ne(0).sum(dim=1).clamp(min=1).cpu()
        emb=self.embedding(ids)
        packed=pack_padded_sequence(emb,lengths,batch_first=True,enforce_sorted=False)
        if self.kind in ("LSTM","Bi-LSTM"): _,(hidden,_)=self.rnn(packed)
        else: _,hidden=self.rnn(packed)
        if self.kind=="Bi-LSTM": features=torch.cat((hidden[-2],hidden[-1]),dim=1)
        else: features=hidden[-1]
        return self.fc(features)

def get_metric_row(name,y_true,y_pred):
    return {"Model":name,"Accuracy (%)":round(100*accuracy_score(y_true,y_pred),2),
            "Precision (spam)":round(precision_score(y_true,y_pred,pos_label=1,zero_division=0),4),
            "Recall (spam)":round(recall_score(y_true,y_pred,pos_label=1,zero_division=0),4),
            "F1-score (spam)":round(f1_score(y_true,y_pred,pos_label=1,zero_division=0),4)}

def save_cm(name,y_true,y_pred,out_dir):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1]); safe=re.sub(r"[^A-Za-z0-9]+","_",name).strip("_").lower()
    cdir=out_dir/"confusion_matrices"; cdir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,interpolation="nearest"); fig.colorbar(im,ax=ax)
    ax.set_xticks([0,1],labels=["Ham","Spam"]); ax.set_yticks([0,1],labels=["Ham","Spam"])
    ax.set_title(f"Confusion Matrix - {name}"); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(cdir/f"cm_{safe}.png",dpi=300); plt.close(fig)
    with open(cdir/f"cm_{safe}.txt","w",encoding="utf-8") as f:
        f.write("Rows=true labels [Ham, Spam]; columns=predicted [Ham, Spam]\n")
        f.write(np.array2string(cm)); f.write("\n\n")
        f.write(classification_report(y_true,y_pred,labels=[0,1],target_names=["Ham","Spam"],zero_division=0))
    return cm

def train_neural(name, model, train_df, test_df, vocab, epochs, out_dir):
    seed_all(SEED); gen=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(SMSDataset(train_df,vocab),batch_size=BATCH_SIZE,shuffle=True,generator=gen)
    test_loader=DataLoader(SMSDataset(test_df,vocab),batch_size=BATCH_SIZE,shuffle=False)
    model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR); criterion=nn.CrossEntropyLoss()
    print(f"\n[{name}] training for {epochs} epochs",flush=True)
    for ep in range(epochs):
        model.train(); total=0; loss_sum=0.0
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); logits=model(x); loss=criterion(logits,y); loss.backward(); opt.step()
            loss_sum+=loss.item()*y.size(0); total+=y.size(0)
        print(f" Epoch {ep+1}/{epochs}: loss={loss_sum/max(total,1):.4f}",flush=True)
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in test_loader:
            pred=model(x.to(DEVICE)).argmax(dim=1).cpu().tolist(); y_pred.extend(pred); y_true.extend(y.tolist())
    row=get_metric_row(name,y_true,y_pred); cm=save_cm(name,y_true,y_pred,out_dir)
    if name in ("LSTM","Bi-LSTM","GRU"):
        model_dir=out_dir/"saved_baselines"; model_dir.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),model_dir/(name.lower().replace("-","_") + ".pth"))
    return row,cm

def train_xlstm(train_df,test_df,vocab,epochs,out_dir):
    try: from model import SpamClassifier
    except Exception as exc:
        print(f"\n[xLSTM] NOT RUN: {type(exc).__name__}: {exc}")
        print("To complete this row, run in the Python environment where `import xlstm` works.")
        return None
    return train_neural("xLSTM (proposed)",SpamClassifier(len(vocab.word2idx)),train_df,test_df,vocab,epochs,out_dir)

def compare(epochs=5, output_dir=None):
    seed_all(SEED); train_df,test_df,vocab=load_data()
    out_dir=Path(output_dir or ROOT/"results"/"comparison_run")
    if not out_dir.is_absolute(): out_dir=ROOT/out_dir
    out_dir.mkdir(parents=True,exist_ok=True)
    print("SMS SPAM MODEL COMPARISON")
    print(f"Train={len(train_df)} | Test={len(test_df)} | Vocabulary={len(vocab.word2idx)}")
    print("Split=stratified 80/20, random_state=42 | Ham=0, Spam=1")
    rows=[]; cms={}
    result=train_xlstm(train_df,test_df,vocab,epochs,out_dir)
    if result is not None: row,cm=result; rows.append(row); cms[row["Model"]]=cm
    for kind in ("LSTM","Bi-LSTM","GRU"):
        row,cm=train_neural(kind,RecurrentClassifier(len(vocab.word2idx),kind),train_df,test_df,vocab,epochs,out_dir)
        rows.append(row); cms[row["Model"]]=cm
    y_train=train_df["label"].astype(int).to_numpy(); y_test=test_df["label"].astype(int).to_numpy()
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=50000,sublinear_tf=True,lowercase=False,token_pattern=r"(?u)\b\w+\b")
    xtr=vec.fit_transform(train_df["message"].astype(str)); xte=vec.transform(test_df["message"].astype(str))
    print("\n[TF-IDF + SVM] training",flush=True)
    svm=LinearSVC(random_state=SEED,max_iter=5000); svm.fit(xtr,y_train); pred=svm.predict(xte)
    row=get_metric_row("TF-IDF + SVM",y_test,pred); cm=save_cm(row["Model"],y_test,pred,out_dir); rows.append(row); cms[row["Model"]]=cm
    print("\n[XGBoost] training",flush=True)
    try:
        from xgboost import XGBClassifier
        xgb=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,objective="binary:logistic",eval_metric="logloss",tree_method="hist",random_state=SEED,n_jobs=2)
        xgb.fit(xtr,y_train); pred=xgb.predict(xte); row=get_metric_row("XGBoost",y_test,pred); cm=save_cm(row["Model"],y_test,pred,out_dir); rows.append(row); cms[row["Model"]]=cm
    except ImportError: print("XGBoost unavailable. Install with: python -m pip install xgboost")
    df=pd.DataFrame(rows); df.to_csv(out_dir/"model_metrics.csv",index=False)
    if cms:
        cols=3; nrows=int(np.ceil(len(cms)/cols)); fig,axes=plt.subplots(nrows,cols,figsize=(5*cols,4*nrows)); axes=np.array(axes).reshape(-1)
        for ax,(name,cm) in zip(axes,cms.items()):
            im=ax.imshow(cm,interpolation="nearest"); ax.set_xticks([0,1],labels=["Ham","Spam"]); ax.set_yticks([0,1],labels=["Ham","Spam"]); ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            for i in range(2):
                for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        for ax in axes[len(cms):]: ax.axis("off")
        fig.suptitle("Confusion Matrices - Shared Test Set"); fig.tight_layout(); fig.savefig(out_dir/"all_confusion_matrices.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    print("\nFINAL RESULTS (actual results from this run; spam class is positive)")
    print(df.to_string(index=False))
    print(f"\nSaved metrics: {out_dir/'model_metrics.csv'}")
    print(f"Saved matrices: {out_dir/'confusion_matrices'}")
    if "xLSTM (proposed)" not in set(df.get("Model",[])):
        print("\nWARNING: xLSTM row was not measured in this environment. Do not fill it using an old-dataset result.")
    return df

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--epochs",type=int,default=5); p.add_argument("--output",default=str(ROOT/"results"/"comparison_run")); a=p.parse_args(); compare(a.epochs,a.output)
if __name__=="__main__":
    main()
