"""Predict Ham/Spam for a new SMS using the trained xLSTM model."""
import pickle
import torch
from dataset import MAX_LEN
from process import clean_text, ROOT
DEVICE = torch.device("cpu")

def main():
    try:
        from model import SpamClassifier
    except Exception as exc:
        raise RuntimeError("xLSTM import failed. Activate the environment where `import xlstm` works. " + str(exc)) from exc
    model_path=ROOT/"saved_model"/"xlstm_model.pth"; vocab_path=ROOT/"saved_model"/"vocab.pkl"
    if not model_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Train the model first with `python main.py train`.")
    with open(vocab_path,"rb") as f: vocab=pickle.load(f)
    model=SpamClassifier(len(vocab.word2idx)).to(DEVICE)
    model.load_state_dict(torch.load(model_path,map_location=DEVICE,weights_only=True)); model.eval()
    print("xLSTM SMS Spam Detector (type 'exit' to stop)")
    while True:
        message=input("\nEnter SMS: ").strip()
        if message.lower()=="exit": break
        if not message:
            print("Please enter a message."); continue
        ids=[vocab.word2idx.get(w,1) for w in clean_text(message).split()]
        ids=(ids+[0]*MAX_LEN)[:MAX_LEN]
        x=torch.tensor([ids],dtype=torch.long)
        with torch.no_grad(): probs=torch.softmax(model(x),dim=1)[0]
        pred=int(torch.argmax(probs).item())
        print("Prediction :", "SPAM" if pred==1 else "HAM")
        print(f"Confidence : {probs[pred].item()*100:.2f}%")
        print(f"Ham probability : {probs[0].item()*100:.2f}% | Spam probability : {probs[1].item()*100:.2f}%")

if __name__=="__main__": main()
