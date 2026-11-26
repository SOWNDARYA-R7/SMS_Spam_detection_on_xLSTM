"""Simple menu for the SMS spam detection project."""
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COMMANDS={"process": "process.py", "train":"train.py", "test":"test.py", "predict":"predict.py", "compare":"compare_models.py"}

def run_command(command, extra_args=None):
    script=ROOT/COMMANDS[command]
    cmd=[sys.executable,str(script)]+(extra_args or [])
    return subprocess.call(cmd,cwd=ROOT)

def menu():
    print("\n======================================")
    print(" SMS SPAM DETECTION USING xLSTM")
    print("======================================")
    print("1. Process dataset and create 80/20 split")
    print("2. Train proposed xLSTM model")
    print("3. Test saved xLSTM model + confusion matrix")
    print("4. Predict a new SMS")
    print("5. Compare xLSTM, LSTM, Bi-LSTM, GRU, TF-IDF+SVM, XGBoost")
    print("0. Exit")
    while True:
        choice=input("\nEnter choice: ").strip()
        mapping={"1":"process","2":"train","3":"test","4":"predict","5":"compare"}
        if choice=="0": print("Exit."); return
        if choice in mapping:
            try: run_command(mapping[choice])
            except KeyboardInterrupt: print("\nStopped by user.")
        else: print("Please enter 0, 1, 2, 3, 4, or 5.")

def main():
    if len(sys.argv)>1:
        name=sys.argv[1].lower()
        if name not in COMMANDS:
            print("Usage: python main.py [process|train|test|predict|compare]"); raise SystemExit(2)
        raise SystemExit(run_command(name,sys.argv[2:]))
    menu()
if __name__=="__main__": main()
