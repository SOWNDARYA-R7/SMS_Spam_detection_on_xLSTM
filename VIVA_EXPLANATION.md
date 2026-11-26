# Simple viva explanation of each file

- **main.py:** Main menu. It starts preprocessing, training, testing, prediction, or model comparison.
- **process.py:** Reads the dataset, combines message text if it spills into extra CSV columns, removes rows with invalid labels or empty messages, standardizes text, and writes `train.csv` and `test.csv`.
- **dataset.py:** Creates a vocabulary and changes each SMS into a list of 50 numbers. It also creates batches for model training.
- **model.py:** Contains the proposed xLSTM architecture. Embedding converts tokens into 128-dimensional vectors. The xLSTM stack processes the sequence. Mean pooling summarizes the sequence, and the final layer selects Ham or Spam.
- **train.py:** Learns model parameters from training SMS messages. Adam updates the weights and CrossEntropyLoss measures classification error.
- **test.py:** Evaluates only the held-out test split. It prints metrics and creates a confusion matrix.
- **predict.py:** Takes a new message, uses the same vocabulary and cleaning steps, and shows Ham/Spam plus confidence.
- **compare_models.py:** Compares the proposed xLSTM with LSTM, Bi-LSTM, GRU, TF-IDF + SVM, and XGBoost. Every model is tested on the same held-out examples.

## Explain the metrics

- **Accuracy:** percentage of all test messages predicted correctly.
- **Precision (Spam):** out of messages predicted as Spam, how many were actually Spam.
- **Recall (Spam):** out of all real Spam messages, how many the model found.
- **F1-score (Spam):** balance between Spam precision and Spam recall.
- **Confusion matrix:** rows are actual classes; columns are predicted classes. Top-left is Ham predicted as Ham; top-right is Ham predicted as Spam; bottom-left is Spam predicted as Ham; bottom-right is Spam predicted as Spam.
