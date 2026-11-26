# SMS Spam Detection Using xLSTM

An AI-powered SMS classification system that uses **Extended Long Short-Term Memory (xLSTM)** networks to classify text messages as **Ham (Legitimate)** or **Spam**. The project also compares xLSTM with traditional machine learning algorithms and other deep learning architectures.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/Model-xLSTM-purple" />
  <img src="https://img.shields.io/badge/Domain-NLP-green" />
</p>

---

## Overview

SMS spam messages are a common form of unwanted communication. This project develops a text classification system that automatically identifies whether an SMS message is legitimate or spam.

The proposed system uses an **xLSTM-based deep learning architecture** with mLSTM and sLSTM blocks to learn contextual information from text sequences.

In addition to xLSTM, multiple machine learning and deep learning models are evaluated using the same dataset split to study their classification performance.

## Key Features

- SMS text preprocessing and cleaning.
- Automatic Ham and Spam classification.
- Tokenization and vocabulary generation.
- Word embedding using 128-dimensional vectors.
- Custom xLSTM architecture with mLSTM and sLSTM blocks.
- Model training and performance evaluation.
- SMS prediction with confidence scores.
- Confusion matrix and training performance visualizations.
- Comparison of multiple machine learning and deep learning models.
- Reproducible stratified 80:20 train-test split.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | xLSTM, LSTM, Bi-LSTM, GRU |
| Machine Learning | Scikit-learn, XGBoost |
| NLP | Text Preprocessing, Tokenization, Word Embeddings |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## Project Structure

```text
SMS_Spam_xLSTM_Complete_Project/
│
├── dataset/
│   └── spam.csv
│
├── processed/
│   ├── cleaned_sms.csv
│   └── train_test_split.csv
│
├── saved_model/
│   ├── model weights
│   └── vocabulary
│
├── results/
│   ├── training_curves/
│   ├── confusion_matrix/
│   └── comparison_run/
│
├── main.py
├── process.py
├── dataset.py
├── model.py
├── train.py
├── test.py
├── predict.py
├── compare_models.py
│
├── requirements.txt
├── README.md
└── run_project.bat
```

## Model Architecture

The proposed xLSTM architecture consists of:

1. **Embedding Layer:** Converts token IDs into 128-dimensional dense vectors.
2. **xLSTM Backbone:** Contains four xLSTM blocks consisting of three mLSTM blocks and one sLSTM block.
3. **Mean Pooling:** Aggregates sequence representations.
4. **Fully Connected Layer:** Performs binary classification.
5. **Output:** Predicts Ham or Spam.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Embedding Dimension | 128 |
| Number of xLSTM Blocks | 4 |
| Block Configuration | mLSTM, sLSTM, mLSTM, mLSTM |
| Number of Heads | 4 |
| Sequence Length | 50 |
| Batch Size | 32 |
| Epochs | 5 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Execution Device | CPU by default |

## Dataset and Preprocessing

The project uses an SMS spam classification dataset stored in `dataset/spam.csv`.

### Preprocessing Steps

1. Load the SMS dataset.
2. Reconstruct messages by joining split text columns where necessary.
3. Remove invalid labels and empty messages.
4. Convert labels into numerical values:
   - Ham = 0
   - Spam = 1
5. Identify and report duplicate label-message pairs.
6. Generate a stratified 80:20 training and testing split.
7. Build vocabulary using training data only.
8. Convert text into numerical token sequences.
9. Apply padding and truncation to a maximum sequence length of 50.

**Data Split:** 80% Training / 20% Testing  
**Random State:** 42

## Model Comparison

The project supports comparison between six classification approaches:

| Model | Type |
|---|---|
| xLSTM | Proposed Deep Learning Model |
| LSTM | Deep Learning |
| Bi-LSTM | Deep Learning |
| GRU | Deep Learning |
| TF-IDF + SVM | Machine Learning |
| XGBoost | Machine Learning |

All models are evaluated using the same training and testing split.

### Evaluation Metrics

The following metrics are used to evaluate classification performance:

- **Accuracy:** Overall percentage of correctly classified messages.
- **Precision:** Percentage of predicted spam messages that are actually spam.
- **Recall:** Percentage of actual spam messages correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Displays correct and incorrect classifications.

Spam is treated as the positive class when calculating Precision, Recall, and F1-Score.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SOWNDARYA-R7/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd SMS_Spam_xLSTM_Complete_Project
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

**Note:** The xLSTM implementation requires a compatible `xlstm` package and PyTorch environment. If installation fails, use a Python environment in which the xLSTM package is already working.

Verify the installation:

```bash
python -c "import torch; import xlstm; print('PyTorch and xLSTM imports work')"
```

## How to Run

### Option 1: Run Using Python

Start the project menu:

```bash
python main.py
```

You can also execute individual stages:

**Step 1 – Data Preprocessing**

```bash
python main.py process
```

**Step 2 – Train xLSTM Model**

```bash
python main.py train
```

**Step 3 – Evaluate the Model**

```bash
python main.py test
```

**Step 4 – Predict SMS Messages**

```bash
python main.py predict
```

**Step 5 – Compare Models**

```bash
python main.py compare
```

### Option 2: Run Using Windows Batch File

Double-click:

```text
run_project.bat
```

This automatically opens the project directory and executes `main.py`.

## Prediction Example

The prediction module allows users to enter a custom SMS message and classify it.

**Example input:**

```text
Congratulations! You have won a free prize.
Call now to claim your reward.
```

**Expected output format:**

```text
Predicted Class : Spam
Confidence      : [Model-generated confidence]
```

The actual prediction and confidence depend on the trained model.

## Output Files

After executing the project, generated files are stored in the following directories:

| Directory | Description |
|---|---|
| `processed/` | Cleaned dataset and train-test split |
| `saved_model/` | Trained model weights and vocabulary |
| `results/` | Evaluation metrics and visualizations |
| `results/comparison_run/` | Model comparison results |

The comparison metrics are exported to:

```text
results/comparison_run/model_metrics.csv
```

## Research and Experimental Notes

- All models should use the same dataset split for a fair comparison.
- xLSTM must be executed in an environment with a working xLSTM installation.
- Do not report previously obtained metrics from a different dataset as results for the current dataset.
- Use the final experiment outputs when preparing research tables and performance comparisons.
- The dataset size and preprocessing details should match the final experiment.

## Future Enhancements

- Develop a web-based interface for real-time SMS classification.
- Integrate the trained model with a Flask or FastAPI backend.
- Support batch SMS classification.
- Improve model performance through hyperparameter tuning.
- Explore additional transformer-based NLP architectures.

## Author

**Sowndarya R**

Full Stack Developer | Data Analytics Enthusiast

- GitHub: [SOWNDARYA-R7](https://github.com/SOWNDARYA-R7)
- LinkedIn: [sowndaryar](https://www.linkedin.com/in/sowndaryar/)

---

<p align="center">
  <b>SMS Spam Detection Using xLSTM</b><br>
  Built with Python, PyTorch, and Natural Language Processing.
</p>