
"""CSV cleaning and deterministic train/test split for the SMS spam project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from text_utils import clean_text

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "spam.csv"
DEFAULT_OUT = ROOT / "processed"
SEED = 42


def _read_raw(path: Path) -> pd.DataFrame:
    # Read the CSV while handling extra columns caused by unquoted commas.
    raw = pd.read_csv(path, encoding="latin1", engine="python")

    if raw.shape[1] < 2:
        raw = pd.read_csv(
            path, encoding="latin1", header=None, engine="python"
        )

    first_name = str(raw.columns[0]).strip().lower()
    known_header_names = {
        "v1", "label", "class", "category", "type", "target"
    }

    if (
        first_name not in known_header_names
        and "label" not in first_name
        and "class" not in first_name
    ):
        first_value = (
            str(raw.iloc[0, 0]).strip().lower() if len(raw) else ""
        )

        if first_value not in {"ham", "spam"}:
            raw = pd.read_csv(
                path, encoding="latin1", header=None, engine="python"
            )

    return raw


def load_clean_dataframe(path: str | Path = DEFAULT_DATA):
    path = Path(path)

    if not path.is_absolute():
        path = (ROOT / path).resolve() if not path.exists() else path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the CSV at dataset/spam.csv or pass --data."
        )

    raw = _read_raw(path)

    if raw.shape[1] < 2:
        raise ValueError("Dataset needs a label column and a message column.")

    labels = raw.iloc[:, 0].astype("string").str.strip().str.lower()

    messages = []

    for _, row in raw.iloc[:, 1:].iterrows():
        parts = []

        for value in row.tolist():
            if pd.notna(value):
                s = str(value).strip()

                if s and s.lower() != "nan":
                    parts.append(s)

        messages.append(" ".join(parts))

    df = pd.DataFrame({
        "label_text": labels,
        "message": messages
    })

    df["label"] = df["label_text"].map({"ham": 0, "spam": 1})

    before = len(df)

    df = df.dropna(subset=["label"]).copy()
    df["message"] = df["message"].fillna("").astype(str).str.strip()
    df = df[df["message"].ne("")].copy()

    df["label"] = df["label"].astype(int)
    df["message"] = df["message"].map(clean_text)
    df = df[df["message"].ne("")].copy()

    invalid_dropped = int(before - len(df))

    duplicate_pairs = int(
        df.duplicated(subset=["label", "message"]).sum()
    )

    return df[["label", "message"]].reset_index(drop=True), {
        "input_file": str(path),
        "raw_rows_read": int(before),
        "valid_rows_used": int(len(df)),
        "invalid_or_blank_rows_dropped": invalid_dropped,
        "repeated_label_message_pairs_reported_not_removed": duplicate_pairs,
        "train_test_split": "stratified 80/20",
        "random_state": SEED,
        "label_mapping": {"ham": 0, "spam": 1},
    }


def process_dataset(
    data_path: str | Path = DEFAULT_DATA,
    output_dir: str | Path = DEFAULT_OUT
):
    output_dir = Path(output_dir)

    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    df, report = load_clean_dataframe(data_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    df.to_csv(
        output_dir / "cleaned_sms.csv",
        index=False,
        encoding="utf-8"
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
        encoding="utf-8"
    )

    report.update({
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_label_counts": {
            str(int(k)): int(v)
            for k, v in train_df["label"].value_counts().sort_index().items()
        },
        "testing_label_counts": {
            str(int(k)): int(v)
            for k, v in test_df["label"].value_counts().sort_index().items()
        },
    })

    with open(
        output_dir / "preprocessing_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nDATA PREPROCESSING COMPLETE")
    print("=" * 46)

    for k, v in report.items():
        print(f"{k}: {v}")

    print(f"\nSaved files in: {output_dir}")

    return train_df, test_df, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the input Kaggle SMS CSV"
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT),
        help="Folder to save cleaned and split data"
    )

    args = parser.parse_args()

    process_dataset(args.data, args.output)


if __name__ == "__main__":
    main()
