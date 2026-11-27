
"""Shared text preprocessing utilities."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_LEN = 50


def clean_text(text) -> str:
    """Keep preprocessing identical during training and prediction."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()
