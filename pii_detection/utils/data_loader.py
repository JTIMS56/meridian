"""
Data loading utilities for the Kaggle PII Detection dataset.
Loads train.json / test.json and converts annotations to BIO-tagged token sequences.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from config import DATA_DIR, LABEL2ID, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO

logger = logging.getLogger(__name__)


def load_kaggle_data(split: str = "train") -> List[Dict]:
    """Load raw Kaggle JSON data."""
    filepath = DATA_DIR / f"{split}.json"
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Download from: "
            "https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} documents from {filepath}")
    return data


def extract_tokens_and_labels(data: List[Dict]) -> List[Dict]:
    """
    Convert Kaggle format to a list of documents, each with:
      - 'document': document ID
      - 'tokens': list of token strings
      - 'labels': list of BIO label strings
      - 'trailing_whitespace': whitespace flags per token
    """
    documents = []
    for doc in data:
        entry = {
            "document": doc["document"],
            "tokens": doc["tokens"],
            "trailing_whitespace": doc["trailing_whitespace"],
            "labels": doc.get("labels", ["O"] * len(doc["tokens"])),
        }
        documents.append(entry)
    return documents


def get_label_distribution(documents: List[Dict]) -> Dict[str, int]:
    """Count label occurrences across all documents."""
    counts = {}
    for doc in documents:
        for label in doc["labels"]:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def split_data(
    documents: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split documents into train / val / test sets."""
    train_docs, temp_docs = train_test_split(
        documents, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
    )
    relative_val = VAL_RATIO / (1 - TRAIN_RATIO)
    val_docs, test_docs = train_test_split(
        temp_docs, test_size=(1 - relative_val), random_state=RANDOM_SEED
    )
    logger.info(
        f"Split: {len(train_docs)} train / {len(val_docs)} val / {len(test_docs)} test"
    )
    return train_docs, val_docs, test_docs


def tokens_to_bio_indices(labels: List[str]) -> List[int]:
    """Convert string labels to integer indices using LABEL2ID."""
    return [LABEL2ID.get(lbl, 0) for lbl in labels]
