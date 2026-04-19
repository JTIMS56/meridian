"""
Model 1 — Logistic Regression + TF-IDF
Baseline token-level classifier for PII detection.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import LOGREG_CONFIG, MODEL_DIR
from preprocessing.feature_eng import build_feature_matrix

logger = logging.getLogger(__name__)


class LogRegPIIModel:
    """Logistic Regression baseline for token-level PII classification."""

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(
            max_iter=LOGREG_CONFIG["max_iter"],
            C=LOGREG_CONFIG["C"],
            solver=LOGREG_CONFIG["solver"],
            class_weight="balanced",
            n_jobs=-1,
        )
        self.name = "LogisticRegression"

    def train(self, train_docs, val_docs=None):
        """Train on token-level features extracted from documents."""
        logger.info(f"[{self.name}] Extracting features...")
        features, labels = build_feature_matrix(train_docs)

        logger.info(f"[{self.name}] Vectorizing {len(features)} tokens...")
        X = self.vectorizer.fit_transform(features)

        logger.info(f"[{self.name}] Training classifier...")
        self.model.fit(X, labels)
        logger.info(f"[{self.name}] Training complete.")

        if val_docs:
            val_preds = self.predict_docs(val_docs)
            return val_preds
        return None

    def predict(self, features):
        """Predict labels for a list of feature dicts."""
        X = self.vectorizer.transform(features)
        return self.model.predict(X)

    def predict_proba(self, features):
        """Predict label probabilities."""
        X = self.vectorizer.transform(features)
        return self.model.predict_proba(X)

    def predict_docs(self, documents):
        """Run prediction on a list of documents, return per-doc label lists."""
        all_preds = []
        for doc in documents:
            tokens = doc["tokens"]
            feats = [
                __import__("preprocessing.feature_eng", fromlist=["extract_token_features"])
                .extract_token_features(tokens, i)
                for i in range(len(tokens))
            ]
            preds = self.predict(feats)
            all_preds.append(list(preds))
        return all_preds

    def save(self, path: Path = None):
        path = path or MODEL_DIR / "logreg_pii.pkl"
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)
        logger.info(f"[{self.name}] Saved to {path}")

    def load(self, path: Path = None):
        path = path or MODEL_DIR / "logreg_pii.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        logger.info(f"[{self.name}] Loaded from {path}")
