"""
Model 2 — Random Forest with Engineered Features
Uses regex, token shape, and context window features for PII detection.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from config import RF_CONFIG, MODEL_DIR
from preprocessing.feature_eng import build_feature_matrix, extract_token_features

logger = logging.getLogger(__name__)


class RandomForestPIIModel:
    """Random Forest classifier for token-level PII detection."""

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = RandomForestClassifier(
            n_estimators=RF_CONFIG["n_estimators"],
            max_depth=RF_CONFIG["max_depth"],
            min_samples_split=RF_CONFIG["min_samples_split"],
            class_weight=RF_CONFIG["class_weight"],
            random_state=42,
            n_jobs=1,
        )
        self.context_window = RF_CONFIG["context_window"]
        self.name = "RandomForest"

    def train(self, train_docs, val_docs=None):
        """Train on token-level features."""
        logger.info(f"[{self.name}] Extracting features (window={self.context_window})...")
        features, labels = build_feature_matrix(train_docs, self.context_window)

        logger.info(f"[{self.name}] Vectorizing {len(features)} tokens...")
        X = self.vectorizer.fit_transform(features)

        logger.info(f"[{self.name}] Training classifier...")
        self.model.fit(X, labels)
        logger.info(f"[{self.name}] Training complete.")

        # Feature importance logging
        if hasattr(self.model, "feature_importances_"):
            feat_names = self.vectorizer.get_feature_names_out()
            importances = self.model.feature_importances_
            top_idx = np.argsort(importances)[-15:][::-1]
            logger.info("Top 15 features:")
            for idx in top_idx:
                logger.info(f"  {feat_names[idx]}: {importances[idx]:.4f}")

        if val_docs:
            return self.predict_docs(val_docs)
        return None

    def predict(self, features):
        X = self.vectorizer.transform(features)
        return self.model.predict(X)

    def predict_proba(self, features):
        X = self.vectorizer.transform(features)
        return self.model.predict_proba(X)

    def predict_docs(self, documents):
        all_preds = []
        for doc in documents:
            tokens = doc["tokens"]
            feats = [
                extract_token_features(tokens, i, self.context_window)
                for i in range(len(tokens))
            ]
            preds = self.predict(feats)
            all_preds.append(list(preds))
        return all_preds

    def save(self, path: Path = None):
        path = path or MODEL_DIR / "rf_pii.pkl"
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)
        logger.info(f"[{self.name}] Saved to {path}")

    def load(self, path: Path = None):
        path = path or MODEL_DIR / "rf_pii.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        logger.info(f"[{self.name}] Loaded from {path}")
