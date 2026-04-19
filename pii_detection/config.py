"""
MERIDIAN PII Detection — Configuration
Paths, hyperparameters, label definitions, and model settings.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "saved_models"

# Create dirs if missing
for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)

# ── PII Labels (BIO tagging scheme) ─────────────────────────────────────────
PII_LABELS = [
    "O",                    # Outside (non-PII)
    "B-NAME_STUDENT",  "I-NAME_STUDENT",
    "B-EMAIL",         "I-EMAIL",
    "B-USERNAME",      "I-USERNAME",
    "B-ID_NUM",        "I-ID_NUM",
    "B-PHONE_NUM",     "I-PHONE_NUM",
    "B-URL_PERSONAL",  "I-URL_PERSONAL",
    "B-STREET_ADDRESS","I-STREET_ADDRESS",
]

LABEL2ID = {label: i for i, label in enumerate(PII_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(PII_LABELS)}
NUM_LABELS = len(PII_LABELS)

# ── Data Splits ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# ── Model 1: Logistic Regression ────────────────────────────────────────────
LOGREG_CONFIG = {
    "max_iter": 1000,
    "C": 1.0,
    "solver": "lbfgs",
    "tfidf_max_features": 50000,
    "tfidf_ngram_range": (1, 2),
}

# ── Model 2: Random Forest ──────────────────────────────────────────────────
RF_CONFIG = {
    "n_estimators": 100,        # was 300
    "max_depth": 20,            # was 30
    "min_samples_split": 5,
    "class_weight": "balanced",
    "context_window": 1,        # was 3
}

# ── Model 3: BiLSTM-CRF ─────────────────────────────────────────────────────
BILSTM_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,           # was 32 — larger batches = fewer steps
    "epochs": 3,                # was 10 — just enough for the paper
    "max_seq_len": 512,
}

# ── Model 4: DeBERTa Transformer ────────────────────────────────────────────
TRANSFORMER_CONFIG = {
    "model_name": "microsoft/deberta-v3-base",
    "learning_rate": 2e-5,
    "batch_size": 8,
    "epochs": 5,
    "max_seq_len": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
}

# ── Evaluation ───────────────────────────────────────────────────────────────
METRICS_LIST = ["precision", "recall", "f1", "auc_roc"]
VISUALIZATION_DPI = 150
