"""
Evaluation metrics for PII detection.
Token-level and entity-level precision, recall, F1, and per-class breakdown.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config import PII_LABELS, LABEL2ID

logger = logging.getLogger(__name__)


def flatten_predictions(
    true_docs: List[Dict], pred_docs: List[List[str]]
) -> Tuple[List[str], List[str]]:
    """Flatten document-level labels and predictions to token-level lists."""
    y_true = []
    y_pred = []
    for doc, preds in zip(true_docs, pred_docs):
        y_true.extend(doc["labels"])
        y_pred.extend(preds)
    return y_true, y_pred


def compute_token_metrics(
    y_true: List[str], y_pred: List[str]
) -> Dict:
    """Compute token-level classification metrics."""
    # Filter to labels present in data
    present_labels = sorted(set(y_true + y_pred))
    report = classification_report(
        y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0
    )
    return report


def compute_entity_f1(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Compute entity-level F1 using strict BIO matching.
    An entity is correct only if both boundaries and label match exactly.
    """
    true_entities = _extract_entities(y_true)
    pred_entities = _extract_entities(y_pred)

    true_set = set(true_entities)
    pred_set = set(pred_entities)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "true_entities": len(true_set),
        "pred_entities": len(pred_set),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
    """Extract (entity_type, start, end) tuples from BIO label sequence."""
    entities = []
    current_type = None
    start = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_type:
                entities.append((current_type, start, i))
            current_type = label[2:]
            start = i
        elif label.startswith("I-") and current_type == label[2:]:
            continue  # extend current entity
        else:
            if current_type:
                entities.append((current_type, start, i))
                current_type = None
                start = None

    if current_type:
        entities.append((current_type, start, len(labels)))

    return entities


def compute_auc_roc(y_true: List[str], y_proba: np.ndarray, classes: List[str]) -> Dict:
    """
    Compute macro-averaged AUC-ROC for multiclass token classification.
    y_proba: shape (n_tokens, n_classes) — predicted probabilities.
    """
    y_true_bin = label_binarize(y_true, classes=classes)
    try:
        auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        auc = None
        logger.warning("AUC-ROC could not be computed (possibly single class in split).")
    return {"auc_roc_macro": auc}


def get_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str] = None):
    """Return confusion matrix array and label list."""
    labels = labels or sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def summarize_results(model_name: str, metrics: Dict) -> str:
    """Format metrics dict as a printable summary."""
    lines = [f"\n{'='*60}", f"  {model_name} — Evaluation Summary", f"{'='*60}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:30s}: {value:.4f}")
        else:
            lines.append(f"  {key:30s}: {value}")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)
