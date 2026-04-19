"""
MERIDIAN PII Detection — End-to-End Pipeline
Trains and evaluates all four models, generates comparison visualizations.

Usage:
    py -3.11 pii_detection/run_pii_pipeline.py --model all
    py -3.11 pii_detection/run_pii_pipeline.py --model logreg
    py -3.11 pii_detection/run_pii_pipeline.py --model rf
    py -3.11 pii_detection/run_pii_pipeline.py --model bilstm
    py -3.11 pii_detection/run_pii_pipeline.py --model transformer
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure pii_detection is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, DATA_DIR
from utils.data_loader import (
    load_kaggle_data,
    extract_tokens_and_labels,
    get_label_distribution,
    split_data,
)
from evaluation.metrics import (
    flatten_predictions,
    compute_token_metrics,
    compute_entity_f1,
    get_confusion_matrix,
    summarize_results,
)
from evaluation.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_error_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pii_pipeline")

MODEL_REGISTRY = {
    "logreg": ("models.baseline_logreg", "LogRegPIIModel"),
    "rf": ("models.random_forest", "RandomForestPIIModel"),
    "bilstm": ("models.bilstm_crf", "BiLSTMPIIModel"),
    "transformer": ("models.transformer_ner", "TransformerPIIModel"),
}


def load_model_class(key: str):
    """Dynamically import a model class from the registry."""
    module_path, class_name = MODEL_REGISTRY[key]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def run_pipeline(model_keys: list):
    """Train and evaluate selected models."""
    # ── 1. Load & split data ─────────────────────────────────────────────
    logger.info("Loading Kaggle PII dataset...")
    raw = load_kaggle_data("train")
    documents = extract_tokens_and_labels(raw)
    if any(k in model_keys for k in ["bilstm", "transformer"]):
        documents = documents[:500]
    dist = get_label_distribution(documents)
    logger.info(f"Label distribution:\n{json.dumps(dist, indent=2)}")

    train_docs, val_docs, test_docs = split_data(documents)

    # ── 2. Train & evaluate each model ───────────────────────────────────
    all_results = {}

    for key in model_keys:
        logger.info(f"\n{'='*60}\n  Training: {key}\n{'='*60}")
        ModelClass = load_model_class(key)
        model = ModelClass()

        # Train
        model.train(train_docs, val_docs)

        # Predict on test set
        test_preds = model.predict_docs(test_docs)

        # Evaluate
        y_true, y_pred = flatten_predictions(test_docs, test_preds)
        token_metrics = compute_token_metrics(y_true, y_pred)
        entity_metrics = compute_entity_f1(y_true, y_pred)

        summary = {
            "precision": token_metrics.get("weighted avg", {}).get("precision", 0),
            "recall": token_metrics.get("weighted avg", {}).get("recall", 0),
            "f1": token_metrics.get("weighted avg", {}).get("f1-score", 0),
            "entity_f1": entity_metrics["entity_f1"],
        }
        all_results[model.name] = summary
        print(summarize_results(model.name, summary))

        # Confusion matrix
        cm, cm_labels = get_confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, cm_labels, model.name)

        # Error analysis
        all_tokens = [tok for doc in test_docs for tok in doc["tokens"]]
        plot_error_analysis(y_true, y_pred, all_tokens, model.name)

        # Save model
        model.save()

    # ── 3. Comparison visualizations ─────────────────────────────────────
    if len(all_results) > 1:
        plot_model_comparison_bar(all_results)

    # Save results JSON
    results_path = OUTPUT_DIR / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    logger.info("\nPipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="MERIDIAN PII Detection Pipeline")
    parser.add_argument(
        "--model",
        choices=["all", "logreg", "rf", "bilstm", "transformer"],
        default="all",
        help="Which model(s) to train and evaluate.",
    )
    args = parser.parse_args()

    if args.model == "all":
        keys = list(MODEL_REGISTRY.keys())
    else:
        keys = [args.model]

    run_pipeline(keys)


if __name__ == "__main__":
    main()
