"""
Model 4 — DeBERTa Transformer (Fine-tuned Token Classification)
State-of-the-art approach using HuggingFace Transformers for PII NER.
"""
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TRANSFORMER_CONFIG, MODEL_DIR, LABEL2ID, ID2LABEL, NUM_LABELS
from preprocessing.tokenizer import align_labels_to_subwords

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────
class TransformerPIIDataset(Dataset):
    """Tokenize documents with HuggingFace tokenizer and align BIO labels."""

    def __init__(self, documents: List[Dict], tokenizer, max_len: int = 512):
        self.data = []
        for doc in documents:
            tokens = doc["tokens"]
            labels = doc["labels"]
            # Tokenize with is_split_into_words for pre-tokenized input
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            aligned = align_labels_to_subwords(encoded, labels)
            # Pad or truncate aligned labels
            aligned = aligned[:max_len]
            aligned += [-100] * (max_len - len(aligned))

            self.data.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": torch.tensor(aligned, dtype=torch.long),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Wrapper ──────────────────────────────────────────────────────────────────
class TransformerPIIModel:
    """HuggingFace DeBERTa fine-tuned for PII token classification."""

    def __init__(self):
        self.name = "DeBERTa-v3"
        self.cfg = TRANSFORMER_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def _init_model(self):
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.cfg["model_name"],
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

    def train(self, train_docs, val_docs=None):
        from transformers import TrainingArguments, Trainer
        import evaluate as hf_evaluate

        self._init_model()

        train_ds = TransformerPIIDataset(
            train_docs, self.tokenizer, self.cfg["max_seq_len"]
        )
        val_ds = (
            TransformerPIIDataset(val_docs, self.tokenizer, self.cfg["max_seq_len"])
            if val_docs
            else None
        )

        seqeval = hf_evaluate.load("seqeval")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            true_labels = []
            true_preds = []
            for pred_seq, label_seq in zip(preds, labels):
                t_labels = []
                t_preds = []
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        t_labels.append(ID2LABEL.get(l, "O"))
                        t_preds.append(ID2LABEL.get(p, "O"))
                true_labels.append(t_labels)
                true_preds.append(t_preds)
            results = seqeval.compute(
                predictions=true_preds, references=true_labels, mode="strict", scheme="IOB2"
            )
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
            }

        training_args = TrainingArguments(
            output_dir=str(MODEL_DIR / "deberta_checkpoints"),
            num_train_epochs=self.cfg["epochs"],
            per_device_train_batch_size=self.cfg["batch_size"],
            per_device_eval_batch_size=self.cfg["batch_size"] * 2,
            learning_rate=self.cfg["learning_rate"],
            warmup_ratio=self.cfg["warmup_ratio"],
            weight_decay=self.cfg["weight_decay"],
            fp16=self.cfg["fp16"] and torch.cuda.is_available(),
            eval_strategy="epoch" if val_ds else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_ds else False,
            metric_for_best_model="f1" if val_ds else None,
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        logger.info(f"[{self.name}] Starting fine-tuning...")
        trainer.train()
        logger.info(f"[{self.name}] Training complete.")

        if val_docs:
            return self.predict_docs(val_docs)
        return None

    def predict_docs(self, documents):
        self.model.eval()
        all_preds = []

        for doc in documents:
            tokens = doc["tokens"]
            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.cfg["max_seq_len"],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits

            pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
            word_ids = encoded.word_ids()

            # Map subword predictions back to word-level
            word_preds = ["O"] * len(tokens)
            for idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(tokens):
                    word_preds[word_id] = ID2LABEL.get(pred_ids[idx], "O")

            all_preds.append(word_preds)

        return all_preds

    def save(self, path=None):
        path = path or MODEL_DIR / "deberta_pii"
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"[{self.name}] Saved to {path}")

    def load(self, path=None):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        path = path or MODEL_DIR / "deberta_pii"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device)
        logger.info(f"[{self.name}] Loaded from {path}")
