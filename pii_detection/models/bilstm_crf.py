"""
Model 3 — BiLSTM-CRF
Sequence labeling model with bidirectional LSTM and CRF output layer.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from config import BILSTM_CONFIG, MODEL_DIR, LABEL2ID, ID2LABEL, NUM_LABELS

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────
class PIITokenDataset(Dataset):
    """Converts documents to padded integer sequences for BiLSTM input."""

    def __init__(self, documents: List[Dict], vocab: Dict[str, int] = None):
        self.documents = documents
        self.vocab = vocab or {}
        self.build_vocab = vocab is None
        self.data = self._prepare()

    def _prepare(self):
        records = []
        for doc in self.documents:
            token_ids = []
            label_ids = []
            for tok, lbl in zip(doc["tokens"], doc["labels"]):
                tok_lower = tok.lower()
                if self.build_vocab and tok_lower not in self.vocab:
                    self.vocab[tok_lower] = len(self.vocab) + 2  # 0=pad, 1=unk
                tid = self.vocab.get(tok_lower, 1)  # 1 = UNK
                lid = LABEL2ID.get(lbl, 0)
                token_ids.append(tid)
                label_ids.append(lid)
            records.append((token_ids, label_ids))
        return records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids, label_ids = self.data[idx]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
        )


def collate_fn(batch):
    """Pad sequences in a batch and return lengths."""
    tokens, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in tokens])
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return tokens_padded, labels_padded, lengths


# ── Model ────────────────────────────────────────────────────────────────────
class BiLSTMCRF(nn.Module):
    """Bidirectional LSTM with a CRF output layer for sequence labeling."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()
        cfg = {**BILSTM_CONFIG, **kwargs}
        self.embedding = nn.Embedding(
            vocab_size + 2, cfg["embedding_dim"], padding_idx=0
        )
        self.lstm = nn.LSTM(
            cfg["embedding_dim"],
            cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            bidirectional=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.hidden2tag = nn.Linear(cfg["hidden_dim"] * 2, NUM_LABELS)

        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(NUM_LABELS, NUM_LABELS))
        self.start_transitions = nn.Parameter(torch.randn(NUM_LABELS))
        self.end_transitions = nn.Parameter(torch.randn(NUM_LABELS))

    def _get_emissions(self, tokens, lengths):
        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        emissions = self.hidden2tag(self.dropout(lstm_out))
        return emissions

    def forward(self, tokens, labels, lengths, mask):
        """Compute negative log-likelihood loss via CRF."""
        emissions = self._get_emissions(tokens, lengths)
        # Simplified CRF loss (forward algorithm)
        log_likelihood = self._crf_log_likelihood(emissions, labels, mask)
        return -log_likelihood.mean()

    def decode(self, tokens, lengths, mask):
        """Viterbi decoding to find best tag sequence."""
        emissions = self._get_emissions(tokens, lengths)
        return self._viterbi_decode(emissions, mask)

    def _crf_log_likelihood(self, emissions, tags, mask):
        """Compute log-likelihood using forward algorithm."""
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            m = mask[:, i].unsqueeze(1)
            score = next_score * m + score * (1 - m)
        score += self.end_transitions
        partition = torch.logsumexp(score, dim=1)

        # Gold score
        gold = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, seq_len):
            m = mask[:, i]
            gold += (
                self.transitions[tags[:, i - 1], tags[:, i]] * m
                + emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1) * m
            )
        return gold - partition

    def _viterbi_decode(self, emissions, mask):
        """Standard Viterbi for CRF decoding."""
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            m = mask[:, i].unsqueeze(1)
            score = next_score * m + score * (1 - m)
            history.append(indices)

        score += self.end_transitions
        _, best_last = score.max(dim=1)
        best_paths = [best_last]
        for hist in reversed(history):
            best_last = hist.gather(1, best_last.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last)
        best_paths.reverse()
        return torch.stack(best_paths, dim=1)


# ── Wrapper ──────────────────────────────────────────────────────────────────
class BiLSTMPIIModel:
    """High-level wrapper matching the interface of Models 1 & 2."""

    def __init__(self):
        self.name = "BiLSTM-CRF"
        self.model = None
        self.vocab = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_docs, val_docs=None):
        cfg = BILSTM_CONFIG
        train_ds = PIITokenDataset(train_docs)
        self.vocab = train_ds.vocab
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn
        )

        self.model = BiLSTMCRF(len(self.vocab)).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])

        self.model.train()
        for epoch in range(cfg["epochs"]):
            total_loss = 0
            for tokens, labels, lengths in train_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                mask = (tokens != 0).float().to(self.device)

                # Clamp labels for CRF (replace -100 with 0)
                labels_clamped = labels.clamp(min=0)

                optimizer.zero_grad()
                loss = self.model(tokens, labels_clamped, lengths, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"[{self.name}] Epoch {epoch+1}/{cfg['epochs']} — Loss: {avg_loss:.4f}")

        if val_docs:
            return self.predict_docs(val_docs)
        return None

    def predict_docs(self, documents):
        self.model.eval()
        all_preds = []
        ds = PIITokenDataset(documents, vocab=self.vocab)
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn)

        with torch.no_grad():
            for tokens, _, lengths in loader:
                tokens = tokens.to(self.device)
                lengths = lengths.to(self.device)
                mask = (tokens != 0).float().to(self.device)
                paths = self.model.decode(tokens, lengths, mask)

                for i, length in enumerate(lengths):
                    pred_ids = paths[i, :length].cpu().tolist()
                    pred_labels = [ID2LABEL.get(p, "O") for p in pred_ids]
                    all_preds.append(pred_labels)
        return all_preds

    def save(self, path=None):
        path = path or MODEL_DIR / "bilstm_crf_pii.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "vocab": self.vocab,
        }, path)
        logger.info(f"[{self.name}] Saved to {path}")

    def load(self, path=None):
        path = path or MODEL_DIR / "bilstm_crf_pii.pt"
        data = torch.load(path, map_location=self.device)
        self.vocab = data["vocab"]
        self.model = BiLSTMCRF(len(self.vocab)).to(self.device)
        self.model.load_state_dict(data["model_state"])
        logger.info(f"[{self.name}] Loaded from {path}")
