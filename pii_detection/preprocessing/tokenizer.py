"""
Tokenization and label-alignment utilities.
Handles alignment between whitespace tokens (Kaggle format) and
subword tokens (for Transformer models).
"""
import logging
from typing import Dict, List, Tuple

from config import LABEL2ID

logger = logging.getLogger(__name__)


def align_labels_to_subwords(
    tokenized_input,
    original_labels: List[str],
    label2id: Dict[str, int] = LABEL2ID,
    ignore_index: int = -100,
) -> List[int]:
    """
    Align BIO labels from whitespace tokens to subword tokens produced
    by a HuggingFace tokenizer.

    - First subword of a token keeps the original label.
    - Subsequent subwords get the I- version if original was B-.
    - Special tokens ([CLS], [SEP], padding) get ignore_index.
    """
    word_ids = tokenized_input.word_ids()
    aligned = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token
            aligned.append(ignore_index)
        elif word_id != prev_word_id:
            # First subword of a new token → keep original label
            lbl = original_labels[word_id] if word_id < len(original_labels) else "O"
            aligned.append(label2id.get(lbl, 0))
        else:
            # Continuation subword → map B-X to I-X
            lbl = original_labels[word_id] if word_id < len(original_labels) else "O"
            if lbl.startswith("B-"):
                lbl = "I-" + lbl[2:]
            aligned.append(label2id.get(lbl, 0))
        prev_word_id = word_id

    return aligned


def reconstruct_text(tokens: List[str], trailing_ws: List[bool]) -> str:
    """Rebuild original text from tokens and trailing whitespace flags."""
    parts = []
    for tok, ws in zip(tokens, trailing_ws):
        parts.append(tok)
        if ws:
            parts.append(" ")
    return "".join(parts)
