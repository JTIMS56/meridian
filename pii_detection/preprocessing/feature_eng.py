"""
Feature engineering for token-level PII classification.
Extracts lexical, contextual, and regex-based features for each token.
Used by Logistic Regression (Model 1) and Random Forest (Model 2).
"""
import numpy as np
from typing import Dict, List
from utils.pii_patterns import (
    match_patterns, token_shape, has_at_symbol,
    is_capitalized, is_all_digits,
)


def extract_token_features(
    tokens: List[str], index: int, context_window: int = 1
) -> Dict:
    """
    Build a feature dict for a single token at `index`.

    Features include:
      - The token itself (lowercased), its shape, length
      - Boolean flags: capitalized, all-digits, has-@, etc.
      - Regex PII pattern matches
      - Context: same features for surrounding tokens within window
    """
    token = tokens[index]
    features = {
        "token_lower": token.lower(),
        "token_shape": token_shape(token),
        "token_len": len(token),
        "is_capitalized": is_capitalized(token),
        "is_all_digits": is_all_digits(token),
        "has_at": has_at_symbol(token),
        "is_title": token.istitle(),
        "is_upper": token.isupper(),
        "has_hyphen": "-" in token,
        "has_dot": "." in token,
        "position_ratio": index / max(len(tokens) - 1, 1),
    }

    # Regex pattern matches
    for name, matched in match_patterns(token).items():
        features[f"regex_{name}"] = matched

    # Context window features
    for offset in range(-context_window, context_window + 1):
        if offset == 0:
            continue
        ctx_idx = index + offset
        prefix = f"ctx_{offset:+d}_"
        if 0 <= ctx_idx < len(tokens):
            ctx_tok = tokens[ctx_idx]
            features[prefix + "cap"] = is_capitalized(ctx_tok)
            features[prefix + "digit"] = is_all_digits(ctx_tok)
        else:
            features[prefix + "cap"] = False
            features[prefix + "digit"] = False

    return features


def build_feature_matrix(documents: List[Dict], context_window: int = 3):
    """
    Convert a list of documents to parallel feature-dict and label arrays.

    Returns:
        features: list of feature dicts (one per token across all docs)
        labels:   list of label strings (one per token)
    """
    all_features = []
    all_labels = []

    for doc in documents:
        tokens = doc["tokens"]
        labels = doc["labels"]
        for i in range(len(tokens)):
            feat = extract_token_features(tokens, i, context_window)
            all_features.append(feat)
            all_labels.append(labels[i])

    return all_features, all_labels
