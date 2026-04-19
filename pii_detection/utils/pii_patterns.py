"""
Regex-based PII pattern detectors.
Used as features for Models 1 & 2 and as a rule-based baseline reference.
"""
import re
from typing import Dict


# ── Compiled patterns ────────────────────────────────────────────────────────
PII_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
    ),
    "PHONE_NUM": re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    ),
    "URL_PERSONAL": re.compile(
        r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
    ),
    "ID_NUM": re.compile(
        r"\b[A-Z]{1,3}\d{5,10}\b"
    ),
    "STREET_ADDRESS": re.compile(
        r"\d{1,5}\s[\w\s]{2,30}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Ln|Lane|Ct|Court)\b",
        re.IGNORECASE,
    ),
}


def match_patterns(token: str) -> Dict[str, bool]:
    """Return a dict of boolean flags indicating which PII patterns match a token."""
    return {name: bool(pat.search(token)) for name, pat in PII_PATTERNS.items()}


def token_shape(token: str) -> str:
    """
    Reduce a token to its 'shape' for feature engineering.
    e.g. 'John' -> 'Xxxx', '123-456' -> 'd-d', 'hello@world.com' -> 'x@x.x'
    """
    shape = []
    for ch in token[:20]:  # cap length
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return "".join(shape)


def has_at_symbol(token: str) -> bool:
    return "@" in token


def is_capitalized(token: str) -> bool:
    return len(token) > 0 and token[0].isupper()


def is_all_digits(token: str) -> bool:
    return token.replace("-", "").replace(".", "").isdigit()
