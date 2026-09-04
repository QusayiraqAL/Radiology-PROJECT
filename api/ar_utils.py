# -*- coding: utf-8 -*-
"""Shared Arabic text utilities — single source of truth for normalization,
imported by data-prep, training, and the API server so preprocessing is identical.

The diacritics class uses explicit \\u escapes with DISJOINT ranges that skip the
Arabic letter block (U+0621..U+064A); a single contiguous range would wrongly
delete the letters themselves.
"""
import re

_AR_DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_KEEP = re.compile("[^؀-ۿ0-9\\s]")
_WS = re.compile(r"\s+")


def normalize_ar(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _AR_DIAC.sub("", s)
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
           .replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
           .replace("ة", "ه"))
    s = _KEEP.sub(" ", s)
    return _WS.sub(" ", s).strip()


def arabic_ratio(s: str) -> float:
    """Fraction of non-space characters that are Arabic (used by the input filter)."""
    if not s:
        return 0.0
    letters = [c for c in s if not c.isspace()]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if "؀" <= c <= "ۿ")
    return ar / len(letters)
