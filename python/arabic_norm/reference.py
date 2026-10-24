"""Pure-Python reference implementation of :mod:`arabic_norm`.

This is the specification. The Rust extension must produce byte-identical
output for every input; ``tests/test_parity.py`` enforces that with
example-based and property-based (Hypothesis) tests.

It is also a strong pure-Python baseline: it uses ``str.translate`` (a C-level
loop) rather than a per-character Python loop.
"""

from __future__ import annotations

import string
from functools import lru_cache

TATWEEL = "\u0640"

# Harakat, tanwin, shadda, sukun, Quranic annotation marks, superscript alif.
TASHKEEL_RANGES = [
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06DC),
    (0x06DF, 0x06E8),
    (0x06EA, 0x06ED),
]
TASHKEEL = frozenset(chr(c) for lo, hi in TASHKEEL_RANGES for c in range(lo, hi + 1))

ALIF_VARIANTS = "\u0623\u0625\u0622\u0671"  # أ إ آ ٱ
ALIF = "\u0627"  # ا
ALIF_MAQSURA = "\u0649"  # ى
YA = "\u064a"  # ي
TA_MARBUTA = "\u0629"  # ة
HA = "\u0647"  # ه

PUNCT_RANGES = [
    (0x060C, 0x060C),  # ، comma
    (0x061B, 0x061B),  # ؛ semicolon
    (0x061F, 0x061F),  # ؟ question mark
    (0x066A, 0x066D),  # ٪ ٫ ٬ ٭
    (0x06D4, 0x06D4),  # ۔ full stop
    (0x00AB, 0x00AB),  # «
    (0x00BB, 0x00BB),  # »
    (0x2010, 0x2027),  # dashes, quotes, bullets, ellipsis
    (0x2030, 0x205E),  # per-mille, primes, misc punctuation
    (0xFD3E, 0xFD3F),  # ﴾ ﴿
]
PUNCT = frozenset(string.punctuation) | frozenset(
    chr(c) for lo, hi in PUNCT_RANGES for c in range(lo, hi + 1)
)


@lru_cache(maxsize=64)
def _table(
    strip_tashkeel: bool,
    strip_tatweel: bool,
    normalize_alif: bool,
    normalize_ya: bool,
    ta_marbuta: bool,
    remove_punct: bool,
) -> dict[int, str | None]:
    table: dict[int, str | None] = {}
    # Order matters only where sets overlap; they do not, but keep it explicit:
    # punctuation first, then letter mappings, then deletions win.
    if remove_punct:
        table.update({ord(c): " " for c in PUNCT})
    if normalize_alif:
        table.update({ord(c): ALIF for c in ALIF_VARIANTS})
    if normalize_ya:
        table[ord(ALIF_MAQSURA)] = YA
    if ta_marbuta:
        table[ord(TA_MARBUTA)] = HA
    if strip_tatweel:
        table[ord(TATWEEL)] = None
    if strip_tashkeel:
        table.update({ord(c): None for c in TASHKEEL})
    return table


def normalize(
    text: str,
    *,
    strip_tashkeel: bool = True,
    strip_tatweel: bool = True,
    normalize_alif: bool = True,
    normalize_ya: bool = True,
    ta_marbuta: bool = False,
    remove_punct: bool = False,
) -> str:
    """Normalize Arabic text for search / embedding."""
    table = _table(
        strip_tashkeel, strip_tatweel, normalize_alif, normalize_ya, ta_marbuta, remove_punct
    )
    return text.translate(table)


def normalize_batch(texts: list[str], **kwargs: bool) -> list[str]:
    return [normalize(t, **kwargs) for t in texts]


def tokenize(text: str, **kwargs: bool) -> list[str]:
    """Normalize with punctuation removed, then split on whitespace."""
    return normalize(text, remove_punct=True, **kwargs).split()


def tokenize_batch_joined(texts: list[str], **kwargs: bool) -> list[str]:
    return [" ".join(tokenize(t, **kwargs)) for t in texts]


def normalize_loop(text: str) -> str:
    """Naive per-character loop with default options (benchmark baseline only)."""
    out = []
    for ch in text:
        if ch in TASHKEEL or ch == TATWEEL:
            continue
        if ch in ALIF_VARIANTS:
            out.append(ALIF)
        elif ch == ALIF_MAQSURA:
            out.append(YA)
        else:
            out.append(ch)
    return "".join(out)
