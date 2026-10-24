"""Fast Arabic text normalization (Rust + PyO3), with a pure-Python reference.

>>> import arabic_norm
>>> arabic_norm.normalize("إِمَامٌ")
'امام'
"""

from arabic_norm._core import (
    normalize,
    normalize_batch,
    num_threads,
    tokenize,
    tokenize_batch_joined,
)

__all__ = [
    "normalize",
    "normalize_batch",
    "num_threads",
    "tokenize",
    "tokenize_batch_joined",
]
