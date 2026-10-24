# Benchmark results

- Hardware: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 8 threads, Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
- Python 3.14.7, pandas 3.1.0, Polars 2.0.0
- Data: 22,618 pages from 36 real Arabic books (13 categories), 40.0 MB UTF-8, 23,249,540 chars
- Median of 5 runs. Every variant is asserted to produce identical output before it is timed.
- The laptop is a 24/7 server: 1-min load average was 4.29 before and 3.16 after the run, so parallel (rayon) numbers are conservative and all numbers carry roughly +-15% noise.

## Normalization (strip tashkeel/tatweel, unify alif, ya)

| Implementation | Seconds | MB/s | vs naive loop |
|---|---:|---:|---:|
| Pure Python, per-char loop | 3.311 | 12 | 1.0x |
| Pure Python, `re.sub` chain | 0.908 | 44 | 3.6x |
| Pure Python, `str.translate` (strong baseline) | 0.681 | 59 | 4.9x |
| Rust `normalize()` called per page | 0.180 | 223 | 18.4x |
| Rust `normalize_batch()` (rayon, GIL released) | 0.056 | 717 | 59.3x |

Against the strongest pure-Python baseline (`str.translate`): Rust single 3.8x, Rust batch 12.2x.

## Tokenization (normalize + remove punctuation + split)

| Implementation | Seconds | MB/s | vs Python |
|---|---:|---:|---:|
| Python `tokenize()` -> list | 1.472 | 27 | 1.0x |
| Rust `tokenize()` -> list | 0.724 | 55 | 2.0x |
| Python, joined string per page | 1.115 | 36 | 1.0x |
| Rust `tokenize_batch_joined()` (rayon) | 0.070 | 569 | 15.8x |

## Word count: top 20 words per category

| Pipeline | Seconds | MB/s | Speedup |
|---|---:|---:|---:|
| pandas: split/explode/groupby (pre-tokenized) | 3.489 | 11 | 1.0x |
| Polars lazy: same pipeline (pre-tokenized) | 0.281 | 142 | 12.4x |
| End-to-end: Python tokenize + pandas | 3.978 | 10 | 1.0x |
| End-to-end: Rust batch tokenize + Polars | 0.345 | 116 | 11.5x |
