"""Quick timing of the Rust paths only (used while tuning)."""

import json
import statistics
import time
from pathlib import Path

import arabic_norm as rs

DATA = Path(__file__).parent / "data" / "sample.jsonl"
texts = [json.loads(line)["text"] for line in DATA.open(encoding="utf-8")]


def med(fn, n=5):
    runs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


print(f"rs_single {med(lambda: [rs.normalize(x) for x in texts]):.3f}s")
print(f"rs_batch  {med(lambda: rs.normalize_batch(texts)):.3f}s")
print(f"tok_batch {med(lambda: rs.tokenize_batch_joined(texts)):.3f}s")
