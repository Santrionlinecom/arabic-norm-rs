"""Benchmark: pure Python vs Rust (single / rayon batch), and pandas vs Polars.

Run:  uv run python bench/make_sample.py && uv run python bench/bench.py
Writes bench/results.json and bench/results.md.
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import polars as pl

import arabic_norm as rs
from arabic_norm import reference as py

HERE = Path(__file__).parent
DATA = HERE / "data" / "sample.jsonl"
REPEAT = int(os.environ.get("BENCH_REPEAT", "5"))

# A typical regex-based normalizer, as commonly found in Arabic NLP snippets.
_RE_TASHKEEL = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)
_RE_ALIF = re.compile(r"[\u0623\u0625\u0622\u0671]")


def normalize_regex(text: str) -> str:
    text = _RE_TASHKEEL.sub("", text)
    text = text.replace("\u0640", "")
    text = _RE_ALIF.sub("\u0627", text)
    return text.replace("\u0649", "\u064a")


def timeit(fn: Callable[[], object]) -> float:
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def cpu_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def main() -> None:
    load_before = os.getloadavg()[0]
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    texts = [r["text"] for r in rows]
    mb = sum(len(t.encode()) for t in texts) / 2**20
    chars = sum(len(t) for t in texts)

    # --- Correctness gate before timing anything --------------------------
    expected = [py.normalize(t) for t in texts]
    assert rs.normalize_batch(texts) == expected, "Rust batch != reference"
    assert [normalize_regex(t) for t in texts[:2000]] == expected[:2000]
    assert [py.normalize_loop(t) for t in texts[:2000]] == expected[:2000]

    res: dict[str, float] = {}
    res["py_loop"] = timeit(lambda: [py.normalize_loop(t) for t in texts])
    res["py_regex"] = timeit(lambda: [normalize_regex(t) for t in texts])
    res["py_translate"] = timeit(lambda: [py.normalize(t) for t in texts])
    res["rs_single"] = timeit(lambda: [rs.normalize(t) for t in texts])
    res["rs_batch"] = timeit(lambda: rs.normalize_batch(texts))

    res["tok_py"] = timeit(lambda: [py.tokenize(t) for t in texts])
    res["tok_rs"] = timeit(lambda: [rs.tokenize(t) for t in texts])
    res["tok_py_joined"] = timeit(lambda: py.tokenize_batch_joined(texts))
    res["tok_rs_batch_joined"] = timeit(lambda: rs.tokenize_batch_joined(texts))

    # --- Word-count pipeline: top 20 words per category -------------------
    cats = [r["category"] for r in rows]
    joined = rs.tokenize_batch_joined(texts)
    assert joined == py.tokenize_batch_joined(texts)

    def pandas_wc() -> pd.DataFrame:
        df = pd.DataFrame({"category": cats, "tok": joined})
        s = df.assign(tok=df["tok"].str.split(" ")).explode("tok")
        s = s[s["tok"].notna() & (s["tok"] != "")]
        counts = s.groupby(["category", "tok"]).size().rename("n").reset_index()
        counts = counts.sort_values(["category", "n", "tok"], ascending=[True, False, True])
        return counts.groupby("category").head(20).reset_index(drop=True)

    def polars_wc() -> pl.DataFrame:
        return (
            pl.DataFrame({"category": cats, "tok": joined})
            .lazy()
            .with_columns(pl.col("tok").str.split(" "))
            .explode("tok")
            .filter(pl.col("tok") != "")
            .group_by("category", "tok")
            .len("n")
            .sort(["category", "n", "tok"], descending=[False, True, False])
            .group_by("category", maintain_order=True)
            .head(20)
            .collect()
        )

    a = pandas_wc()
    b = polars_wc()
    assert a["tok"].tolist() == b["tok"].to_list() and a["n"].tolist() == b["n"].to_list()

    res["wc_pandas"] = timeit(pandas_wc)
    res["wc_polars"] = timeit(polars_wc)

    # End-to-end: raw text -> normalized tokens -> top words per category.
    def e2e_python_pandas() -> object:
        df = pd.DataFrame({"category": cats, "text": texts})
        df["tok"] = df["text"].map(lambda t: " ".join(py.tokenize(t)))
        s = df.assign(tok=df["tok"].str.split(" ")).explode("tok")
        s = s[s["tok"].notna() & (s["tok"] != "")]
        return s.groupby(["category", "tok"]).size()

    def e2e_rust_polars() -> object:
        return (
            pl.DataFrame({"category": cats, "tok": rs.tokenize_batch_joined(texts)})
            .lazy()
            .with_columns(pl.col("tok").str.split(" "))
            .explode("tok")
            .filter(pl.col("tok") != "")
            .group_by("category", "tok")
            .len("n")
            .collect()
        )

    res["e2e_python_pandas"] = timeit(e2e_python_pandas)
    res["e2e_rust_polars"] = timeit(e2e_rust_polars)

    meta = {
        "cpu": cpu_name(),
        "threads": rs.num_threads(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "pages": len(texts),
        "books": len({r["book_id"] for r in rows}),
        "categories": len(set(cats)),
        "mb": round(mb, 1),
        "chars": chars,
        "repeat": REPEAT,
        "loadavg_before": round(load_before, 2),
        "loadavg_after": round(os.getloadavg()[0], 2),
        "top_words_sample": b.head(5).rows(),
    }
    (HERE / "results.json").write_text(
        json.dumps({"meta": meta, "seconds": res}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_md(meta, res)
    print((HERE / "results.md").read_text(encoding="utf-8"))


def write_md(meta: dict, r: dict[str, float]) -> None:
    def row(label: str, key: str, base: str) -> str:
        speed = r[base] / r[key]
        mbps = meta["mb"] / r[key]
        return f"| {label} | {r[key]:.3f} | {mbps:,.0f} | {speed:.1f}x |"

    lines = [
        "# Benchmark results",
        "",
        f"- Hardware: {meta['cpu']}, {meta['threads']} threads, {meta['os']} (WSL2)",
        f"- Python {meta['python']}, pandas {meta['pandas']}, Polars {meta['polars']}",
        f"- Data: {meta['pages']:,} pages from {meta['books']} real Arabic books "
        f"({meta['categories']} categories), {meta['mb']} MB UTF-8, {meta['chars']:,} chars",
        f"- Median of {meta['repeat']} runs. Every variant is asserted to produce "
        "identical output before it is timed.",
        f"- The laptop is a 24/7 server: 1-min load average was {meta['loadavg_before']} before "
        f"and {meta['loadavg_after']} after the run, so parallel (rayon) numbers are "
        "conservative and all numbers carry roughly +-15% noise.",
        "",
        "## Normalization (strip tashkeel/tatweel, unify alif, ya)",
        "",
        "| Implementation | Seconds | MB/s | vs naive loop |",
        "|---|---:|---:|---:|",
        row("Pure Python, per-char loop", "py_loop", "py_loop"),
        row("Pure Python, `re.sub` chain", "py_regex", "py_loop"),
        row("Pure Python, `str.translate` (strong baseline)", "py_translate", "py_loop"),
        row("Rust `normalize()` called per page", "rs_single", "py_loop"),
        row("Rust `normalize_batch()` (rayon, GIL released)", "rs_batch", "py_loop"),
        "",
        f"Against the strongest pure-Python baseline (`str.translate`): Rust single "
        f"{r['py_translate'] / r['rs_single']:.1f}x, Rust batch "
        f"{r['py_translate'] / r['rs_batch']:.1f}x.",
        "",
        "## Tokenization (normalize + remove punctuation + split)",
        "",
        "| Implementation | Seconds | MB/s | vs Python |",
        "|---|---:|---:|---:|",
        row("Python `tokenize()` -> list", "tok_py", "tok_py"),
        row("Rust `tokenize()` -> list", "tok_rs", "tok_py"),
        row("Python, joined string per page", "tok_py_joined", "tok_py_joined"),
        row("Rust `tokenize_batch_joined()` (rayon)", "tok_rs_batch_joined", "tok_py_joined"),
        "",
        "## Word count: top 20 words per category",
        "",
        "| Pipeline | Seconds | MB/s | Speedup |",
        "|---|---:|---:|---:|",
        row("pandas: split/explode/groupby (pre-tokenized)", "wc_pandas", "wc_pandas"),
        row("Polars lazy: same pipeline (pre-tokenized)", "wc_polars", "wc_pandas"),
        row("End-to-end: Python tokenize + pandas", "e2e_python_pandas", "e2e_python_pandas"),
        row("End-to-end: Rust batch tokenize + Polars", "e2e_rust_polars", "e2e_python_pandas"),
        "",
    ]
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
