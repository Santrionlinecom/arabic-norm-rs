"""Render bench/results.json as a 1280x720 PNG (bench/chart.png)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

GROUPS = [
    (
        "Normalize 40 MB of Arabic text",
        [
            ("Python loop", "py_loop"),
            ("Python re.sub", "py_regex"),
            ("Python str.translate", "py_translate"),
            ("Rust, per call", "rs_single"),
            ("Rust batch (rayon)", "rs_batch"),
        ],
        "py_translate",
    ),
    (
        "Word count, end-to-end",
        [
            ("Python tokenize + pandas", "e2e_python_pandas"),
            ("Rust tokenize + Polars", "e2e_rust_polars"),
        ],
        "e2e_python_pandas",
    ),
]


def main() -> None:
    data = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    sec, meta = data["seconds"], data["meta"]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 7.2), dpi=100, gridspec_kw={"width_ratios": [5, 3]}
    )
    for ax, (title, items, base) in zip(axes, GROUPS, strict=True):
        labels = [label for label, _ in items]
        values = [sec[key] for _, key in items]
        fast = ("rs_", "e2e_rust")
        colors = ["#d97706" if key.startswith(fast) else "#64748b" for _, key in items]
        bars = ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
        ax.set_xlabel("seconds (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, (_, key) in zip(bars, items, strict=True):
            speed = sec[base] / sec[key]
            if key == base:
                rel = "baseline"
            elif speed >= 1:
                rel = f"{speed:.1f}x faster"
            else:
                rel = f"{1 / speed:.1f}x slower"
            note = f" {sec[key]:.2f}s  ({rel})"
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, note, va="center")
        ax.set_xlim(0, max(values) * 1.45)

    fig.suptitle(
        "arabic-norm: Rust + PyO3 vs pure Python, Polars vs pandas", fontsize=17, x=0.02, ha="left"
    )
    fig.text(
        0.02,
        0.015,
        f"{meta['pages']:,} pages from {meta['books']} real Arabic books ({meta['mb']} MB) | "
        f"{meta['cpu'].replace('11th Gen Intel(R) Core(TM) ', '')}, {meta['threads']} threads, "
        f"WSL2 | Python {meta['python']}, Polars {meta['polars']}, pandas {meta['pandas']} | "
        f"median of {meta['repeat']} runs, outputs asserted identical",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = HERE / "chart.png"
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
