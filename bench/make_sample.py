"""Build a <= 40 MB benchmark sample from a local corpus of Arabic books.

Expected layout (Shamela export): <root>/<category>/<book>/pages.jsonl with one
JSON object per page and the page text in the ``body`` field.

If no corpus is found, a synthetic Arabic corpus with realistic harakat,
tatweel and punctuation density is generated instead.

The sample stays on the local disk (bench/data/ is git-ignored).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_ROOT = Path.home() / "kitab-corpus" / "shamela4" / "books"
OUT = Path(__file__).parent / "data" / "sample.jsonl"


def from_corpus(root: Path, budget: int, seed: int) -> list[dict]:
    books = sorted(root.glob("*/*/pages.jsonl"))
    random.Random(seed).shuffle(books)
    rows, used = [], 0
    for path in books:
        category = path.parent.parent.name.split("__", 1)[-1]
        book_id = path.parent.name.split("__", 1)[0]
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                body = json.loads(line).get("body") or ""
                if not body.strip():
                    continue
                rows.append({"book_id": book_id, "category": category, "text": body})
                used += len(body.encode())
                if used >= budget:
                    return rows
    return rows


def synthetic(budget: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    letters = [chr(c) for c in range(0x0621, 0x064B) if c != 0x0640]
    marks = [chr(c) for c in range(0x064B, 0x0653)]
    punct = ["،", ".", "؛", "؟", ":", "«", "»", "(", ")"]
    rows, used = [], 0
    while used < budget:
        words = []
        for _ in range(rng.randint(150, 400)):
            w = "".join(
                rng.choice(letters) + (rng.choice(marks) if rng.random() < 0.5 else "")
                for _ in range(rng.randint(2, 7))
            )
            if rng.random() < 0.02:
                w = w[:1] + "ـــ" + w[1:]
            if rng.random() < 0.08:
                w += rng.choice(punct)
            words.append(w)
        text = " ".join(words)
        rows.append({"book_id": str(rng.randint(1, 500)), "category": "synthetic", "text": text})
        used += len(text.encode())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--mb", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    budget = args.mb * 1024 * 1024
    if args.root.exists():
        rows, source = from_corpus(args.root, budget, args.seed), f"corpus:{args.root}"
    else:
        rows, source = synthetic(budget, args.seed), "synthetic"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    size = sum(len(r["text"].encode()) for r in rows)
    books = len({r["book_id"] for r in rows})
    cats = len({r["category"] for r in rows})
    print(f"{source}: {len(rows)} pages, {books} books, {cats} categories, {size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
