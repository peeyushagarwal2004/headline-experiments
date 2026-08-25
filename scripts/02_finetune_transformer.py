"""Fine-tune a cross-encoder on the significance-filtered pairs.

    python scripts/02_finetune_transformer.py --threshold 0.01 --epochs 3

NOT RUN in the environment this repo was built in - it needs `torch` and
`transformers`, and realistically a GPU. The numbers in the README come from
the heuristic and TF-IDF baselines, which were run. Treat this script as the
next step, not as a reported result.

DESIGN
------
A cross-encoder reads both headlines at once (`[CLS] A [SEP] B [SEP]`) so
attention can compare them directly, which is what the task actually asks for.
A bi-encoder scoring each headline alone cannot express "B is punchier *than*
A".

Antisymmetry is enforced by data rather than architecture: every pair is
emitted in both orderings with flipped labels. A transformer has no structural
reason to be antisymmetric, so without this it will learn a position bias and
score suspiciously well. At inference the two orderings are averaged.

Splits are grouped by experiment id, matching the baselines, and the
significance filter is applied BEFORE splitting so train and test have the
same label quality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hx.data import build_pairs
from hx.significance import filter_by_significance, pair_significance


def build_both_orderings(pairs):
    """Emit each pair twice, swapped, with flipped labels."""
    texts_a = list(pairs["headline_a"]) + list(pairs["headline_b"])
    texts_b = list(pairs["headline_b"]) + list(pairs["headline_a"])
    labels = list(pairs["b_wins"]) + list(1 - pairs["b_wins"])
    groups = list(pairs["clickability_test_id"]) * 2
    return texts_a, texts_b, np.array(labels), np.array(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="keep pairs whose winner is established at p < this")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        sys.exit(
            "This script needs torch and transformers:\n"
            "    pip install torch transformers\n"
            "The baselines in scripts/01_significance_sweep.py have no such dependency."
        )

    pairs = filter_by_significance(pair_significance(build_pairs()), args.threshold)
    print(f"{len(pairs):,} pairs at p < {args.threshold:g}")

    texts_a, texts_b, labels, groups = build_both_orderings(pairs)
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(
            labels, labels, groups
        )
    )
    print(f"train {len(train_idx):,} / test {len(test_idx):,} (grouped by experiment)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class PairData(Dataset):
        def __init__(self, idx):
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            enc = tokenizer(
                texts_a[j], texts_b[j], truncation=True,
                max_length=args.max_length, padding="max_length",
            )
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"] = torch.tensor(int(labels[j]))
            return item

    train_loader = DataLoader(PairData(train_idx), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(PairData(test_idx), batch_size=args.batch_size)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}: train loss {total / len(train_loader):.4f}")

        model.eval()
        correct = []
        with torch.no_grad():
            for batch in test_loader:
                y = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(**batch).logits.argmax(-1)
                correct.append((pred == y).cpu().numpy())
        acc = np.concatenate(correct).mean()
        print(f"          test accuracy {acc:.4f}")

    print(
        "\nCompare against the baselines at the same threshold "
        "(see reports/sweep_results.json).\n"
        "If this does not clearly beat TF-IDF, say so - that is a real finding "
        "about how much signal short headlines carry, not a failure to report."
    )


if __name__ == "__main__":
    main()
