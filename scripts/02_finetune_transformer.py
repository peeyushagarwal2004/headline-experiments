"""Fine-tune a DistilBERT cross-encoder on the significance-filtered pairs.

This is the transformer half of the project. It reuses the TESTED pairing and
significance code in hx/, so the data it trains on is identical to what the
baselines saw - the only thing that changes is the model.

    python scripts/02_finetune_transformer.py --threshold 0.01 --epochs 4

Needs a GPU. On CPU it will run but take hours. Designed to drop straight into
Google Colab or Kaggle (see the README's "Run on GPU" section).

DESIGN (matches the baselines so the comparison is fair)
--------------------------------------------------------
* Cross-encoder: both headlines are read together as `[CLS] A [SEP] B [SEP]`,
  so attention can compare them directly. A bi-encoder scoring each headline
  alone cannot express "B is punchier THAN A".
* Antisymmetry by data: every pair is emitted in both orderings with flipped
  labels, and predictions are averaged over the two orderings at eval time. A
  transformer has no structural reason to be antisymmetric, so without this it
  learns a position bias and scores spuriously well.
* Grouped split by clickability_test_id into train / val / test (60/20/20), so
  no experiment's headlines leak across the split.
* The significance filter is applied BEFORE splitting, so train and test share
  the same label quality - the whole point of the significance sweep.

It reports accuracy AND log-loss on the held-out test set, and prints the
TF-IDF baseline at the same threshold (read from reports/sweep_results.json) so
the numbers are directly comparable. If the transformer does not clearly beat
TF-IDF, that is a real finding about how much signal a short headline carries -
report it, do not bury it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hx.data import build_pairs
from hx.significance import filter_by_significance, pair_significance

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def both_orderings(pairs):
    """Emit each pair twice (swapped) with flipped labels; keep the group id."""
    a = list(pairs["headline_a"]) + list(pairs["headline_b"])
    b = list(pairs["headline_b"]) + list(pairs["headline_a"])
    y = list(pairs["b_wins"]) + list(1 - pairs["b_wins"])
    g = list(pairs["clickability_test_id"]) * 2
    # pair_id ties the two orderings of the same pair together for averaging.
    pid = list(range(len(pairs))) + list(range(len(pairs)))
    return a, b, np.array(y), np.array(g), np.array(pid)


def grouped_split(groups, seed=42, fracs=(0.6, 0.2, 0.2)):
    """Assign whole experiments to train / val / test."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    n = len(uniq)
    c1, c2 = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    tr, va, te = set(uniq[:c1]), set(uniq[c1:c2]), set(uniq[c2:])
    which = np.array(["train" if g in tr else "val" if g in va else "test" for g in groups])
    return which


def baseline_at(threshold):
    """The TF-IDF and heuristic accuracy at this threshold, if the sweep was run."""
    p = REPORTS / "sweep_results.json"
    if not p.exists():
        return None
    for row in json.loads(p.read_text())["sweep"]:
        if abs(row["threshold"] - threshold) < 1e-9:
            return {"tfidf": row["tfidf"]["accuracy"], "heuristic": row["heuristic"]["accuracy"],
                    "n_pairs": row["n_pairs"]}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="keep pairs whose winner is established at p < this")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  get_linear_schedule_with_warmup)
    except ImportError:
        sys.exit("Needs torch + transformers:  pip install torch transformers")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " - CPU will be slow"))

    pairs = filter_by_significance(pair_significance(build_pairs()), args.threshold)
    print(f"{len(pairs):,} pairs at p < {args.threshold:g}")

    texts_a, texts_b, labels, groups, pair_ids = both_orderings(pairs)
    split = grouped_split(groups, seed=args.seed)
    print(f"split (rows, both orderings): "
          f"train {int((split=='train').sum()):,} / "
          f"val {int((split=='val').sum()):,} / test {int((split=='test').sum()):,}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    class DS(Dataset):
        def __init__(self, mask):
            self.idx = np.where(mask)[0]

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            enc = tok(texts_a[j], texts_b[j], truncation=True,
                      max_length=args.max_length, padding="max_length")
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"] = torch.tensor(int(labels[j]))
            item["row"] = torch.tensor(int(j))
            return item

    train_dl = DataLoader(DS(split == "train"), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(DS(split == "val"), batch_size=args.batch_size)
    test_dl = DataLoader(DS(split == "test"), batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    @torch.no_grad()
    def evaluate(dl):
        """Average the two orderings per pair, then score. Returns acc + log-loss."""
        model.eval()
        probs, rows = {}, {}
        for batch in dl:
            row = batch.pop("row").numpy()
            y = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**batch).logits.float()
            p1 = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            for r, pr, yy in zip(row, p1, y):
                pid = pair_ids[r]
                # If this row is a swapped ordering, its P(b wins) is P(a wins)
                # of the canonical pair, so flip it before averaging.
                canon = pr if labels[r] == pairs["b_wins"].iloc[pid] else 1 - pr
                probs.setdefault(pid, []).append(canon)
                rows[pid] = pairs["b_wins"].iloc[pid]
        pids = list(probs)
        p = np.array([np.mean(probs[k]) for k in pids])
        y = np.array([rows[k] for k in pids])
        acc = float(((p > 0.5).astype(int) == y).mean())
        eps = 1e-7
        pc = np.clip(p, eps, 1 - eps)
        logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        return acc, logloss, len(pids)

    best_val, best_state, history = 0.0, None, []
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for batch in train_dl:
            batch.pop("row")
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        va, vll, _ = evaluate(val_dl)
        history.append({"epoch": ep + 1, "train_loss": tot / len(train_dl),
                        "val_acc": va, "val_logloss": vll})
        print(f"epoch {ep+1}: train_loss {tot/len(train_dl):.4f}  "
              f"val_acc {va:.4f}  val_logloss {vll:.4f}")
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_ll, n_test = evaluate(test_dl)
    base = baseline_at(args.threshold)

    results = {
        "model": args.model, "threshold": args.threshold, "epochs": args.epochs,
        "n_pairs": len(pairs), "n_test_pairs": n_test,
        "test_accuracy": test_acc, "test_logloss": test_ll,
        "best_val_accuracy": best_val, "history": history,
        "baseline_tfidf": base["tfidf"] if base else None,
        "baseline_heuristic": base["heuristic"] if base else None,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"transformer_p{args.threshold}.json"
    out.write_text(json.dumps(results, indent=2))

    # ---- copy-paste block ---------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS  (copy this whole block back)")
    print("=" * 60)
    print(f"model           : {args.model}")
    print(f"threshold       : p < {args.threshold:g}   ({len(pairs):,} pairs)")
    print(f"test pairs      : {n_test:,}  (held-out experiments)")
    print(f"TEST ACCURACY   : {test_acc:.4f}")
    print(f"TEST LOG-LOSS   : {test_ll:.4f}")
    if base:
        delta = test_acc - base["tfidf"]
        print(f"TF-IDF baseline : {base['tfidf']:.4f}   (transformer {delta:+.4f})")
        print(f"heuristic base  : {base['heuristic']:.4f}")
        verdict = ("BEATS TF-IDF" if delta > 0.01 else
                   "ties TF-IDF" if abs(delta) <= 0.01 else "loses to TF-IDF")
        print(f"verdict         : {verdict}")
    print("=" * 60)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
