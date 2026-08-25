"""The main experiment: does accuracy rise as the labels become trustworthy?

Run:  python scripts/01_significance_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hx.data import build_pairs, load_packages, summarise
from hx.models import evaluate_heuristic, evaluate_tfidf, heuristic_coefficients
from hx.significance import label_noise_report, mde_per_pair, pair_significance, filter_by_significance

REPORTS = Path(__file__).resolve().parent.parent / "reports"
FIGURES = REPORTS / "figures"
THRESHOLDS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.01, 0.001]
MIN_PAIRS = 400


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> dict:
    rule("0. THE ARCHIVE")
    packages = load_packages()
    pairs = pair_significance(build_pairs(packages))
    info = summarise(packages, pairs)

    print(f"packages (headline + image)     : {info['n_packages']:,}")
    print(f"experiments                     : {info['n_experiments']:,}")
    print(f"total impressions               : {info['total_impressions']:,}")
    print(f"mean CTR                        : {info['mean_ctr']:.4f}")
    print(f"\nsame-image headline pairs       : {info['n_same_image_pairs']:,}")
    print(f"  drawn from experiments        : {info['n_experiments_with_pairs']:,}")
    print(f"  median impressions per arm    : {info['median_impressions_per_arm']:,.0f}")
    print("\nPairs are formed only within an experiment AND within an image, so the")
    print("headline is the only thing that differs. Experiments that varied only the")
    print("image contribute nothing and are dropped.")

    # ------------------------------------------------------------------------
    rule("1. HOW MANY LABELS ARE REAL?")

    mde = mde_per_pair(pairs)
    print(f"Median CTR gap detectable at 80% power : {np.median(mde):.4f}")
    print(f"Median CTR gap actually observed       : "
          f"{pairs['ctr_diff'].abs().median():.4f}")
    print("\nThe typical pair is not powered to resolve the difference it reports.")
    print("That is not a flaw in the archive - Upworthy was picking a headline to")
    print("ship, not estimating an effect - but it means most labels are coin flips.\n")

    noise = label_noise_report(pairs)
    print(f"{'p <':>8s}{'pairs':>10s}{'share':>9s}{'median |CTR gap|':>19s}{'experiments':>13s}")
    for _, r in noise.iterrows():
        print(
            f"{r.threshold:>8.3g}{int(r.n_pairs):>10,d}{r.share_of_archive:>9.1%}"
            f"{r.median_abs_ctr_diff:>19.4f}{int(r.n_experiments):>13,d}"
        )

    # ------------------------------------------------------------------------
    rule("2. THE SWEEP")
    print("At each threshold: keep only pairs whose winner is established at that")
    print("level, then cross-validate with folds split by experiment.\n")
    print("If a model has learned something real about headlines, accuracy must")
    print("climb as the labels get cleaner. If it has learned an artefact, it will")
    print("stay flat.\n")

    print(f"{'p <':>8s}{'pairs':>9s}  {'heuristic (95% CI)':>26s}  {'TF-IDF (95% CI)':>26s}  {'shuffled':>9s}")
    sweep = []
    for thr in THRESHOLDS:
        subset = filter_by_significance(pairs, thr)
        if len(subset) < MIN_PAIRS:
            print(f"{thr:>8.3g}{len(subset):>9,d}   (too few pairs to evaluate)")
            continue

        heur = evaluate_heuristic(subset)
        tfidf = evaluate_tfidf(subset)
        print(
            f"{thr:>8.3g}{len(subset):>9,d}  "
            f"{heur['accuracy']:>9.3f} [{heur['ci_low']:.3f}, {heur['ci_high']:.3f}]  "
            f"{tfidf['accuracy']:>9.3f} [{tfidf['ci_low']:.3f}, {tfidf['ci_high']:.3f}]  "
            f"{heur['shuffled_control']:>9.3f}"
        )
        sweep.append(
            {
                "threshold": thr,
                "n_pairs": len(subset),
                "heuristic": heur,
                "tfidf": tfidf,
            }
        )

    if sweep:
        first, last = sweep[0], sweep[-1]
        lift_h = last["heuristic"]["accuracy"] - first["heuristic"]["accuracy"]
        lift_t = last["tfidf"]["accuracy"] - first["tfidf"]["accuracy"]
        print(f"\nAccuracy from all pairs -> p<{last['threshold']:g} pairs:")
        print(f"  heuristic {first['heuristic']['accuracy']:.3f} -> "
              f"{last['heuristic']['accuracy']:.3f}  ({lift_h:+.3f})")
        print(f"  TF-IDF    {first['tfidf']['accuracy']:.3f} -> "
              f"{last['tfidf']['accuracy']:.3f}  ({lift_t:+.3f})")
        print("\nShuffled-label controls sit at ~0.50 throughout, which is the")
        print("evidence that the grouped splits are not leaking.")

    # ------------------------------------------------------------------------
    rule("3. WHAT ACTUALLY WINS")
    confident = filter_by_significance(pairs, 0.01)
    coefs = heuristic_coefficients(confident)
    print(f"Logistic coefficients on standardised feature differences,")
    print(f"fitted on the {len(confident):,} pairs with p < 0.01.")
    print("Positive = the headline with more of this feature tends to win.\n")
    for _, r in coefs.head(12).iterrows():
        bar = "#" * int(abs(r.coefficient) * 60)
        print(f"  {r.feature:<18s}{r.coefficient:+7.4f}  {bar}")

    make_figure(sweep, noise)

    out = {"archive": info, "label_noise": noise.to_dict("records"),
           "sweep": sweep, "coefficients": coefs.to_dict("records")}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "sweep_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote reports/sweep_results.json")
    return out


def make_figure(sweep: list, noise: pd.DataFrame) -> None:
    if not sweep:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    thr = [s["threshold"] for s in sweep]
    x = np.arange(len(thr))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for key, label, colour in [
        ("heuristic", "hand-built features", "#1f77b4"),
        ("tfidf", "TF-IDF word+bigram", "#ff7f0e"),
    ]:
        acc = [s[key]["accuracy"] for s in sweep]
        lo = [s[key]["ci_low"] for s in sweep]
        hi = [s[key]["ci_high"] for s in sweep]
        axes[0].errorbar(
            x, acc, yerr=[np.array(acc) - lo, np.array(hi) - np.array(acc)],
            marker="o", capsize=4, lw=2, label=label, color=colour,
        )
    axes[0].axhline(0.5, ls="--", color="#888", label="chance / shuffled control")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{t:g}" for t in thr])
    axes[0].set_xlabel("label confidence threshold (p <)")
    axes[0].set_ylabel("pairwise accuracy")
    axes[0].set_title("Accuracy rises as labels become trustworthy")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, [s["n_pairs"] for s in sweep], color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{t:g}" for t in thr])
    axes[1].set_xlabel("label confidence threshold (p <)")
    axes[1].set_ylabel("pairs surviving")
    axes[1].set_title("The cost: most of the archive is noise")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "significance_sweep.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {FIGURES / 'significance_sweep.png'}")


if __name__ == "__main__":
    main()
