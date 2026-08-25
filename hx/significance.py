"""Deciding which pairs carry signal, and which are noise wearing a label.

THE CENTRAL PROBLEM WITH THIS DATASET
-------------------------------------
Every pair has a winner. Take the arm with the higher observed CTR, call it
the label, train a model, report accuracy. That pipeline runs perfectly and
teaches the model nothing, because most "winners" won by chance.

The median arm here has ~3,650 impressions at a ~1.5% CTR, i.e. roughly 55
clicks. The standard error on that CTR is around 0.2 percentage points, so a
gap smaller than about 0.5pp is indistinguishable from noise. Across the
19,942 same-image pairs, only 28.9% differ significantly at p < 0.05 - which
means roughly seven pairs in ten have a label that would flip if Upworthy had
run the same test again the next morning.

A model cannot exceed the ceiling that label noise imposes. If 70% of labels
are coin flips, accuracy near 65% is what perfect knowledge of the remaining
30% would look like, and a reported 85% would be evidence of a bug rather than
of skill.

So the experiment here is not "what accuracy can I get". It is: *how does
accuracy change as the labels get more trustworthy?* A model that has learned
something real must do better on pairs whose winner is certain than on pairs
whose winner is a coin flip. A model that has learned an artefact will not
care. The sweep is therefore both the result and the diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pair_significance(pairs: pd.DataFrame) -> pd.DataFrame:
    """Vectorised two-proportion z-test on every pair's CTR difference.

    As elsewhere, the statistic uses the pooled proportion (the variance under
    the null that the two headlines perform identically) while the interval
    uses the unpooled proportions.
    """
    out = pairs.copy()
    n_a, x_a = out["n_a"].values, out["x_a"].values
    n_b, x_b = out["n_b"].values, out["x_b"].values

    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a

    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_pooled > 0, diff / se_pooled, 0.0)

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)

    out["ctr_a"] = p_a
    out["ctr_b"] = p_b
    out["ctr_diff"] = diff
    out["z"] = z
    out["p_value"] = 2 * stats.norm.sf(np.abs(z))
    out["ci_low"] = diff - 1.959963985 * se_unpooled
    out["ci_high"] = diff + 1.959963985 * se_unpooled
    # b_wins is the label. Ties are dropped downstream rather than guessed.
    out["b_wins"] = (diff > 0).astype(int)
    out["is_tie"] = diff == 0
    return out


def mde_per_pair(pairs: pd.DataFrame, power: float = 0.80) -> np.ndarray:
    """Smallest CTR gap each pair could have detected at 80% power.

    Answers "was this particular test capable of resolving a realistic
    difference?" - the question that explains why so many labels are noise.
    """
    n_a, n_b = pairs["n_a"].values, pairs["n_b"].values
    p_bar = (pairs["x_a"].values + pairs["x_b"].values) / (n_a + n_b)
    z_alpha = stats.norm.isf(0.025)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt(p_bar * (1 - p_bar) * (1 / n_a + 1 / n_b))


def filter_by_significance(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Keep pairs whose winner is established at ``p < threshold``.

    ``threshold=1.0`` keeps everything, which is the "train on whatever has a
    label" baseline that this project exists to argue against.
    """
    kept = pairs[~pairs["is_tie"]]
    if threshold < 1.0:
        kept = kept[kept["p_value"] < threshold]
    return kept.reset_index(drop=True)


def label_noise_report(pairs: pd.DataFrame) -> pd.DataFrame:
    """How much of the archive survives at each level of label confidence."""
    rows = []
    total = len(pairs)
    for thr in (1.0, 0.5, 0.25, 0.10, 0.05, 0.01, 0.001):
        kept = filter_by_significance(pairs, thr)
        rows.append(
            {
                "threshold": thr,
                "n_pairs": len(kept),
                "share_of_archive": len(kept) / total,
                "median_abs_ctr_diff": float(kept["ctr_diff"].abs().median()),
                "n_experiments": kept["clickability_test_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)
