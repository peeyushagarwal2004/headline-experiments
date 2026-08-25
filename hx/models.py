"""Pairwise models and the evaluation protocol.

THREE THINGS THAT KEEP THIS HONEST
----------------------------------
1. **Antisymmetry.** Every model works on ``f(b) - f(a)`` with no intercept, so
   swapping the headlines flips the prediction exactly. A model that could
   learn "the second slot usually wins" would score well while knowing nothing.

2. **Grouped splits.** Folds are split by ``clickability_test_id``, never by
   pair. Two pairs from a five-arm experiment share headlines; letting them
   straddle the split leaks the answer and inflates accuracy.

3. **A shuffled-label control.** Every configuration is also run with labels
   permuted within the training set. That must land at 50%. If it does not,
   the leak is in the pipeline and every other number is meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from hx.features import pair_feature_matrix


def _accuracy_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Percentile bootstrap CI on accuracy - an accuracy without one is a rumour."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, correct.size, size=(n_boot, correct.size))
    draws = correct[idx].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(correct.mean()), float(lo), float(hi)


def evaluate_heuristic(pairs: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """Logistic regression on differenced hand-built features."""
    x = pair_feature_matrix(pairs)
    y = pairs["b_wins"].values
    groups = pairs["clickability_test_id"].values
    return _cross_validate(x, y, groups, n_splits, seed, sparse_input=False)


def evaluate_tfidf(
    pairs: pd.DataFrame, n_splits: int = 5, seed: int = 0, max_features: int = 30_000
) -> dict:
    """TF-IDF word+char n-grams, differenced across the pair.

    The vectoriser is fitted inside each training fold. Fitting it on all the
    text first would let test-fold vocabulary and document frequencies inform
    training - a subtle leak that quietly adds a point or two of accuracy.
    """
    y = pairs["b_wins"].values
    groups = pairs["clickability_test_id"].values
    accs, controls = [], []

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(y, y, groups):
        vec = TfidfVectorizer(
            sublinear_tf=True, min_df=3, max_features=max_features,
            ngram_range=(1, 2), strip_accents="unicode", lowercase=True,
        )
        vec.fit(pd.concat([
            pairs.iloc[train_idx]["headline_a"], pairs.iloc[train_idx]["headline_b"]
        ]))

        def diff(idx):
            return vec.transform(pairs.iloc[idx]["headline_b"]) - vec.transform(
                pairs.iloc[idx]["headline_a"]
            )

        x_tr, x_te = diff(train_idx), diff(test_idx)
        model = LogisticRegression(max_iter=2000, fit_intercept=False, C=1.0)
        model.fit(x_tr, y[train_idx])
        accs.append(model.predict(x_te) == y[test_idx])

        rng = np.random.default_rng(seed)
        shuffled = LogisticRegression(max_iter=2000, fit_intercept=False, C=1.0)
        shuffled.fit(x_tr, rng.permutation(y[train_idx]))
        controls.append(shuffled.predict(x_te) == y[test_idx])

    correct = np.concatenate(accs)
    control = np.concatenate(controls)
    acc, lo, hi = _accuracy_ci(correct)
    return {
        "accuracy": acc, "ci_low": lo, "ci_high": hi,
        "shuffled_control": float(control.mean()), "n_eval": int(correct.size),
    }


def _cross_validate(x, y, groups, n_splits, seed, sparse_input=False) -> dict:
    accs, controls = [], []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        x_tr, x_te = x[train_idx], x[test_idx]
        if not sparse_input:
            # Scale but do NOT centre. Subtracting a mean from a differenced
            # feature destroys the antisymmetry the design depends on: after
            # centring, swapping the two headlines no longer negates the input
            # exactly, and the model can quietly encode a slot preference.
            scaler = StandardScaler(with_mean=False).fit(x_tr)
            x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)

        model = LogisticRegression(max_iter=2000, fit_intercept=False)
        model.fit(x_tr, y[train_idx])
        accs.append(model.predict(x_te) == y[test_idx])

        rng = np.random.default_rng(seed)
        shuffled = LogisticRegression(max_iter=2000, fit_intercept=False)
        shuffled.fit(x_tr, rng.permutation(y[train_idx]))
        controls.append(shuffled.predict(x_te) == y[test_idx])

    correct = np.concatenate(accs)
    control = np.concatenate(controls)
    acc, lo, hi = _accuracy_ci(correct)
    return {
        "accuracy": acc, "ci_low": lo, "ci_high": hi,
        "shuffled_control": float(control.mean()), "n_eval": int(correct.size),
    }


def heuristic_coefficients(pairs: pd.DataFrame) -> pd.DataFrame:
    """Fit on everything and read off what actually drives clicks."""
    from hx.features import FEATURE_NAMES

    x = pair_feature_matrix(pairs)
    y = pairs["b_wins"].values
    scaler = StandardScaler(with_mean=False).fit(x)
    model = LogisticRegression(max_iter=2000, fit_intercept=False)
    model.fit(scaler.transform(x), y)

    out = pd.DataFrame(
        {"feature": FEATURE_NAMES, "coefficient": model.coef_[0]}
    )
    out["abs"] = out["coefficient"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
