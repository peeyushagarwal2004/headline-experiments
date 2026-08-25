"""Tests for the pairing logic, the significance test, and antisymmetry.

The pairing rules are where this project can silently go wrong: a pair that
crosses images, or a fold that splits an experiment, produces a believable
number that means nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hx import features, significance
from hx.data import build_pairs
from hx.models import evaluate_heuristic


def toy_packages() -> pd.DataFrame:
    """Two experiments; the first has two images, the second only one headline."""
    rows = [
        # experiment 1, image A: two headlines -> 1 pair
        ("t1", "imgA", "Headline one", 1000, 20),
        ("t1", "imgA", "Headline two", 1000, 40),
        # experiment 1, image B: three headlines -> 3 pairs
        ("t1", "imgB", "Third headline", 1000, 10),
        ("t1", "imgB", "Fourth headline", 1000, 30),
        ("t1", "imgB", "Fifth headline", 1000, 50),
        # experiment 2: one headline across two images -> 0 pairs
        ("t2", "imgC", "Only headline", 1000, 25),
        ("t2", "imgD", "Only headline", 1000, 35),
    ]
    df = pd.DataFrame(
        rows, columns=["clickability_test_id", "eyecatcher_id", "headline",
                       "impressions", "clicks"]
    )
    df["ctr"] = df.clicks / df.impressions
    return df


def test_pairs_are_within_image_only():
    """A pair must never span two different images."""
    pairs = build_pairs(toy_packages())
    assert len(pairs) == 1 + 3
    merged = pairs.merge(
        toy_packages(), left_on=["clickability_test_id", "eyecatcher_id"],
        right_on=["clickability_test_id", "eyecatcher_id"],
    )
    assert not merged.empty
    # Every pair carries exactly one image id, so the image is held constant.
    assert pairs["eyecatcher_id"].notna().all()


def test_single_headline_experiments_contribute_nothing():
    """Experiments that varied only the image must produce zero pairs."""
    pairs = build_pairs(toy_packages())
    assert "t2" not in set(pairs["clickability_test_id"])


def test_each_unordered_pair_appears_once():
    pairs = build_pairs(toy_packages())
    keys = {
        frozenset([r.headline_a, r.headline_b]) for r in pairs.itertuples()
    }
    assert len(keys) == len(pairs)


def test_significance_matches_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest

    pairs = pd.DataFrame(
        {"n_a": [1000, 5000], "x_a": [20, 100], "n_b": [1000, 5000], "x_b": [40, 90],
         "clickability_test_id": ["t1", "t2"],
         "headline_a": ["a", "c"], "headline_b": ["b", "d"]}
    )
    out = significance.pair_significance(pairs)
    for i in range(len(pairs)):
        z_sm, p_sm = proportions_ztest(
            [pairs.x_b[i], pairs.x_a[i]], [pairs.n_b[i], pairs.n_a[i]]
        )
        assert out.z[i] == pytest.approx(z_sm, rel=1e-9)
        assert out.p_value[i] == pytest.approx(p_sm, rel=1e-9)


def test_label_is_the_higher_ctr_arm():
    pairs = pd.DataFrame(
        {"n_a": [1000], "x_a": [20], "n_b": [1000], "x_b": [40],
         "clickability_test_id": ["t1"], "headline_a": ["a"], "headline_b": ["b"]}
    )
    out = significance.pair_significance(pairs)
    assert out.b_wins.iloc[0] == 1     # b has the higher CTR


def test_filtering_is_monotone():
    """Tighter thresholds must keep a strict subset."""
    rng = np.random.default_rng(0)
    n = 500
    pairs = pd.DataFrame(
        {"n_a": np.full(n, 3000), "x_a": rng.binomial(3000, 0.015, n),
         "n_b": np.full(n, 3000), "x_b": rng.binomial(3000, 0.016, n),
         "clickability_test_id": [f"t{i}" for i in range(n)],
         "headline_a": ["a"] * n, "headline_b": ["b"] * n}
    )
    scored = significance.pair_significance(pairs)
    sizes = [len(significance.filter_by_significance(scored, t))
             for t in (1.0, 0.5, 0.1, 0.05, 0.01)]
    assert sizes == sorted(sizes, reverse=True)


def test_features_are_antisymmetric_under_swap():
    """f(b) - f(a) must be exactly the negation of f(a) - f(b)."""
    pairs = pd.DataFrame({"headline_a": ["You won't believe this!"],
                          "headline_b": ["A calm and measured statement"]})
    forward = features.pair_feature_matrix(pairs)
    swapped = features.pair_feature_matrix(
        pairs.rename(columns={"headline_a": "headline_b", "headline_b": "headline_a"})
    )
    assert np.allclose(forward, -swapped)


def test_feature_vector_length_matches_names():
    assert len(features.featurise_one("test headline")) == len(features.FEATURE_NAMES)


def test_shuffled_labels_give_chance_accuracy():
    """The leakage canary: permuted labels must not be predictable."""
    rng = np.random.default_rng(3)
    n = 1200
    words = ["you", "this", "amazing", "why", "never", "best", "the", "a"]
    pairs = pd.DataFrame(
        {
            "headline_a": [" ".join(rng.choice(words, 8)) for _ in range(n)],
            "headline_b": [" ".join(rng.choice(words, 8)) for _ in range(n)],
            "b_wins": rng.integers(0, 2, n),
            "clickability_test_id": [f"t{i // 3}" for i in range(n)],
        }
    )
    out = evaluate_heuristic(pairs, n_splits=4)
    assert out["ci_low"] <= 0.5 <= out["ci_high"]
    assert out["shuffled_control"] == pytest.approx(0.5, abs=0.08)


def test_mde_is_larger_for_smaller_arms():
    """Less traffic means a coarser detectable difference."""
    small = pd.DataFrame({"n_a": [500], "x_a": [8], "n_b": [500], "x_b": [10]})
    large = pd.DataFrame({"n_a": [50_000], "x_a": [800], "n_b": [50_000], "x_b": [1000]})
    assert significance.mde_per_pair(small)[0] > significance.mde_per_pair(large)[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
