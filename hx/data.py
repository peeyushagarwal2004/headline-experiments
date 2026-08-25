"""Loading the Upworthy Research Archive and turning it into headline pairs.

Upworthy ran a randomised trial for every article they published between 2013
and 2015. Each experiment ("clickability test") fielded several packages, and a
package is a headline *and* an image shown together. Readers were randomly
assigned to a package; the archive records impressions and clicks per package.

That randomisation is what makes this dataset unusual as an NLP corpus. The
labels are not survey responses or engagement proxies scraped after the fact -
they are outcomes of a controlled experiment, so a headline that "won" beat its
rival under random assignment rather than through some confound.

THE CONFOUND THAT MUST BE CONTROLLED
------------------------------------
A package varies headline AND image. Comparing two packages with different
images and attributing the difference to the headline is simply wrong - the
image is doing unknown work. So pairs are formed only WITHIN an image: same
experiment, same ``eyecatcher_id``, different headline. The image is then held
constant by construction and the text is the only thing that varies.

This is expensive. Of 4,873 experiments in the exploratory release, 2,264 vary
only the image and contribute nothing to a headline model. What survives is
19,942 clean same-image pairs, and they are worth far more than a larger dirty
set would be.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
UPWORTHY_URL = "https://osf.io/download/3vqmp/"
FILENAME = "upworthy-exploratory.csv"


def _fetch() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / FILENAME
    if not path.exists():
        print(f"downloading {FILENAME} (~14 MB) ...")
        urllib.request.urlretrieve(UPWORTHY_URL, path)
        print(f"  saved {path.stat().st_size:,} bytes")
    return path


def load_packages() -> pd.DataFrame:
    """One row per package (headline + image) with its impressions and clicks."""
    df = pd.read_csv(_fetch(), low_memory=False)
    df = df.dropna(subset=["headline", "eyecatcher_id"])
    df = df[df["impressions"] > 0].copy()

    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]

    # The same headline+image can appear as more than one row within a test;
    # pool them, because they are the same treatment and splitting them would
    # understate the sample size behind each arm.
    pooled = (
        df.groupby(["clickability_test_id", "eyecatcher_id", "headline"], as_index=False)[
            ["impressions", "clicks"]
        ]
        .sum()
    )
    pooled["ctr"] = pooled["clicks"] / pooled["impressions"]
    return pooled


def build_pairs(packages: pd.DataFrame | None = None) -> pd.DataFrame:
    """All within-experiment, within-image headline pairs.

    Returns one row per unordered pair with both arms' counts. Orientation is
    arbitrary here and is randomised later during modelling, so that no model
    can exploit which side a winner happens to sit on.
    """
    if packages is None:
        packages = load_packages()

    pkg = packages.copy()
    pkg["_slot"] = pkg.groupby(["clickability_test_id", "eyecatcher_id"]).cumcount()

    left = pkg.rename(
        columns={
            "headline": "headline_a", "impressions": "n_a",
            "clicks": "x_a", "ctr": "ctr_a", "_slot": "slot_a",
        }
    )
    right = pkg.rename(
        columns={
            "headline": "headline_b", "impressions": "n_b",
            "clicks": "x_b", "ctr": "ctr_b", "_slot": "slot_b",
        }
    )

    pairs = left.merge(right, on=["clickability_test_id", "eyecatcher_id"])
    # Strict inequality gives each unordered pair exactly once.
    pairs = pairs[pairs["slot_a"] < pairs["slot_b"]].drop(columns=["slot_a", "slot_b"])

    pairs = pairs[pairs["headline_a"] != pairs["headline_b"]]
    return pairs.reset_index(drop=True)


def summarise(packages: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Descriptives worth stating before any modelling happens."""
    return {
        "n_packages": len(packages),
        "n_experiments": packages["clickability_test_id"].nunique(),
        "n_images": packages["eyecatcher_id"].nunique(),
        "total_impressions": int(packages["impressions"].sum()),
        "total_clicks": int(packages["clicks"].sum()),
        "mean_ctr": float(packages["ctr"].mean()),
        "median_ctr": float(packages["ctr"].median()),
        "n_same_image_pairs": len(pairs),
        "n_experiments_with_pairs": pairs["clickability_test_id"].nunique(),
        "median_impressions_per_arm": float(
            np.median(np.concatenate([pairs["n_a"].values, pairs["n_b"].values]))
        ),
    }
