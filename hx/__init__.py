"""hx - headline experiments over the Upworthy Research Archive.

    data.py          load the archive, build same-image headline pairs
    significance.py  which pairs have a trustworthy winner, and which do not
    features.py      interpretable headline features
    models.py        antisymmetric pairwise models with grouped CV
"""

from hx import data, features, models, significance

__all__ = ["data", "significance", "features", "models"]
__version__ = "0.1.0"
