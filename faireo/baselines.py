"""Lightweight statistical baselines for country-level validation."""
import numpy as np


def heldout_country_mean(values):
    """Predict each country's outcome from the *other* countries only.

    Leave-one-country-out evaluation must not use the held-out outcome
    when constructing its own reference prediction.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Expected at least two finite country-level values")
    return (y.sum() - y) / (len(y) - 1)
