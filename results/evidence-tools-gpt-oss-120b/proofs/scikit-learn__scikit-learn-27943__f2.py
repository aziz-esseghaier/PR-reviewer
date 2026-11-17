# Proof (executed) for scikit-learn__scikit-learn-27943 f2: sklearn/linear_model/_quantile.py:150
# Claim: Changing default solver may alter model behavior for existing code
# Expected: AssertionError

import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor


def test_default_solver_behavior_consistency():
    """Ensure that the default ``solver`` yields the same predictions as the
    historic ``interior-point`` solver. On the base commit the default is
    ``interior-point`` so the predictions match and the assertion passes. After
    the default change to ``highs`` the predictions differ, causing an
    ``AssertionError`` as claimed.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    y = rng.randn(100)

    # Baseline using the historic default solver explicitly
    baseline = QuantileRegressor(solver="interior-point", quantile=0.5).fit(X, y)
    pred_baseline = baseline.predict(X)

    # Model using the current default solver (no explicit solver argument)
    default = QuantileRegressor(quantile=0.5).fit(X, y)
    pred_default = default.predict(X)

    # The predictions should match; a mismatch indicates the default change altered behavior
    assert np.allclose(pred_default, pred_baseline, atol=1e-6)
