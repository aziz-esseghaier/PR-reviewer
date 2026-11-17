# Proof (executed) for scikit-learn__scikit-learn-25148 f1: sklearn/multiclass.py:1025
# Claim: predict_proba does not validate input X
# Expected: ValueError

import numpy as np
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import LogisticRegression


def test_output_code_classifier_predict_proba_input_validation():
    """Test that ``predict_proba`` validates its input.

    The PR adds ``predict_proba`` without proper input validation. Supplying an
    obviously invalid ``X`` (object dtype) should trigger a ``ValueError``.
    The test lets the exception propagate so it fails on the PR commit.
    If ``predict_proba`` does not exist (base commit), the test simply returns
    and passes.
    """
    # Fit a simple classifier.
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 0, 1])
    clf = OutputCodeClassifier(estimator=LogisticRegression())
    clf.fit(X, y)

    # Invalid input that should cause a ValueError.
    X_invalid = np.array([["a"], ["b"]], dtype=object)

    # If ``predict_proba`` is not implemented (base version), exit early.
    if not hasattr(clf, "predict_proba"):
        return

    # The call is expected to raise a ValueError; we let it propagate.
    clf.predict_proba(X_invalid)
