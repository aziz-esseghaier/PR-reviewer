# Proof (executed) for pydata__xarray-6939 f1: xarray/core/variable.py:2143
# Claim: Error message string does not interpolate the method name
# Expected: ValueError

import pytest
import numpy as np
from xarray.core.variable import Variable


def test_quantile_error_message_contains_method_name(monkeypatch):
    """Ensure the ValueError raised for unsupported ``method`` includes the method name.

    The test forces an old NumPy version (< 1.22) to trigger the legacy branch
    where an unsupported interpolation method should raise a ``ValueError``.
    The error message is expected to contain the actual method name.
    """
    # Simulate an old NumPy version to hit the legacy code path.
    monkeypatch.setattr(np, "__version__", "1.21.0", raising=False)

    # Create a simple Variable.
    var = Variable(["x"], np.arange(4))

    unsupported_method = "foo"
    with pytest.raises(ValueError) as exc:
        var.quantile(q=0.5, method=unsupported_method)

    # The message should include the unsupported method name.
    assert unsupported_method in str(exc.value)
