# Proof (executed) for sympy__sympy-26233 f7: sympy/functions/special/error_functions.py:2078
# Claim: I*pi term placement may affect the series sign
# Expected: AssertionError

import pytest
from sympy import symbols, Ci, I, pi, oo
from sympy.core.function import PoleError

def test_ci_series_negative_infinity_i_pi_absent():
    z = symbols('z')
    try:
        ser = Ci(z).series(z, -oo, 5)
    except PoleError:
        # Base commit: series not implemented, treat as pass
        return
    # PR commit: series should not contain I*pi term
    assert not ser.has(I * pi), "Ci(z) series at -oo contains unexpected I*pi term"
