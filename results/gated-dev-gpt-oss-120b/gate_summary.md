# Gate: candidates-gpt-oss-120b checked by evidence-tools-gpt-oss-120b

- PRs: 68 (26 with known issues, 42 clean)
- Policy: tier <= 1, confidence >= 0.0, one comment per place: yes, at most 3 comments per PR
- Candidates: 204; by tier: 1: 5, 2: 4, 3: 20, 4: 175
- Passing tier and confidence: 5; kept after one per place and the cap: 5 (tier 1: 5)
- PRs with a comment: 5, of which clean: 0
- Candidates whose check failed (counted as tier 4): 0

## Comments kept

- pydata__xarray-6939 f1 (tier 1, confidence 0.90): xarray/core/variable.py:2143: Error message string does not interpolate the method name
- scikit-learn__scikit-learn-25148 f1 (tier 1, confidence 0.60): sklearn/multiclass.py:1025: predict_proba does not validate input X
- scikit-learn__scikit-learn-27943 f2 (tier 1, confidence 0.65): sklearn/linear_model/_quantile.py:150: Changing default solver may alter model behavior for existing code
- sphinx-doc__sphinx-12491 f1 (tier 1, confidence 0.85): sphinx/util/nodes.py:344: _lift_sections may raise AttributeError when state_parent.parent is None
- sympy__sympy-26233 f7 (tier 1, confidence 0.40): sympy/functions/special/error_functions.py:2078: I*pi term placement may affect the series sign
