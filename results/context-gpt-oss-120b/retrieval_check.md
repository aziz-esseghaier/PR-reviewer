# Retrieval check: context-gpt-oss-120b

57 known issues in 26 PRs. For each, the functions its fix commit changed (outside tests) show where the problem was; the check finds the best place the model could have seen one of them. No model is called; fix commits are used only to measure.

| Where | All | F | E | Meaning |
|---|---|---|---|---|
| in_diff | 23 | 8 | 15 | in the PR's diff (a function the PR changed) |
| in_context | 0 | 0 | 0 | in the context the request held |
| left_out | 0 | 0 | 0 | found by retrieval, but left out for lack of room |
| not_found | 1 | 1 | 0 | not found by retrieval |
| no_function | 28 | 15 | 13 | the fix changed no function outside tests (tests, module-level lines or new files only) |
| no_fix | 5 | 5 | 0 | SWR-Bench records no fix commit |
| fix_unreadable | 0 | 0 | 0 | git couldn't read the fix commit |

Of the 1 issues whose fix changed code outside the PR's diff, the context held 0 (0%); 0 were found but didn't fit, and 1 weren't found.

## Issues

| PR | # | Category | Where | Fix's functions |
|---|---|---|---|---|
| astropy__astropy-12747 | 1 | F.5 | no_function |  |
| astropy__astropy-16735 | 1 | F.4 | in_diff | `astropy/io/ascii/cds.py:_is_section_delimiter` |
| astropy__astropy-16735 | 2 | E.1.1 | in_diff | `astropy/io/ascii/cds.py:_is_section_delimiter` |
| astropy__astropy-16735 | 3 | E.1.1 | in_diff | `astropy/io/ascii/cds.py:_is_section_delimiter` |
| astropy__astropy-17268 | 1 | E.1.1 | in_diff | `astropy/table/table.py:Table.show_in_notebook` |
| astropy__astropy-17268 | 2 | F.5 | no_function |  |
| matplotlib__matplotlib-24674 | 1 | F.6 | no_fix |  |
| matplotlib__matplotlib-25737 | 1 | E.1.1 | no_function |  |
| matplotlib__matplotlib-25737 | 2 | F.5 | no_function |  |
| matplotlib__matplotlib-27750 | 1 | F.5 | no_function |  |
| pydata__xarray-6260 | 1 | E.3.2 | no_function |  |
| pydata__xarray-6260 | 2 | F.6 | no_fix |  |
| pydata__xarray-6939 | 1 | F.2 | in_diff | `xarray/core/variable.py:Variable.quantile` |
| pydata__xarray-8246 | 1 | E.1.1 | no_function |  |
| pydata__xarray-8246 | 2 | E.1.1 | no_function |  |
| pydata__xarray-8246 | 3 | F.2 | no_function |  |
| pylint-dev__pylint-7257 | 1 | F.6 | no_fix |  |
| pylint-dev__pylint-7918 | 1 | F.5 | no_function |  |
| pylint-dev__pylint-8706 | 1 | F.2 | no_function |  |
| pylint-dev__pylint-8706 | 2 | E.1.1 | no_function |  |
| pylint-dev__pylint-8706 | 3 | F.2 | no_function |  |
| pytest-dev__pytest-12232 | 1 | E.1.2 | in_diff | `src/_pytest/python_api.py:ApproxNumpy._repr_compare` |
| pytest-dev__pytest-12232 | 2 | E.1.1 | in_diff | `src/_pytest/python_api.py:ApproxNumpy._repr_compare` |
| pytest-dev__pytest-12232 | 3 | F.4 | in_diff | `src/_pytest/python_api.py:ApproxNumpy._repr_compare` |
| scikit-learn__scikit-learn-22454 | 1 | F.5 | no_function |  |
| scikit-learn__scikit-learn-22729 | 1 | F.2 | no_function |  |
| scikit-learn__scikit-learn-22729 | 2 | E.3.2 | no_function |  |
| scikit-learn__scikit-learn-22957 | 1 | F.2 | no_function |  |
| scikit-learn__scikit-learn-22957 | 2 | F.2 | no_function |  |
| scikit-learn__scikit-learn-23459 | 1 | F.4 | in_diff | `sklearn/utils/_param_validation.py:StrOptions.__init__` |
| scikit-learn__scikit-learn-23459 | 2 | F.4 | in_diff | `sklearn/utils/_param_validation.py:StrOptions.__init__` |
| scikit-learn__scikit-learn-25108 | 1 | E.2 | in_diff | `sklearn/metrics/_ranking.py:roc_curve` |
| scikit-learn__scikit-learn-25108 | 2 | F.1 | in_diff | `sklearn/metrics/_ranking.py:roc_curve` |
| scikit-learn__scikit-learn-25108 | 3 | E.1.1 | in_diff | `sklearn/metrics/_ranking.py:roc_curve` |
| scikit-learn__scikit-learn-25108 | 4 | E.2 | no_function |  |
| scikit-learn__scikit-learn-25148 | 1 | E.3.2 | in_diff | `sklearn/multiclass.py:OutputCodeClassifier.predict_proba` |
| scikit-learn__scikit-learn-25148 | 2 | E.3.2 | in_diff | `sklearn/multiclass.py:OutputCodeClassifier.predict_proba` |
| scikit-learn__scikit-learn-25148 | 3 | E.1.1 | in_diff | `sklearn/multiclass.py:OutputCodeClassifier.predict_proba` |
| scikit-learn__scikit-learn-25148 | 4 | E.3.1 | no_function |  |
| scikit-learn__scikit-learn-25148 | 5 | E.1.1 | in_diff | `sklearn/multiclass.py:OutputCodeClassifier.predict_proba` |
| scikit-learn__scikit-learn-25148 | 6 | E.1.1 | in_diff | `sklearn/multiclass.py:OutputCodeClassifier.predict_proba` |
| scikit-learn__scikit-learn-25148 | 7 | E.1.1 | in_diff | `sklearn/multiclass.py:OutputCodeClassifier.predict_proba` |
| scikit-learn__scikit-learn-25148 | 8 | E.1.1 | no_function |  |
| scikit-learn__scikit-learn-25148 | 9 | F.6 | no_fix |  |
| scikit-learn__scikit-learn-26928 | 1 | F.5 | no_function |  |
| scikit-learn__scikit-learn-26928 | 2 | E.1.1 | no_function |  |
| scikit-learn__scikit-learn-27943 | 1 | F.5 | not_found | `setup.py:setup_package`, `sklearn/utils/_array_api.py:_ArrayAPIWrapper.take`, `sklearn/utils/_testing.py:_array_api_for_tests` and 1 more |
| scikit-learn__scikit-learn-28564 | 1 | E.2 | in_diff | `sklearn/datasets/_samples_generator.py:make_low_rank_matrix` |
| scikit-learn__scikit-learn-28564 | 2 | F.2 | in_diff | `sklearn/datasets/_samples_generator.py:make_low_rank_matrix` |
| scikit-learn__scikit-learn-30581 | 1 | F.2 | no_function |  |
| sphinx-doc__sphinx-12491 | 1 | F.6 | no_fix |  |
| sympy__sympy-25735 | 1 | E.3.2 | no_function |  |
| sympy__sympy-25735 | 2 | F.4 | no_function |  |
| sympy__sympy-25735 | 3 | E.3.2 | no_function |  |
| sympy__sympy-26233 | 1 | F.2 | in_diff | `sympy/functions/special/error_functions.py:Ci._eval_aseries` |
| sympy__sympy-26233 | 2 | E.3.1 | in_diff | `sympy/functions/special/error_functions.py:Ci._eval_aseries` |
| sympy__sympy-26233 | 3 | E.2 | no_function |  |
