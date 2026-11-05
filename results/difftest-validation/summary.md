# Differential test runner: validation on known tests

- 36 cases on 28 dev PRs; each side's tests ran 3 times, at most 5 minutes a run
- Outcomes are from the base to the PR state. Expected: the PR's own tests fail→pass or pass→pass; related tests unchanged (pass→pass or fail→fail); a bug fix's tests fail at the PR state and pass at the fix.
- Skipped: tests the repo's own skip marks skipped at both sides (such as a missing optional package). They're not in "not run" and count as neither expected nor not.

## Outcomes by kind

| Kind | Cases | Tests | pass→fail | pass→pass | fail→pass | fail→fail | flaky | not run | skipped | As expected |
|---|---|---|---|---|---|---|---|---|---|---|
| pr_tests | 23 | 857 | 0 | 276 | 106 | 2 | 0 | 0 | 473 | 382 |
| fix_tests | 9 | 176 | 0 | 34 | 1 | 95 | 0 | 0 | 46 | 5 |
| related | 4 | 141 | 0 | 127 | 0 | 0 | 0 | 0 | 14 | 127 |

## Known bugs: the fix's tests

Caught: a test fails at the PR state and passes at the fix. Provable: a test also passes at the base, so the bug shows as pass→fail.

| PR | Issue | Fix | Tests | Caught | PR→fix outcomes | Provable from base |
|---|---|---|---|---|---|---|
| pydata__xarray-8246 | F.2 | 88155cd2ef | 45 | no | skipped 45 | 0 |
| scikit-learn__scikit-learn-22957 | F.2 | bbba528de2 | 9 | no | pass→pass 9 | 0 |
| scikit-learn__scikit-learn-22957 | F.2 | b3fac2aa06 | 10 | no | pass→pass 10 | 0 |
| scikit-learn__scikit-learn-23459 | F.4 | 06502921fd | 1 | yes | fail→pass 1 | 0 |
| scikit-learn__scikit-learn-23459 | F.4 | ab65a8988e | 2 | yes | fail→pass 2 | 0 |
| scikit-learn__scikit-learn-25108 | F.1 | 4344c491fb | 14 | yes | pass→pass 13, fail→pass 1 | 0 |
| scikit-learn__scikit-learn-27943 | F.5 | 151312de45 | 93 | no | not run 93 | 0 |
| sympy__sympy-25735 | F.4 | 30ce30d019 | 1 | no | pass→pass 1 | 0 |
| sympy__sympy-26233 | F.2 | de7bdfda60 | 1 | yes | fail→pass 1 | 0 |

Caught 4 of 9 known bugs; 0 provable as pass→fail from the base.

## Not as expected

- pr_tests, pydata__xarray-8246: xarray/tests/test_rolling.py::TestDatasetRolling::test_raise_no_warning_dask_rolling_assert_close[numpy-max-2]: fail_fail
- pr_tests, pydata__xarray-8246: xarray/tests/test_rolling.py::TestDatasetRolling::test_raise_no_warning_dask_rolling_assert_close[numpy-mean-2]: fail_fail
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/datasets/tests/test_svmlight_format.py::test_load_invalid_file2: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[False-None-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-None-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/model_selection/tests/test_split.py::test_array_api_train_test_split[True-stratify1-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_asarray_with_order[numpy.array_api]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_asarray_with_order[numpy]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-False-10.0-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[None-True-2.5-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight2-False-5.5-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight3-True-2.75-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight4-False-30.0-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_array_api.py::test_weighted_sum[sample_weight5-True-3.0-torch-mps-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[cupy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[cupy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[numpy-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[numpy.array_api-None-None]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[torch-cpu-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[torch-cpu-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[torch-cuda-float32]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[torch-cuda-float64]: not_run
- fix_tests, scikit-learn__scikit-learn-27943: sklearn/utils/tests/test_multiclass.py::test_is_multilabel_array_api_compliance[torch-mps-float32]: not_run

## Files or fixes that gave no case

- astropy__astropy-16735: astropy/io/ascii/tests/test_read.py: no test function changed
- astropy__astropy-16735: fix e1551d8c21 (F.4): changes no test
- scikit-learn__scikit-learn-22454: fix bdeff65e5c (F.5): changes no test
- scikit-learn__scikit-learn-25108: sklearn/tests/test_public_functions.py: no test function changed
- scikit-learn__scikit-learn-27943: sklearn/decomposition/tests/test_sparse_pca.py: no test function changed
- scikit-learn__scikit-learn-27943: sklearn/decomposition/tests/test_pca.py: no test function changed
- scikit-learn__scikit-learn-27943: sklearn/metrics/tests/test_common.py: no test function changed
- scikit-learn__scikit-learn-31106: sklearn/model_selection/tests/test_search.py: no test function changed
- scikit-learn__scikit-learn-31106: sklearn/tests/test_pipeline.py: no test function changed
