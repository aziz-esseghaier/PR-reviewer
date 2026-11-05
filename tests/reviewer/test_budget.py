"""Tests for fitting a PR's text into a size limit (reviewer/budget.py).

Checks that a PR that fits is sent exactly as SWR-Bench writes it, that a big one loses whole
file diffs (source files kept first, smaller later diffs still let in), that a long
description is shortened when the diffs fit, and that the text never goes over the limit.

How to run (from the project folder):
    pytest tests/reviewer/test_budget.py
"""

import pytest

from pr_reviewer.benchmark.review_input import render_pr_details
from pr_reviewer.reviewer.budget import CUT_NOTE, LEFT_OUT_NOTE, fit_pr_details, is_test_file
from tests.reviewer.helpers import make_pr

SOURCE = ("pkg/cart.py", "a" * 3000)
TESTS = ("pkg/tests/test_cart.py", "b" * 3000)
DOCS = ("docs/cart.rst", "c" * 1000)


def test_a_pr_that_fits_is_sent_as_swr_bench_writes_it() -> None:
    """Check the text is untouched when it fits."""
    pr = make_pr([SOURCE])
    fitted = fit_pr_details(pr, 10_000)
    assert fitted.text == render_pr_details(pr)
    assert not fitted.trimmed
    assert fitted.full_chars == len(fitted.text)


def test_whole_diffs_are_left_out_source_files_first() -> None:
    """Check a big PR keeps its source diff, skips the test diff that doesn't fit, and still
    lets the smaller docs diff in; the left-out file is named at the end."""
    pr = make_pr([TESTS, DOCS, SOURCE])
    fitted = fit_pr_details(pr, 5_000)
    assert len(fitted.text) <= 5_000
    assert "a" * 3000 in fitted.text
    assert "c" * 1000 in fitted.text
    assert "b" * 10 not in fitted.text
    sha = pr.pr_commits[0].sha[:7]
    assert fitted.left_out == (f"pkg/tests/test_cart.py (commit {sha})",)
    assert fitted.text.endswith(f"{LEFT_OUT_NOTE}\n- pkg/tests/test_cart.py (commit {sha})\n")
    assert fitted.trimmed and not fitted.cut


def test_a_long_description_is_cut_when_the_diffs_fit() -> None:
    """Check every diff stays, and the description takes the room that's left."""
    pr = make_pr([SOURCE], description="d" * 20_000)
    fitted = fit_pr_details(pr, 8_000)
    assert len(fitted.text) <= 8_000
    assert "a" * 3000 in fitted.text
    assert CUT_NOTE in fitted.text
    assert fitted.left_out == ()
    assert fitted.cut


@pytest.mark.parametrize("limit", [600, 2_000, 4_000])
def test_the_text_never_goes_over_the_limit(limit: int) -> None:
    """Check the limit holds even when the description alone is too big."""
    pr = make_pr([SOURCE, TESTS, DOCS], description="d" * 20_000)
    assert len(fit_pr_details(pr, limit).text) <= limit


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/tests/test_io.py", True),
        ("tests/helpers.py", True),
        ("pkg/test_io.py", True),
        ("pkg/io_test.py", True),
        ("conftest.py", True),
        ("pkg/io.py", False),
        ("pkg/testing_utils.py", False),
        ("docs/test.rst", False),
    ],
)
def test_test_files_are_told_apart_by_path(path: str, expected: bool) -> None:
    """Check which paths count as tests.

    Args:
        path (str): A file path.
        expected (bool): Whether it's a test file.
    """
    assert is_test_file(path) is expected
