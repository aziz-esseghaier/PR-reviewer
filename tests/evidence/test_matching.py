"""Tests for matching a generated test's run against its claim (evidence/matching.py).

Builds outcomes by hand, so no Docker is needed. Checks each rule: a proof needs the claimed
error (or a subclass) through a changed line and not raised by the test itself; a wrong value
needs the test to run a changed line; new code counts only when the base fails because a name
the PR adds is missing; and broken tests (skipped, failing to load, failing on the base) go
back to the test writer with the error.

How to run (from the project folder):
    pytest tests/evidence/test_matching.py
"""

import pytest

from pr_reviewer.difftest.outcomes import DiffOutcome, SideResult
from pr_reviewer.difftest.reports import Failure
from pr_reviewer.evidence.matching import judge, missing_new_name
from tests.evidence.helpers import TEST, failure, outcome, side

CHANGED = {"pkg/cart.py": {10, 11, 12}}
NEW = {"average", "Cart"}


def check(found: DiffOutcome, expected: str = "AttributeError") -> tuple[str, str | None, str]:
    """Judge an outcome and give the verdict, proof kind and reason.

    Args:
        found (DiffOutcome): The outcome.
        expected (str): The claimed exception.

    Returns:
        tuple[str, str | None, str]: The verdict, the proof kind and the reason.
    """
    match = judge(found, expected, TEST, CHANGED, NEW)
    assert match is not None
    return match.verdict, match.proof, match.reason


def test_the_claimed_error_through_a_changed_line_is_a_proof() -> None:
    """Check the plain proof, and the facts the match records for the evidence."""
    match = judge(
        outcome(side("pass"), side("fail", failure())), "AttributeError", TEST, CHANGED, NEW
    )
    assert match is not None
    assert (match.verdict, match.proof) == ("proof", "executed")
    assert (match.base_result, match.pr_result, match.failure) == (
        "passed",
        "failed",
        "AttributeError",
    )
    assert match.through == ("pkg/cart.py:11",)
    assert match.failure_classes == ("Exception", "BaseException")


def test_a_subclass_of_the_claimed_error_counts() -> None:
    """Check that a repo's own subclass of ValueError proves a ValueError claim."""
    sub = failure("CartError", classes=("CartError", "ValueError", "Exception"))
    assert check(outcome(side("pass"), side("fail", sub)), "ValueError")[:2] == (
        "proof",
        "executed",
    )


@pytest.mark.parametrize(
    ("new", "reason"),
    [
        (failure("TypeError"), "the PR run failed with TypeError"),
        (failure(frames=(f"{TEST}:5", "pkg/cart.py:30")), "didn't come through a line the PR"),
        (failure(frames=("pkg/cart.py:11", f"{TEST}:9")), "the test's own code raised"),
    ],
    ids=["another error", "not through a changed line", "raised by the test"],
)
def test_a_failure_that_doesnt_match_proves_nothing(new: Failure, reason: str) -> None:
    """Check the three ways a PR failure can miss the claim."""
    verdict, proof, why = check(outcome(side("pass"), side("fail", new)))
    assert (verdict, proof) == ("no_proof", None)
    assert reason in why


def test_a_changed_line_of_the_test_file_itself_doesnt_count() -> None:
    """Check the generated test's own lines never count as the PR's changed lines."""
    found = failure(frames=(f"{TEST}:11",))
    match = judge(
        outcome(side("pass"), side("fail", found)),
        "AttributeError",
        TEST,
        {**CHANGED, TEST: {11}},
        NEW,
    )
    assert match is not None and match.verdict == "no_proof"


def test_a_wrong_value_needs_the_test_to_run_a_changed_line() -> None:
    """Check AssertionError claims: proof only when the PR run went through changed lines."""
    wrong = failure("AssertionError", frames=(f"{TEST}:8",), message="assert 1 == 2")
    ran = outcome(side("pass"), side("fail", wrong, hits=("pkg/cart.py:11",)))
    assert check(ran, "AssertionError")[:2] == ("proof", "executed")
    missed = outcome(side("pass"), side("fail", wrong))
    assert check(missed, "AssertionError")[0] == "no_proof"
    other = outcome(side("pass"), side("fail", failure(), hits=("pkg/cart.py:11",)))
    assert check(other, "AssertionError")[0] == "no_proof"


def test_new_code_counts_when_the_base_only_misses_the_new_name() -> None:
    """Check the new-code proof, and that a base failure for another reason is a broken test."""
    missing = failure(
        "ImportError",
        frames=(f"{TEST}:1",),
        phase="collect",
        message="ImportError: cannot import name 'average' from 'pkg.cart'",
    )
    assert check(outcome(side("fail", missing), side("fail", failure()))) == (
        "proof",
        "new_code",
        "new code fails as claimed",
    )
    old_name = missing.model_copy(update={"message": "cannot import name 'total' from 'pkg.cart'"})
    assert check(outcome(side("fail", old_name), side("fail", failure())))[0] == "broken"
    passes = outcome(side("fail", missing), side("pass"))
    assert check(passes)[:2] == ("no_proof", None)


CAUGHT = """\
import pytest
from pkg.cart import Cart


def test_x():
    with pytest.raises(ValueError) as error:
        Cart(None).average()
    assert "average" in str(error.value)
"""

MATCHED = """\
import pytest
from pkg.cart import Cart


def test_x():
    with pytest.raises(ValueError, match="average"):
        Cart(None).average()
"""

UNRELATED = """\
import pytest
from pkg.cart import Cart


def test_x():
    with pytest.raises(ValueError) as error:
        Cart(None).average()
    assert Cart([1]).total == 2
"""


def caught_failure(line: int) -> Failure:
    """Build an AssertionError raised at a line of the test file.

    Args:
        line (int): The line.

    Returns:
        Failure: The failure.
    """
    return failure("AssertionError", frames=(f"{TEST}:{line}",), message="assert 'average' in ...")


@pytest.mark.parametrize(
    ("code", "line", "hits", "verdict"),
    [
        (CAUGHT, 8, ("pkg/cart.py:11",), "proof"),
        (MATCHED, 7, ("pkg/cart.py:11",), "proof"),
        (UNRELATED, 8, ("pkg/cart.py:11",), "no_proof"),
        (CAUGHT, 8, (), "no_proof"),
        (CAUGHT.replace("ValueError", "Exception"), 8, ("pkg/cart.py:11",), "no_proof"),
    ],
    ids=[
        "assert on the caught error",
        "pytest.raises with match",
        "assert on something else",
        "no changed line ran",
        "caught a broader error",
    ],
)
def test_a_failed_check_on_the_caught_claimed_error_is_a_proof(
    code: str, line: int, hits: tuple[str, ...], verdict: str
) -> None:
    """Check the shape: a check on the error caught with pytest.raises of the claimed type."""
    found = outcome(side("pass"), side("fail", caught_failure(line), hits=hits))
    match = judge(found, "ValueError", TEST, CHANGED, NEW, code)
    assert match is not None and match.verdict == verdict
    if verdict == "proof":
        assert (match.proof, match.caught, match.failure) == (
            "executed",
            "ValueError",
            "AssertionError",
        )
        assert "caught ValueError" in match.reason


def test_a_name_missing_on_both_commits_is_a_broken_test() -> None:
    """Check a package missing at both sides isn't new code, even when the PR names it."""
    missing = failure(
        "ModuleNotFoundError",
        frames=(f"{TEST}:1",),
        message="ModuleNotFoundError: No module named 'average'",
    )
    match = judge(
        outcome(side("fail", missing), side("fail", missing)), "AttributeError", TEST, CHANGED, NEW
    )
    assert match is not None and match.verdict == "broken"


def test_new_code_never_proves_a_wrong_value() -> None:
    """Check a wrong-value claim on new code: no old behavior checks the expected value."""
    missing = failure("AttributeError", message="module 'pkg.cart' has no attribute 'average'")
    wrong = failure("AssertionError", frames=(f"{TEST}:8",), message="assert 1 == 2")
    found = outcome(side("fail", missing), side("fail", wrong, hits=("pkg/cart.py:11",)))
    assert check(found, "AssertionError")[0] == "no_proof"


@pytest.mark.parametrize(
    ("message", "missing"),
    [
        ("cannot import name 'average' from 'pkg.cart'", True),
        ("No module named 'pkg.average'", True),
        ("'Cart' object has no attribute 'average'", True),
        ("name 'average' is not defined", True),
        ("Cart.__init__() got an unexpected keyword argument 'average'", True),
        ("module 'numpy' has no attribute 'foo'", False),
        ("'NoneType' object has no attribute 'Cart'", True),
        ("division by zero", False),
    ],
)
def test_missing_new_names_are_read_from_the_message(message: str, missing: bool) -> None:
    """Check which messages say that a name the PR adds is missing."""
    assert missing_new_name(failure("ImportError", message=message), NEW) is missing


def test_a_test_failing_on_the_base_commit_is_broken() -> None:
    """Check that a failing base run goes back to the test writer with the base's error."""
    base = failure("KeyError", frames=(f"{TEST}:7",), message="KeyError: 'x'")
    match = judge(
        outcome(side("fail", base), side("fail", base)), "AttributeError", TEST, CHANGED, NEW
    )
    assert match is not None and match.verdict == "broken"
    assert "On the base commit" in match.feedback and "KeyError" in match.feedback


def test_a_test_that_doesnt_load_is_broken() -> None:
    """Check a load error on both sides (such as a bad import) is a broken test."""
    bad = failure(
        "ModuleNotFoundError",
        frames=(f"{TEST}:1",),
        phase="collect",
        message="No module named 'nope'",
    )
    match = judge(
        outcome(side("fail", bad), side("fail", bad)), "AttributeError", TEST, CHANGED, NEW
    )
    assert match is not None and match.verdict == "broken"
    assert "while loading the file" in match.feedback


@pytest.mark.parametrize("reason", ["skipped", "not collected"])
def test_a_skipped_or_missing_test_is_broken(reason: str) -> None:
    """Check a test that never ran gets a note to write a plain test function."""
    found = outcome(side("not_run", reason=reason), side("not_run", reason=reason))
    match = judge(found, "AttributeError", TEST, CHANGED, NEW)
    assert match is not None and match.verdict == "broken"
    assert "test_" in match.feedback


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        (side("not_run", reason="timeout"), side("pass"), "not run (timeout)"),
        (side("pass"), side("flaky", failure()), "flaky"),
        (side("pass"), side("pass"), "passes on both commits"),
        (side("pass"), side("fail", failure(phase="setup")), "failed outside the test"),
    ],
    ids=["timeout", "flaky", "passes on both", "setup error only on the PR"],
)
def test_runs_that_prove_nothing(old: SideResult, new: SideResult, reason: str) -> None:
    """Check timeouts, flaky runs, passing runs and PR-only setup errors aren't proofs."""
    verdict, proof, why = check(outcome(old, new))
    assert (verdict, proof) == ("no_proof", None)
    assert reason in why


def test_no_outcome_gives_no_verdict() -> None:
    """Check judge() gives None when the run reported no test."""
    assert judge(None, "AttributeError", TEST, CHANGED, NEW) is None
