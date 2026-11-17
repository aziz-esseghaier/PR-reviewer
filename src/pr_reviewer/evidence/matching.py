"""Matching a generated test's run against its claim: proof, broken test, or no proof.

The differential runner gives the test's verdict at the base commit and at the PR commit, with
each failure's exception, its parent classes and its traceback, and the PR's changed lines the
test ran. judge() reads them with these rules:

- Proof ("executed"): the test passes on the base commit and fails on the PR commit with the
  claimed exception or a subclass of it. For an exception claim, the traceback must go through
  a line the PR changed (outside the test file), and the error must not be raised by the test's
  own code: an exception raised from the test file proves nothing about the PR. For a wrong
  value claim (AssertionError), the test's own assert fails, so the test must instead have run a
  line the PR changed.
- Proof ("executed") through a caught error: some claims name an error the code raises before
  and after the PR, when the bug is the error's content (such as a message missing a name). A
  test then catches the error with `with pytest.raises(ValueError) as error:` and checks it
  afterwards (`assert "x" in str(error.value)`, or `pytest.raises(..., match=...)`), and that
  check fails with AssertionError. It counts when our code, reading the test with Python's
  parser, finds that the failed assert reads an error caught with pytest.raises of the claimed
  type, and the test passes on the base commit, fails on the PR and ran a line the PR changed.
  The assert can only run after pytest.raises caught the claimed error, so the proof still
  holds it.
- Proof ("new_code"): for code the PR adds, the base run can't pass, since the code isn't
  there. It counts when the base run fails only because a name the PR adds is missing (its
  error says which name: "cannot import name 'x'", "has no attribute 'x'", "unexpected keyword
  argument 'x'"...) and the PR run fails with the claimed exception through a changed line, as
  above. Never for a wrong value: with no old behavior to compare with, nothing checks the
  expected value.
- Broken: the test itself didn't work: it was skipped or never collected, failed to load or
  set up (a bad import or fixture), or already fails on the base commit for a reason other
  than missing new code. The test writer gets the error and may fix the test.
- No proof: everything else: it passes on both commits, is flaky, timed out, or fails on the PR
  with another error or without going through the changed lines.

Imported by other modules; the file isn't run directly.
"""

import ast
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from pr_reviewer.core.schemas import ProofKind, RunResult
from pr_reviewer.difftest.outcomes import DiffOutcome, SideResult
from pr_reviewer.difftest.reports import Failure

# Raised whenever the rules below change, so runs judged with other rules are told apart.
# 2: a failed check on an error caught with pytest.raises of the claimed type counts.
# 3: the same missing-name error on both commits is a broken test, not new code; only names
#    the PR defines count as new.
MATCHING_VERSION = 3

# How a test run ended for the evidence agent.
Verdict = Literal["proof", "broken", "no_proof"]

# What the errors a missing name raises say, with the missing name in the first group: a
# function, class or module the PR adds, or a keyword argument it adds.
MISSING_NAME = (
    re.compile(r"cannot import name '(\w+)'"),
    re.compile(r"No module named '([\w.]+)'"),
    re.compile(r"has no attribute '(\w+)'"),
    re.compile(r"name '(\w+)' is not defined"),
    re.compile(r"unexpected keyword argument '(\w+)'"),
)

# Most characters of an error shown to the test writer.
ERROR_CHARS = 1200


@dataclass(frozen=True)
class Match:
    """The verdict on one test run.

    Attributes:
        verdict (Verdict): "proof", "broken" or "no_proof".
        proof (ProofKind | None): "executed" or "new_code" for a proof, else None.
        reason (str): Why, in a few words, for reports.
        feedback (str): For a broken test, what went wrong, to show the test writer.
        base_result (RunResult): How the test ended on the base commit.
        pr_result (RunResult): How it ended on the PR commit.
        failure (str | None): The PR run's exception type, if it failed.
        failure_classes (tuple[str, ...]): That exception's parent classes.
        through (tuple[str, ...]): The changed lines the PR run's traceback went through.
        caught (str | None): For a proof through a caught error, the claimed error the test
            caught before its check failed; else None.
    """

    verdict: Verdict
    proof: ProofKind | None
    reason: str
    feedback: str
    base_result: RunResult
    pr_result: RunResult
    failure: str | None
    failure_classes: tuple[str, ...]
    through: tuple[str, ...]
    caught: str | None = None


def same_name(seen: str, expected: str) -> bool:
    """Compare two exception names by their last part.

    Args:
        seen (str): Such as "json.decoder.JSONDecodeError".
        expected (str): Such as "JSONDecodeError".

    Returns:
        bool: True when the class names are equal.
    """
    return seen.rsplit(".", 1)[-1] == expected.rsplit(".", 1)[-1]


def matches(failure: Failure, expected: str) -> bool:
    """Tell whether a failure's exception is the expected one or a subclass of it.

    Args:
        failure (Failure): The failure.
        expected (str): The claimed exception.

    Returns:
        bool: True when its type, or one of its parent classes, has the expected name.
    """
    names = [failure.exc_type, *failure.classes]
    return any(name and same_name(name, expected) for name in names)


def result_of(side: SideResult) -> RunResult:
    """Give how a test ended at one side, in the evidence's terms.

    Args:
        side (SideResult): Its verdict there.

    Returns:
        RunResult: "passed"; "failed" for a failure while the test ran; "error" for anything
        else (a load or setup error, a flaky or unrun test).
    """
    if side.verdict == "pass":
        return "passed"
    if side.verdict == "fail" and side.failure is not None and side.failure.phase == "call":
        return "failed"
    return "error"


def _place(frame: str) -> tuple[str, int] | None:
    """Split a traceback place into its file and line.

    Args:
        frame (str): Such as "pkg/a.py:12".

    Returns:
        tuple[str, int] | None: ("pkg/a.py", 12), or None when it has no line number.
    """
    path, sep, line = frame.rpartition(":")
    if not sep or not line.isdigit():
        return None
    return path, int(line)


def through_changed(
    failure: Failure, changed: Mapping[str, Collection[int]], test_path: str
) -> tuple[str, ...]:
    """List the changed lines a failure's traceback went through, outside the test file.

    Args:
        failure (Failure): The failure, with its traceback frames.
        changed (Mapping[str, Collection[int]]): The PR's changed lines, by file.
        test_path (str): The generated test's file, whose own lines never count.

    Returns:
        tuple[str, ...]: Such as ("pkg/a.py:12",), in traceback order.
    """
    found = []
    for frame in failure.frames:
        place = _place(frame)
        if place is None or place[0] == test_path:
            continue
        if place[1] in changed.get(place[0], ()):
            found.append(frame)
    return tuple(found)


def raised_by_test(failure: Failure, test_path: str) -> bool:
    """Tell whether the test's own code raised the error (its innermost frame is the test).

    Args:
        failure (Failure): The failure.
        test_path (str): The generated test's file.

    Returns:
        bool: True when the innermost frame is in the test file.
    """
    if not failure.frames:
        return failure.where.rpartition(":")[0] == test_path
    place = _place(failure.frames[-1])
    return place is not None and place[0] == test_path


def missing_new_name(failure: Failure, new_names: Collection[str]) -> bool:
    """Tell whether a failure is only a name the PR adds being missing.

    Args:
        failure (Failure): The base run's failure.
        new_names (Collection[str]): The names on the PR's added lines.

    Returns:
        bool: True when its message says a name is missing (such as "cannot import name
        'total'", "has no attribute 'total'" or "unexpected keyword argument 'total'") and
        that name is one the PR adds.
    """
    for pattern in MISSING_NAME:
        for found in pattern.findall(failure.message):
            if found.rsplit(".", 1)[-1] in new_names:
                return True
    return False


def describe_failure(failure: Failure | None, side: str) -> str:
    """Write a failure for the test writer.

    Args:
        failure (Failure | None): The failure.
        side (str): Which commit, such as "the base commit".

    Returns:
        str: Its phase, type, message and place, cut to ERROR_CHARS.
    """
    if failure is None:
        return f"The test didn't pass on {side}."
    phase = {"collect": "while loading the file", "setup": "while setting up its fixtures"}
    when = phase.get(failure.phase, "while running")
    text = (
        f"On {side}, {when}: {failure.exc_type or 'an error'} at {failure.where or 'unknown'}: "
        f"{failure.message}"
    )
    return text[:ERROR_CHARS]


def _match(
    verdict: Verdict,
    reason: str,
    outcome: DiffOutcome,
    *,
    proof: ProofKind | None = None,
    feedback: str = "",
    through: tuple[str, ...] = (),
    caught: str | None = None,
) -> Match:
    """Build a Match with the run's facts.

    Args:
        verdict (Verdict): The verdict.
        reason (str): Why.
        outcome (DiffOutcome): The test's outcome.
        proof (ProofKind | None): The proof's kind, for a proof.
        feedback (str): What to tell the test writer, for a broken test.
        through (tuple[str, ...]): The changed lines the PR run went through.
        caught (str | None): The claimed error the test caught, for a proof through it.

    Returns:
        Match: The verdict with the run's results and failure.
    """
    failure = outcome.new.failure
    name = failure.exc_type if failure is not None and failure.exc_type else None
    return Match(
        verdict=verdict,
        proof=proof,
        reason=reason,
        feedback=feedback,
        base_result=result_of(outcome.old),
        pr_result=result_of(outcome.new),
        failure=name,
        failure_classes=tuple(failure.classes[1:]) if failure is not None else (),
        through=through,
        caught=caught,
    )


def _raises_of(node: ast.expr, expected: str) -> ast.Call | None:
    """Give a `pytest.raises(...)` call of the claimed error, if a with-item is one.

    Args:
        node (ast.expr): The with-item's expression.
        expected (str): The claimed exception.

    Returns:
        ast.Call | None: The call, when it's pytest.raises (or raises) whose first argument
        names the claimed exception; else None.
    """
    if not isinstance(node, ast.Call) or not node.args:
        return None
    func = node.func
    is_raises = (isinstance(func, ast.Attribute) and func.attr == "raises") or (
        isinstance(func, ast.Name) and func.id == "raises"
    )
    first = node.args[0]
    name = (
        first.id
        if isinstance(first, ast.Name)
        else first.attr
        if isinstance(first, ast.Attribute)
        else ""
    )
    return node if is_raises and name and same_name(name, expected) else None


def expects_claimed_error(test_code: str, expected: str) -> bool:
    """Tell whether a test expects the claimed error instead of failing with it.

    `with pytest.raises(<expected>):` passes when the error is raised, so the test passes
    where the bug is and fails where it isn't: the opposite of a proof. Catching it is fine
    only to check what it says: with match=..., or an assert that reads the caught error
    (`as error`, then `assert ... error.value ...`).

    Args:
        test_code (str): The generated test's code.
        expected (str): The claimed exception.

    Returns:
        bool: True when some pytest.raises of the claimed error checks nothing about it.
    """
    try:
        tree = ast.parse(test_code)
    except (SyntaxError, ValueError):
        return False
    read = {
        name.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        for name in ast.walk(node.test)
        if isinstance(name, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _raises_of(node, expected) and len(node.args) > 1:
            return True  # pytest.raises(Error, function, ...): the old call form
        if not isinstance(node, ast.With | ast.AsyncWith):
            continue
        for item in node.items:
            call = _raises_of(item.context_expr, expected)
            if call is None or any(k.arg == "match" for k in call.keywords):
                continue
            bound = item.optional_vars
            if not (isinstance(bound, ast.Name) and bound.id in read):
                return True
    return False


def checks_caught_error(test_code: str, line: int, expected: str) -> bool:
    """Tell whether a failed line checks an error the test caught with pytest.raises.

    It does when the line is an assert that reads the name bound by
    `with pytest.raises(<expected>) as name:`, or when it's within such a with block that
    passes match=..., whose message check failed.

    Args:
        test_code (str): The generated test's code.
        line (int): The test file's line the AssertionError came from.
        expected (str): The claimed exception.

    Returns:
        bool: True for a failed check on the caught claimed error.
    """
    try:
        tree = ast.parse(test_code)
    except (SyntaxError, ValueError):
        return False
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With | ast.AsyncWith):
            continue
        for item in node.items:
            call = _raises_of(item.context_expr, expected)
            if call is None:
                continue
            if isinstance(item.optional_vars, ast.Name):
                names.add(item.optional_vars.id)
            end = node.end_lineno or node.lineno
            if any(k.arg == "match" for k in call.keywords) and node.lineno <= line <= end:
                return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and node.lineno <= line <= (node.end_lineno or node.lineno):
            used = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            return bool(used & names)
    return False


def _failed_line(failure: Failure, test_path: str) -> int | None:
    """Give the test file's line an error came from: its innermost frame in the test file.

    Args:
        failure (Failure): The failure.
        test_path (str): The generated test's file.

    Returns:
        int | None: The line, or None when no frame is in the test file.
    """
    frames = failure.frames or ((failure.where,) if failure.where else ())
    for frame in reversed(frames):
        place = _place(frame)
        if place is not None and place[0] == test_path:
            return place[1]
    return None


def _pr_side(
    outcome: DiffOutcome,
    expected: str,
    test_path: str,
    changed: Mapping[str, Collection[int]],
) -> tuple[bool, str, tuple[str, ...]]:
    """Check the PR run's failure against an exception claim.

    Args:
        outcome (DiffOutcome): The test's outcome.
        expected (str): The claimed exception.
        test_path (str): The generated test's file.
        changed (Mapping[str, Collection[int]]): The PR's changed lines.

    Returns:
        tuple[bool, str, tuple[str, ...]]: Whether it matches, why not, and the changed lines
        its traceback went through.
    """
    failure = outcome.new.failure
    if failure is None or result_of(outcome.new) != "failed":
        return False, "the PR run didn't fail while the test ran", ()
    if not matches(failure, expected):
        return False, f"the PR run failed with {failure.exc_type or 'another error'}", ()
    through = through_changed(failure, changed, test_path)
    if raised_by_test(failure, test_path):
        return False, "the test's own code raised the error", through
    if not through:
        return False, "the error didn't come through a line the PR changed", through
    return True, "", through


def judge(
    outcome: DiffOutcome | None,
    expected: str,
    test_path: str,
    changed: Mapping[str, Collection[int]],
    new_names: Collection[str],
    test_code: str = "",
) -> Match | None:
    """Decide whether a test run proves its claim, the test is broken, or neither.

    Args:
        outcome (DiffOutcome | None): The test's outcome; None when the run gave none.
        expected (str): The claimed exception, such as "AttributeError" or "AssertionError".
        test_path (str): The generated test's file.
        changed (Mapping[str, Collection[int]]): The PR's changed lines, by file.
        new_names (Collection[str]): The names on the PR's added lines.
        test_code (str): The generated test's code, to recognize a check on a caught error.

    Returns:
        Match | None: The verdict, or None when there's no outcome to judge.
    """
    if outcome is None:
        return None
    old, new = outcome.old, outcome.new
    wrong_value = same_name(expected, "AssertionError")
    if outcome.outcome == "not_run":
        reasons = {old.reason, new.reason}
        if reasons & {"skipped", "not collected"}:
            text = "skipped" if "skipped" in reasons else "never collected (no test found)"
            feedback = (
                f"The test was {text}. Write one test function whose name starts with test_, "
                "with no skip mark and nothing that skips it."
            )
            return _match("broken", f"test {text}", outcome, feedback=feedback)
        reason = next((r for r in (old.reason, new.reason) if r), "unknown")
        return _match("no_proof", f"not run ({reason})", outcome)
    if outcome.outcome == "flaky":
        return _match("no_proof", "flaky", outcome)
    if (
        new.failure is not None
        and new.failure.phase in ("collect", "setup")
        and old.verdict != "pass"
    ):
        feedback = describe_failure(new.failure, "both commits")
        return _match("broken", f"test broken ({new.failure.phase})", outcome, feedback=feedback)
    if old.verdict == "fail":
        base_failure = old.failure
        # The same error on the PR means the name is missing there too: not new code, but a
        # test needing something that isn't there (such as a package that isn't installed).
        same_on_pr = (
            base_failure is not None
            and new.failure is not None
            and new.failure.exc_type == base_failure.exc_type
            and new.failure.message == base_failure.message
        )
        if (
            base_failure is not None
            and not same_on_pr
            and missing_new_name(base_failure, new_names)
        ):
            if new.verdict == "pass":
                return _match("no_proof", "new code: passes on the PR", outcome)
            if wrong_value:
                return _match("no_proof", "new code: a wrong value can't be checked", outcome)
            ok, why, through = _pr_side(outcome, expected, test_path, changed)
            if ok:
                return _match(
                    "proof", "new code fails as claimed", outcome, proof="new_code", through=through
                )
            return _match("no_proof", f"new code: {why}", outcome, through=through)
        feedback = (
            describe_failure(base_failure, "the base commit")
            + "\nA correct test passes on the base commit, before the pull request."
        )
        return _match("broken", "fails on the base commit", outcome, feedback=feedback)
    if new.verdict == "pass":
        return _match("no_proof", "passes on both commits", outcome)
    failure = new.failure
    if failure is None or failure.phase != "call":
        phase = failure.phase if failure is not None else "unknown"
        return _match("no_proof", f"the PR run failed outside the test ({phase})", outcome)
    if wrong_value:
        if not matches(failure, expected):
            return _match("no_proof", f"the PR run failed with {failure.exc_type}", outcome)
        if not new.hits:
            return _match("no_proof", "the test ran no line the PR changed", outcome)
        return _match("proof", "wrong value on the PR's changed lines", outcome, proof="executed")
    if same_name(failure.exc_type, "AssertionError") and new.hits:
        line = _failed_line(failure, test_path)
        if line is not None and checks_caught_error(test_code, line, expected):
            name = expected.rsplit(".", 1)[-1]
            return _match(
                "proof",
                f"a check on the caught {name} fails",
                outcome,
                proof="executed",
                caught=name,
            )
    ok, why, through = _pr_side(outcome, expected, test_path, changed)
    if ok:
        return _match("proof", "fails as claimed", outcome, proof="executed", through=through)
    return _match("no_proof", why, outcome, through=through)
