"""Reading one test run: which tests passed or failed, and how each failure looked.

A run's result per test is one of:
- passed;
- failed: an assertion or exception in the test itself;
- error: the test couldn't get going, because its setup (a fixture) failed or its file
  didn't import;
- skipped: skipped, or an expected failure (xfail), so it says nothing either way.
A failure keeps its exception type (such as "AttributeError"), its message and the line it
came from. The evidence gate later checks that a failure matches the reviewer's claim, so an
ImportError can't pass for proof of an AttributeError.

Two readers:
- read_pytest_report() reads the lines our pytest plugin wrote (pytest_probe.py). Test IDs are
  made relative to the repo root, whatever pytest's own root folder was.
- read_django_output() reads Django's runtests.py output at verbosity 2: one line per test
  ("test_x (app.tests.T.test_x) ... ok"), then a block per failure with its traceback. Other
  lines can land between a test's name and its status, so a status alone on a later line
  still counts for the test that's waiting for one.

Imported by other modules; the file isn't run directly.
"""

import json
import posixpath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

# What one test did in one run.
Status = Literal["passed", "failed", "error", "skipped"]

# An exception's name at the start of a traceback's last line, such as "ImportError" or
# "django.core.exceptions.ImproperlyConfigured".
_EXCEPTION = r"[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failed|Failure)"

# A test's status from one phase's outcome. Setup passing says nothing yet: the call decides.
_SETUP_STATUS: dict[str, Status] = {"failed": "error", "skipped": "skipped"}
_CALL_STATUS: dict[str, Status] = {"passed": "passed", "failed": "failed", "skipped": "skipped"}


class Failure(BaseModel):
    """How a test failed.

    Attributes:
        exc_type (str): The exception's type, such as "AttributeError". "" when unknown.
        message (str): The exception's message, as the runner printed it.
        where (str): The line it was raised at, such as "pkg/a.py:12", relative to the repo
            root when inside it. "" when unknown.
        phase (str): When it happened: "setup", "call", "teardown" or "collect".
    """

    model_config = ConfigDict(frozen=True)

    exc_type: str
    message: str
    where: str
    phase: str


class Result(BaseModel):
    """One test's result in one run.

    Attributes:
        test (str): The test's ID: a pytest node ID such as "pkg/tests/test_a.py::test_b",
            or a Django label such as "app.tests.ATests.test_b".
        status (Status): "passed", "failed", "error" or "skipped".
        failure (Failure | None): How it failed, for "failed" and "error".
    """

    model_config = ConfigDict(frozen=True)

    test: str
    status: Status
    failure: Failure | None = None


class Readout(BaseModel):
    """What one run's output says about each test.

    Attributes:
        results (dict[str, Result]): Each test's result, by test ID.
        collect_errors (dict[str, Failure]): Files or classes that failed to load, by ID (a
            file path such as "pkg/tests/test_a.py", or a Django module or class label). Their
            tests have no result of their own.
        collect_skips (dict[str, str]): Files or classes skipped while loading, such as a file
            that calls pytest.importorskip() for a package that isn't installed, with the
            reason, by ID. Their tests have no result of their own either.
    """

    model_config = ConfigDict(frozen=True)

    results: dict[str, Result]
    collect_errors: dict[str, Failure]
    collect_skips: dict[str, str] = {}


class RunReport(Readout):
    """One run of the tests at one side of a PR.

    Attributes:
        results (dict[str, Result]): Each test's result, by test ID.
        collect_errors (dict[str, Failure]): Files or classes that failed to load, by ID.
        collect_skips (dict[str, str]): Files or classes skipped while loading, by ID.
        problem (str | None): Why the run couldn't run at all or stopped early, such as
            "timeout", "out_of_memory", "usage_error" or "runner_error". None when it ran.
        detail (str): The end of the run's output.
        seconds (float): How long it took.
    """

    problem: str | None
    detail: str
    seconds: float


def exception_type(text: str) -> str:
    """Find the exception type in a failure's text.

    Args:
        text (str): A message such as "KeyError: 'a'", or a traceback whose error lines start
            with "E   ", as pytest prints them.

    Returns:
        str: The last exception type named, "AssertionError" for a bare pytest assertion
        message ("assert 1 == 2"), or "" when none is found.
    """
    found = re.findall(rf"^(?:E\s+)?({_EXCEPTION})(?::|$)", text, re.MULTILINE)
    if found:
        return str(found[-1])
    if text.lstrip().startswith("assert "):
        return "AssertionError"
    return ""


def _relative(path: str, root: str, repo: str) -> str:
    """Make a path relative to the repo root.

    Args:
        path (str): A path relative to `root`, or absolute.
        root (str): pytest's root folder.
        repo (str): The repo's working copy.

    Returns:
        str: The path relative to the repo, or as given when it's outside the repo.
    """
    full = posixpath.normpath(posixpath.join(root, path))
    if full == repo or full.startswith(repo.rstrip("/") + "/"):
        return posixpath.relpath(full, repo)
    return path


def _repo_id(nodeid: str, root: str, repo: str) -> str:
    """Make a pytest node ID relative to the repo root.

    Args:
        nodeid (str): Such as "tests/test_a.py::test_b", relative to pytest's root folder.
        root (str): pytest's root folder.
        repo (str): The repo's working copy.

    Returns:
        str: The same ID with its file relative to the repo root.
    """
    path, sep, rest = nodeid.partition("::")
    return _relative(path, root, repo) + sep + rest if path else nodeid


def _where(where: str, repo: str) -> str:
    """Make a failure's place ("/work/repo/a.py:12") relative to the repo root.

    Args:
        where (str): The place, with an absolute path.
        repo (str): The repo's working copy.

    Returns:
        str: Such as "a.py:12".
    """
    path, sep, line = where.rpartition(":")
    if not sep or not posixpath.isabs(path):
        return where
    return f"{_relative(path, '/', repo)}:{line}"


def read_pytest_report(text: str, repo: str) -> Readout:
    """Read what the pytest plugin wrote during one run.

    A test's result comes from its phases: a failed setup makes it an error, a failed call a
    failure, and a failed teardown turns a pass into an error. An expected failure (xfail)
    counts as skipped.

    Args:
        text (str): The plugin's JSON lines. Lines that aren't valid JSON are ignored.
        repo (str): The repo's working copy, such as "/work/psf__requests".

    Returns:
        Readout: Each test's result by ID, and the files or classes that failed to collect or
        were skipped while collecting, by ID.
    """
    root = repo
    results: dict[str, Result] = {}
    collect: dict[str, Failure] = {}
    skips: dict[str, str] = {}
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        if kind == "session":
            root = str(entry.get("rootdir") or repo)
            continue
        nodeid = _repo_id(str(entry.get("nodeid", "")), root, repo)
        message = str(entry.get("message", ""))
        failure = Failure(
            exc_type=str(entry.get("exc_type") or exception_type(message)),
            message=message,
            where=_where(str(entry.get("where", "")), repo),
            phase=str(entry.get("when", "collect")),
        )
        if kind == "collect" and entry.get("outcome") == "skipped":
            skips[nodeid] = message
            continue
        if kind == "collect":
            text_type = exception_type(str(entry.get("text", "")))
            collect[nodeid] = failure.model_copy(
                update={"exc_type": text_type or failure.exc_type, "phase": "collect"}
            )
            continue
        if kind != "test":
            continue
        when, outcome = entry.get("when"), str(entry.get("outcome"))
        before = results.get(nodeid)
        status: Status | None = None
        if entry.get("xfail"):
            status = "skipped"
        elif when == "setup":
            status = _SETUP_STATUS.get(outcome)
        elif when == "call":
            status = _CALL_STATUS.get(outcome)
        elif when == "teardown" and outcome == "failed" and before and before.status == "passed":
            status = "error"
        if status is None:
            continue
        keep = failure if status in ("failed", "error") else None
        results[nodeid] = Result(test=nodeid, status=status, failure=keep)
    return Readout(results=results, collect_errors=collect, collect_skips=skips)


# A test's line at verbosity 2: "test_b (app.tests.ATests.test_b) ... ok". Python before 3.11
# prints "(app.tests.ATests)", without the method. A docstring line may come between.
_DJANGO_TEST = re.compile(r"^(\w+) \(([\w.]+)\)")
_DJANGO_WORDS = r"(ok|FAIL|ERROR|skipped\b.*|expected failure|unexpected success)"
_DJANGO_STATUS = re.compile(rf" \.\.\. {_DJANGO_WORDS}\s*$")
# The status alone on its line: other output (such as Django's own start-up lines, which go
# to stdout while test lines go to stderr) can land between a test's line and its status.
_DJANGO_LONE_STATUS = re.compile(rf"^{_DJANGO_WORDS}\s*$")
# A failure's block: "FAIL: test_b (app.tests.ATests.test_b)", maybe a docstring line, a line
# of dashes, then the traceback, up to the next line of "=" or "-".
_DJANGO_BLOCK = re.compile(
    r"^(FAIL|ERROR): ([\w.]+) \(([\w.]+)\)[^\n]*\n(?:[^\n-][^\n]*\n)?-{20,}\n"
    r"(.*?)(?=^={20,}$|^-{20,}$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_DJANGO_STATUSES: dict[str, Status] = {
    "ok": "passed",
    "FAIL": "failed",
    "ERROR": "error",
    "expected failure": "skipped",
    "unexpected success": "failed",
}

# unittest's names for problems outside any one test.
_CLASS_FIXTURES = {"setUpClass", "tearDownClass", "setUpModule", "tearDownModule"}


def _django_id(method: str, where: str) -> str:
    """Give a Django test's full label.

    Args:
        method (str): The test method, such as "test_b".
        where (str): What unittest printed in brackets: "app.tests.ATests.test_b", or
            "app.tests.ATests" before Python 3.11.

    Returns:
        str: Such as "app.tests.ATests.test_b".
    """
    return where if where.endswith(f".{method}") else f"{where}.{method}"


def _traceback_failure(traceback: str, repo: str, phase: str) -> Failure:
    """Read a unittest traceback's exception type, message and failing line.

    Args:
        traceback (str): The traceback, as printed under "FAIL: ..." or "ERROR: ...".
        repo (str): The repo's working copy.
        phase (str): "call" or "collect".

    Returns:
        Failure: The failure.
    """
    lines = [line for line in traceback.rstrip().splitlines() if line.strip()]
    last = next(
        (line for line in reversed(lines) if re.match(rf"^{_EXCEPTION}(?::|$)", line)),
        lines[-1] if lines else "",
    )
    places = re.findall(r'File "([^"]+)", line (\d+)', traceback)
    where = _where(f"{places[-1][0]}:{places[-1][1]}", repo) if places else ""
    return Failure(
        exc_type=exception_type(last), message=last.strip()[:1000], where=where, phase=phase
    )


def read_django_output(output: str, repo: str) -> Readout:
    """Read one run of Django's runtests.py at verbosity 2.

    Args:
        output (str): The run's output.
        repo (str): The repo's working copy.

    Returns:
        Readout: Each test's result by label, and the modules or classes that failed to
        load, by label.
    """
    statuses: dict[str, Status] = {}
    current: str | None = None
    for line in output.splitlines():
        started = _DJANGO_TEST.match(line)
        if started:
            method, where = started.groups()
            current = None if "_FailedTest" in where else _django_id(method, where)
        found = _DJANGO_STATUS.search(line) or (
            None if started else _DJANGO_LONE_STATUS.match(line)
        )
        if found and current:
            word = found.group(1)
            statuses[current] = "skipped" if word.startswith("skipped") else _DJANGO_STATUSES[word]
            current = None
    failures: dict[str, Failure] = {}
    collect: dict[str, Failure] = {}
    for block in _DJANGO_BLOCK.finditer(output):
        _, name, where, traceback = block.groups()
        if "_FailedTest" in where:
            # A test module that didn't import: "ERROR: app.tests (unittest.loader._FailedTest)".
            collect[name] = _traceback_failure(traceback, repo, "collect")
        elif name in _CLASS_FIXTURES:
            collect[where] = _traceback_failure(traceback, repo, "collect")
        else:
            failures[_django_id(name, where)] = _traceback_failure(traceback, repo, "call")
    results = {
        test: Result(
            test=test,
            status=status,
            failure=failures.get(test) if status in ("failed", "error") else None,
        )
        for test, status in statuses.items()
    }
    return Readout(results=results, collect_errors=collect)
