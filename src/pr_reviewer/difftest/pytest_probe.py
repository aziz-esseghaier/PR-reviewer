"""A pytest plugin that writes down each test's result, for the differential test runner.

It's copied into a sandbox as pr_reviewer_probe.py, in a folder outside the repo, and loaded
with `-p pr_reviewer_probe`, so the repo's own code and tests stay as they are. It writes one
JSON line:
- when the session starts: pytest's root folder, which test IDs are relative to;
- for every phase of every test (setup, call, teardown): its outcome, and for a failure the
  exception's type, its message and the line it was raised at;
- for every file or class that fails to collect, such as a test file whose imports fail, or
  that is skipped while collecting, such as a file needing an optional package.
Our code (difftest/reports.py) turns these lines into one result per test.

It must work on every Python and pytest version the benchmark's repos use (Python 3.7 on), so
it uses only the standard library and long-standing pytest hooks, and its type hints are never
evaluated. The file it writes is named by the PR_REVIEWER_REPORT environment variable; without
it, the plugin does nothing.

Loaded by pytest inside a sandbox; our own code never imports it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from typing import Any

import pytest

# The file the results go to.
_REPORT = os.environ.get("PR_REVIEWER_REPORT", "")

# Name under which a failure's exception type travels with its report. Reports carry their
# user_properties across pytest-xdist workers, so the type survives parallel runs too.
_TYPE_KEY = "pr_reviewer_exception"

# Most characters kept of a failure's message, and of a collection error's text.
_MESSAGE_CHARS = 1000
_TEXT_CHARS = 3000


def _write(entry: dict[str, Any]) -> None:
    """Append one result line to the report file.

    Args:
        entry (dict[str, Any]): The result.
    """
    if _REPORT:
        with open(_REPORT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def _crash(report: Any) -> tuple[str, str]:
    """Give a failure's message and the place it was raised.

    Args:
        report (Any): A failed test or collection report.

    Returns:
        tuple[str, str]: The message, such as "AttributeError: 'A' has no attribute 'b'", and
        the place, such as "/work/repo/pkg/a.py:12" ("" when unknown).
    """
    crash = getattr(report.longrepr, "reprcrash", None)
    if crash is not None:
        return str(crash.message)[:_MESSAGE_CHARS], f"{crash.path}:{crash.lineno}"
    return str(report.longrepr or "")[-_MESSAGE_CHARS:], ""


def pytest_sessionstart(session: pytest.Session) -> None:
    """Write down pytest's root folder.

    Args:
        session (pytest.Session): The test session.
    """
    # rootpath is pytest 6.1's name; older versions only have rootdir.
    config = session.config
    root = getattr(config, "rootpath", None) or getattr(config, "rootdir", "")
    _write({"kind": "session", "rootdir": str(root)})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    """Attach the exception type of a failing phase to its report.

    Args:
        item (pytest.Item): The test.
        call (pytest.CallInfo[None]): The phase that just ran, with its exception if any.

    Yields:
        None: Control to pytest, which makes the report.
    """
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None:
        report.user_properties.append((_TYPE_KEY, call.excinfo.typename))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Write down one phase of one test.

    Args:
        report (pytest.TestReport): The phase's report.
    """
    entry: dict[str, Any] = {
        "kind": "test",
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "xfail": hasattr(report, "wasxfail"),
        "seconds": round(float(getattr(report, "duration", 0.0)), 3),
    }
    if report.failed:
        types = [value for key, value in report.user_properties if key == _TYPE_KEY]
        entry["exc_type"] = str(types[-1]) if types else ""
        entry["message"], entry["where"] = _crash(report)
    _write(entry)


def _skip_reason(report: Any) -> str:
    """Give the reason a file or class was skipped while collecting.

    Args:
        report (Any): A skipped collection report.

    Returns:
        str: The reason, such as "could not import 'sparse': No module named 'sparse'".
    """
    # A skip's longrepr is a (path, line, reason) tuple.
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])[:_MESSAGE_CHARS]
    return str(longrepr or "")[-_MESSAGE_CHARS:]


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Write down a file or class that failed to collect, or was skipped while collecting.

    A file that calls `pytest.importorskip()` at the top, for an optional package that isn't
    installed, is skipped this way: none of its tests run, but nothing is broken.

    Args:
        report (pytest.CollectReport): The collection report.
    """
    if report.failed:
        message, where = _crash(report)
        _write(
            {
                "kind": "collect",
                "nodeid": report.nodeid,
                "outcome": "failed",
                "message": message,
                "where": where,
                "text": str(report.longrepr)[-_TEXT_CHARS:],
            }
        )
    elif report.skipped:
        _write(
            {
                "kind": "collect",
                "nodeid": report.nodeid,
                "outcome": "skipped",
                "message": _skip_reason(report),
            }
        )
