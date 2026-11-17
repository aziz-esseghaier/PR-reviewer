"""A pytest plugin that writes down each test's result, for the differential test runner.

It's copied into a sandbox as pr_reviewer_probe.py, in a folder outside the repo, and loaded
with `-p pr_reviewer_probe`, so the repo's own code and tests stay as they are. It writes one
JSON line:
- when the session starts: pytest's root folder, which test IDs are relative to;
- for every phase of every test (setup, call, teardown): its outcome, and for a failure the
  exception's type, its message, the line it was raised at, the names of its classes (the
  type and its parents, so a subclass of the expected error is recognized) and the place of
  every frame of its traceback (so we can tell which lines the error went through);
- for every file or class that fails to collect, such as a test file whose imports fail, or
  that is skipped while collecting, such as a file needing an optional package.
Our code (difftest/reports.py) turns these lines into one result per test.

When the PR_REVIEWER_WATCH environment variable names a JSON file of lines to watch (the lines
a PR changed), the plugin also notes which of them each test ran while it was called, with
Python's line tracer, switched on only for the test's call and only inside the watched files.

It must work on every Python and pytest version the benchmark's repos use (Python 3.7 on), so
it uses only the standard library and long-standing pytest hooks, and its type hints are never
evaluated. The file it writes is named by the PR_REVIEWER_REPORT environment variable; without
it, the plugin does nothing.

Loaded by pytest inside a sandbox; our own code never imports it.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Generator
from typing import Any

import pytest

# The file the results go to.
_REPORT = os.environ.get("PR_REVIEWER_REPORT", "")

# The file listing the lines to watch: {"root": the working copy, "lines": {path: [lines]}}.
_WATCH = os.environ.get("PR_REVIEWER_WATCH", "")

# Names under which a failure's details, and the watched lines a test ran, travel with its
# report. Reports carry their user_properties across pytest-xdist workers, so they survive
# parallel runs too.
_TYPE_KEY = "pr_reviewer_exception"
_CLASSES_KEY = "pr_reviewer_classes"
_FRAMES_KEY = "pr_reviewer_frames"
_HITS_KEY = "pr_reviewer_hits"

# Most characters kept of a failure's message, and of a collection error's text.
_MESSAGE_CHARS = 1000
_TEXT_CHARS = 3000

# Most traceback frames kept (the innermost ones), and most watched lines noted per test.
_MAX_FRAMES = 60
_MAX_HITS = 500


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


def _classes(excinfo: Any) -> list[str]:
    """Give the names of an exception's class and its parents.

    Args:
        excinfo (Any): pytest's ExceptionInfo.

    Returns:
        list[str]: Such as ["UnsupportedOperation", "OSError", "ValueError", "Exception",
        "BaseException"]; [] when it can't be read.
    """
    try:
        return [str(cls.__name__) for cls in excinfo.type.__mro__ if cls is not object]
    except Exception:  # the plugin must never break a test run
        return []


def _frames(excinfo: Any) -> list[str]:
    """Give the place of every frame of an exception's traceback, innermost last.

    pytest's and pluggy's own frames, which only call the test, are left out.

    Args:
        excinfo (Any): pytest's ExceptionInfo.

    Returns:
        list[str]: Such as ["/work/repo/tests/test_a.py:5", "/work/repo/pkg/a.py:12"], the
        innermost _MAX_FRAMES; [] when it can't be read.
    """
    try:
        places = []
        for entry in excinfo.traceback:
            path = str(entry.path)
            parts = path.replace("\\", "/").split("/")
            if "_pytest" in parts or "pluggy" in parts:
                continue
            places.append(f"{path}:{entry.lineno + 1}")
    except Exception:  # the plugin must never break a test run
        return []
    return places[-_MAX_FRAMES:]


class _Watcher:
    """Notes which watched lines run, through Python's line tracer.

    Only frames of watched files get a line tracer; every other call is let through at once,
    so the test isn't slowed much.

    Args:
        root (str): The working copy, such as "/work/psf__requests".
        lines (dict[str, set[int]]): The lines to watch, by path relative to the repo root.
    """

    def __init__(self, root: str, lines: dict[str, set[int]]) -> None:
        self.root = root.replace("\\", "/").rstrip("/")
        self.lines = lines
        self.hits: set[str] = set()
        self._known: dict[str, Any] = {}

    def _relative(self, filename: str) -> Any:
        """Give the watched path a code file is, or None.

        A file under the working copy is matched by its path; a file elsewhere (a package
        installed into site-packages) by its path's end.

        Args:
            filename (str): The code's file, as Python knows it.

        Returns:
            Any: The watched path (str), or None when the file isn't watched.
        """
        if filename in self._known:
            return self._known[filename]
        found = None
        name = filename.replace("\\", "/")
        prefix = self.root + "/"
        if name.startswith(prefix) and name[len(prefix) :] in self.lines:
            found = name[len(prefix) :]
        if found is None:
            for path in self.lines:
                if name.endswith("/" + path):
                    found = path
                    break
        self._known[filename] = found
        return found

    def trace(self, frame: Any, event: str, arg: Any) -> Any:
        """Python's global tracer: give watched frames a line tracer.

        Args:
            frame (Any): The frame being called.
            event (str): "call".
            arg (Any): Unused.

        Returns:
            Any: The line tracer for a watched file, else None.
        """
        path = self._relative(frame.f_code.co_filename)
        if path is None:
            return None
        wanted = self.lines[path]
        hits = self.hits

        def local(frame: Any, event: str, arg: Any) -> Any:
            if event == "line" and frame.f_lineno in wanted:
                hits.add(f"{path}:{frame.f_lineno}")
            return local

        return local


def _watcher() -> Any:
    """Read the lines to watch, if any were given.

    Returns:
        Any: A _Watcher, or None when nothing is watched or the file can't be read.
    """
    if not _WATCH:
        return None
    try:
        with open(_WATCH, encoding="utf-8") as handle:
            data = json.load(handle)
        lines = {str(path): {int(n) for n in numbers} for path, numbers in data["lines"].items()}
        return _Watcher(str(data.get("root", "")), lines)
    except Exception:  # the plugin must never break a test run
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, Any, None]:
    """Watch which changed lines a test runs while it's called.

    Args:
        item (pytest.Item): The test.

    Yields:
        None: Control to pytest, which calls the test.
    """
    watcher = _watcher()
    if watcher is None:
        yield
        return
    before = sys.gettrace()
    sys.settrace(watcher.trace)
    try:
        yield
    finally:
        sys.settrace(before)
        item.user_properties.append((_HITS_KEY, sorted(watcher.hits)[:_MAX_HITS]))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    """Attach the exception type, its classes and its traceback to a failing phase's report.

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
        report.user_properties.append((_CLASSES_KEY, _classes(call.excinfo)))
        report.user_properties.append((_FRAMES_KEY, _frames(call.excinfo)))


def _last(report: Any, key: str) -> Any:
    """Give the last value a report carries under a name.

    Args:
        report (Any): The report.
        key (str): The name, such as _TYPE_KEY.

    Returns:
        Any: The value, or None when there's none.
    """
    values = [value for name, value in report.user_properties if name == key]
    return values[-1] if values else None


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
        found = _last(report, _TYPE_KEY)
        entry["exc_type"] = str(found) if found else ""
        entry["message"], entry["where"] = _crash(report)
        entry["classes"] = _last(report, _CLASSES_KEY) or []
        entry["frames"] = _last(report, _FRAMES_KEY) or []
    if report.when == "call":
        hits = _last(report, _HITS_KEY)
        if hits is not None:
            entry["hits"] = hits
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
