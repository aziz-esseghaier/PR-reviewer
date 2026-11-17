"""
Tests for reading one test run (difftest/reports.py), and for the pytest plugin itself.

The plugin test runs pytest on this machine, on small test files written by the test (no PR
code), with the plugin loaded the way sandboxes load it. The other tests read fixed text.
No Docker needed.

How to run (from the project folder):
    pytest tests/difftest/test_reports.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from pr_reviewer.difftest import reports

SAMPLE_TESTS = """
import pytest

def test_pass():
    assert True

def test_fail():
    assert 1 == 2

def test_attribute():
    None.foo

@pytest.fixture
def broken():
    raise KeyError("x")

def test_setup_error(broken):
    pass

@pytest.mark.skip(reason="not today")
def test_skipped():
    pass

@pytest.mark.xfail
def test_expected_failure():
    assert False

@pytest.fixture
def bad_teardown():
    yield
    raise RuntimeError("teardown")

def test_teardown_error(bad_teardown):
    pass

@pytest.mark.parametrize("n", [1, 2])
def test_param(n):
    assert n == 1
"""


OPTIONAL_TESTS = """
import pytest

pytest.importorskip("no_such_module_xyz")

def test_y():
    pass
"""


def run_probe(
    folder: Path, *targets: str, extra: dict[str, str] | None = None
) -> tuple[int, reports.Readout]:
    """Run pytest with the plugin on test files in a folder, and read what it wrote.

    Args:
        folder (Path): The folder, which holds the test files.
        *targets (str): What pytest runs, such as "test_a.py" or "test_a.py::test_b".
        extra (dict[str, str] | None): More environment variables, such as the watch file.

    Returns:
        tuple[int, reports.Readout]: pytest's exit code, and the results read.
    """
    report = folder / "report.jsonl"
    report.unlink(missing_ok=True)
    env = {**os.environ, "PR_REVIEWER_REPORT": str(report), **(extra or {})}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pr_reviewer.difftest.pytest_probe",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            "-q",
            "--rootdir",
            str(folder),
            *targets,
        ],
        cwd=folder,
        env=env,
        capture_output=True,
        check=False,
    )
    return done.returncode, reports.read_pytest_report(
        report.read_text(encoding="utf-8"), str(folder)
    )


def test_the_plugin_records_each_tests_result_and_exception(tmp_path: Path) -> None:
    """Run pytest with the plugin on sample tests and check every kind of result it reads."""
    (tmp_path / "test_sample.py").write_text(SAMPLE_TESTS, encoding="utf-8")
    (tmp_path / "test_broken_import.py").write_text(
        "from no_such_module_xyz import thing\n\ndef test_x():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "test_optional.py").write_text(OPTIONAL_TESTS, encoding="utf-8")
    _, read = run_probe(tmp_path, "test_sample.py", "test_broken_import.py", "test_optional.py")
    results, collect = read.results, read.collect_errors

    def status(name: str) -> str:
        return results[f"test_sample.py::{name}"].status

    def failure(name: str) -> reports.Failure:
        found = results[f"test_sample.py::{name}"].failure
        assert found is not None
        return found

    assert status("test_pass") == "passed"
    assert status("test_fail") == "failed"
    assert failure("test_fail").exc_type == "AssertionError"
    assert failure("test_fail").phase == "call"
    assert failure("test_attribute").exc_type == "AttributeError"
    assert "foo" in failure("test_attribute").message
    assert failure("test_attribute").where.endswith("test_sample.py:11")
    assert status("test_setup_error") == "error"
    assert (failure("test_setup_error").exc_type, failure("test_setup_error").phase) == (
        "KeyError",
        "setup",
    )
    assert status("test_skipped") == "skipped"
    assert status("test_expected_failure") == "skipped"
    assert status("test_teardown_error") == "error"
    assert failure("test_teardown_error").exc_type == "RuntimeError"
    assert status("test_param[1]") == "passed"
    assert status("test_param[2]") == "failed"
    assert collect["test_broken_import.py"].exc_type == "ModuleNotFoundError"
    assert collect["test_broken_import.py"].phase == "collect"
    assert "test_optional.py" not in collect
    assert "no_such_module_xyz" in read.collect_skips["test_optional.py"]


CODE = """\
class CartError(ValueError):
    pass


def total(items):
    if not items:
        raise CartError("empty cart")
    return sum(items)


def unused():
    return 1
"""

CODE_TESTS = """\
from cart import total


def test_empty():
    total([])


def test_full():
    assert total([1, 2]) == 3
"""


def test_the_plugin_records_classes_traceback_and_watched_lines(tmp_path: Path) -> None:
    """Check a failure's classes and traceback frames, and the watched lines each test ran."""
    (tmp_path / "cart.py").write_text(CODE, encoding="utf-8")
    (tmp_path / "test_cart.py").write_text(CODE_TESTS, encoding="utf-8")
    watch = tmp_path / "watch.json"
    watch.write_text(
        json.dumps({"root": str(tmp_path), "lines": {"cart.py": [6, 7, 8, 12]}}), encoding="utf-8"
    )
    _, read = run_probe(tmp_path, "test_cart.py", extra={"PR_REVIEWER_WATCH": str(watch)})
    empty = read.results["test_cart.py::test_empty"]
    full = read.results["test_cart.py::test_full"]
    assert empty.failure is not None and empty.failure.exc_type == "CartError"
    assert empty.failure.classes[:2] == ("CartError", "ValueError")
    # On this machine the frames' paths are Windows paths, which stay absolute.
    names = [frame.replace("\\", "/").rsplit("/", 1)[-1] for frame in empty.failure.frames]
    assert names == ["test_cart.py:5", "cart.py:7"]
    assert empty.hits == ("cart.py:6", "cart.py:7")
    assert full.hits == ("cart.py:6", "cart.py:8"), "line 12 never runs"
    _, unwatched = run_probe(tmp_path, "test_cart.py")
    assert unwatched.results["test_cart.py::test_full"].hits == ()


def test_asking_for_a_test_in_a_file_that_doesnt_load_stops_pytest(tmp_path: Path) -> None:
    """Check the pytest behavior the runner works around: the whole run stops (exit code 4).

    It happens both for a file that fails to import and for one that skips itself. The
    plugin still records both, which tells the runner which tests to leave out when it runs
    the others again.
    """
    (tmp_path / "test_sample.py").write_text(SAMPLE_TESTS, encoding="utf-8")
    (tmp_path / "test_optional.py").write_text(OPTIONAL_TESTS, encoding="utf-8")
    (tmp_path / "test_broken_import.py").write_text(
        "from no_such_module_xyz import thing\n\ndef test_x():\n    pass\n", encoding="utf-8"
    )
    for unloaded in ("test_optional.py::test_y", "test_broken_import.py::test_x"):
        code, read = run_probe(tmp_path, "test_sample.py::test_pass", unloaded)
        assert code == 4
        assert read.results == {}
    assert list(read.collect_errors) == ["test_broken_import.py"]


def line(**entry: object) -> str:
    """Write one plugin line.

    Args:
        **entry (object): The entry's fields.

    Returns:
        str: The JSON line.
    """
    return json.dumps(entry)


def test_ids_and_places_become_relative_to_the_repo() -> None:
    """Check test IDs from a sub-folder root and absolute failure places are made relative."""
    text = "\n".join(
        [
            line(kind="session", rootdir="/work/repo/sub"),
            line(kind="test", nodeid="tests/test_a.py::test_b", when="setup", outcome="passed"),
            line(
                kind="test",
                nodeid="tests/test_a.py::test_b",
                when="call",
                outcome="failed",
                exc_type="ValueError",
                message="ValueError: bad",
                where="/work/repo/sub/pkg/a.py:7",
            ),
            "not json at all",
            line(
                kind="test",
                nodeid="tests/test_a.py::test_b",
                when="teardown",
                outcome="failed",
                exc_type="OSError",
                message="",
                where="",
            ),
        ]
    )
    read = reports.read_pytest_report(text, "/work/repo")
    result = read.results["sub/tests/test_a.py::test_b"]
    assert result.status == "failed"
    assert result.failure is not None
    assert (result.failure.exc_type, result.failure.where) == ("ValueError", "sub/pkg/a.py:7")
    assert (read.collect_errors, read.collect_skips) == ({}, {})


def test_files_skipped_while_loading_are_kept_apart_from_errors() -> None:
    """Check a skipped collection line is a skip, and a line without an outcome an error."""
    text = "\n".join(
        [
            line(kind="session", rootdir="/work/repo"),
            line(
                kind="collect",
                nodeid="tests/test_sparse.py",
                outcome="skipped",
                message="could not import 'sparse'",
            ),
            line(kind="collect", nodeid="tests/test_gone.py", message="ImportError: gone"),
        ]
    )
    read = reports.read_pytest_report(text, "/work/repo")
    assert read.collect_skips == {"tests/test_sparse.py": "could not import 'sparse'"}
    assert list(read.collect_errors) == ["tests/test_gone.py"]


def test_exception_types_are_found_in_messages_and_tracebacks() -> None:
    """Check the exception type read from pytest messages and error lines."""
    assert reports.exception_type("KeyError: 'a'") == "KeyError"
    assert reports.exception_type("assert 1 == 2") == "AssertionError"
    assert (
        reports.exception_type("E   ImportError: cannot import name 'x'\nE   more") == "ImportError"
    )
    assert reports.exception_type("django.core.exceptions.FieldError: no") == (
        "django.core.exceptions.FieldError"
    )
    assert reports.exception_type("something else happened") == ""


DJANGO_OUTPUT = """\
Testing against Django installed in '/work/django__django/django' with up to 1 processes
test_a (app.tests.ATests.test_a) ... ok
test_b (app.tests.ATests)
Checks b. ... FAIL
test_c (app.tests.ATests.test_c) ... ERROR
test_d (app.tests.ATests.test_d) ... skipped 'no db'
test_e (app.tests.ATests.test_e) ... expected failure
other.tests (unittest.loader._FailedTest.other.tests) ... ERROR

======================================================================
ERROR: other.tests (unittest.loader._FailedTest.other.tests)
----------------------------------------------------------------------
ImportError: Failed to import test module: other.tests
Traceback (most recent call last):
  File "/work/django__django/tests/other/tests.py", line 3, in <module>
    from app.models import Gone
ImportError: cannot import name 'Gone'

======================================================================
FAIL: test_b (app.tests.ATests)
Checks b.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/work/django__django/tests/app/tests.py", line 20, in test_b
    self.assertEqual(1, 2)
AssertionError: 1 != 2

======================================================================
ERROR: test_c (app.tests.ATests.test_c)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/work/django__django/tests/app/tests.py", line 25, in test_c
    obj.missing
AttributeError: 'A' object has no attribute 'missing'

----------------------------------------------------------------------
Ran 6 tests in 0.1s

FAILED (failures=1, errors=2, skipped=1, expected failures=1)
"""


def test_django_output_gives_each_tests_result_and_failure() -> None:
    """Check Django's verbose output: both label formats, failures and a broken module."""
    read = reports.read_django_output(DJANGO_OUTPUT, "/work/django__django")
    results, collect = read.results, read.collect_errors
    assert results["app.tests.ATests.test_a"].status == "passed"
    failed = results["app.tests.ATests.test_b"]
    assert failed.status == "failed"
    assert failed.failure is not None
    assert (failed.failure.exc_type, failed.failure.where) == (
        "AssertionError",
        "tests/app/tests.py:20",
    )
    errored = results["app.tests.ATests.test_c"]
    assert errored.status == "error"
    assert errored.failure is not None and errored.failure.exc_type == "AttributeError"
    assert results["app.tests.ATests.test_d"].status == "skipped"
    assert results["app.tests.ATests.test_e"].status == "skipped"
    assert collect["other.tests"].exc_type == "ImportError"
    assert "other.tests" not in results


# A real run's start, where Django's own lines landed between the first test and its status.
DJANGO_INTERLEAVED = """\
test_addslashes01 (template_tests.filter_tests.test_addslashes.AddslashesTests) ... Testing \
against Django installed in '/work/django__django/django'
Importing application template_tests
Found 2 test(s).
System check identified no issues (0 silenced).
ok
test_addslashes02 (template_tests.filter_tests.test_addslashes.AddslashesTests) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.002s

OK
"""


def test_a_django_status_after_other_lines_still_counts() -> None:
    """Check a status alone on a later line goes to the test waiting for one."""
    read = reports.read_django_output(DJANGO_INTERLEAVED, "/work/django__django")
    label = "template_tests.filter_tests.test_addslashes.AddslashesTests"
    assert {test: result.status for test, result in read.results.items()} == {
        f"{label}.test_addslashes01": "passed",
        f"{label}.test_addslashes02": "passed",
    }
