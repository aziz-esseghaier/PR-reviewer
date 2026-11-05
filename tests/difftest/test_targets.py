"""
Tests for finding the tests a diff added or changed (difftest/targets.py).

No Docker needed.

How to run (from the project folder):
    pytest tests/difftest/test_targets.py
"""

from pr_reviewer.difftest import targets

DIFF = """\
diff --git a/pkg/tests/test_a.py b/pkg/tests/test_a.py
index 1111111..2222222 100644
--- a/pkg/tests/test_a.py
+++ b/pkg/tests/test_a.py
@@ -4 +4 @@ def test_one():
-    assert f() == 1
+    assert f() == 2
@@ -9,0 +10,4 @@ def test_two():
+
+@pytest.mark.slow
+def test_three():
+    assert g()
@@ -20,2 +23,0 @@ class TestFour:
-        x = 1
-        y = 2
diff --git a/pkg/old_test.py b/pkg/old_test.py
deleted file mode 100644
--- a/pkg/old_test.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
"""

SOURCE = """\
import pytest

def test_one():
    assert f() == 2

def helper():
    return 1

def test_two():
    assert helper()

@pytest.mark.slow
def test_three():
    assert g()

class TestFour:
    def setup_method(self):
        pass

    def test_five(self):
        a = 1

        b = 2

    async def test_six(self):
        pass
"""


def test_changed_lines_are_read_from_a_zero_context_diff() -> None:
    """Check changed, added and deleted lines, and that deleted files are left out."""
    lines = targets.changed_lines(DIFF)
    assert lines == {"pkg/tests/test_a.py": {4, 10, 11, 12, 13, 23, 24}}


def test_tests_are_found_with_their_decorators_and_classes() -> None:
    """Check test functions and methods with their line ranges."""
    found = targets.find_tests(SOURCE)
    assert [(t.cls, t.name, t.start, t.end) for t in found] == [
        (None, "test_one", 3, 4),
        (None, "test_two", 9, 10),
        (None, "test_three", 12, 14),
        ("TestFour", "test_five", 20, 23),
        ("TestFour", "test_six", 25, 26),
    ]
    assert targets.find_tests("def broken(:\n") == []


def test_only_touched_tests_are_kept() -> None:
    """Check a decorator line or a line inside a method selects that test, nothing else."""
    touched = targets.changed_tests(SOURCE, {4, 12, 22})
    assert [t.name for t in touched] == ["test_one", "test_three", "test_five"]


def test_ids_follow_each_runner() -> None:
    """Check pytest node IDs and Django labels."""
    method = targets.FoundTest("test_five", "TestFour", 20, 23)
    function = targets.FoundTest("test_one", None, 3, 4)
    assert targets.node_id("pkg/tests/test_a.py", method, "pytest") == (
        "pkg/tests/test_a.py::TestFour::test_five"
    )
    assert targets.node_id("pkg/tests/test_a.py", function, "pytest") == (
        "pkg/tests/test_a.py::test_one"
    )
    assert targets.node_id("tests/app/tests.py", method, "django") == (
        "app.tests.TestFour.test_five"
    )
    assert targets.node_id("django/tests.py", method, "django") == ""


# A base class and a mixin whose tests run only in the test classes that inherit them.
CLASSES = """\
import unittest

class DatasetIOBase:
    def test_roundtrip(self):
        pass

    def test_redefined(self):
        pass

class ZarrBase(DatasetIOBase):
    pass

class TestZarrDict(ZarrBase):
    pass

class TestZarrDir(ZarrBase):
    def test_redefined(self):
        pass

class TestNetCDF(DatasetIOBase):
    def __init__(self):
        pass

class TestOff(DatasetIOBase):
    __test__ = False

class CheckMixin:
    def test_check(self):
        pass

class CheckCase(CheckMixin, unittest.TestCase):
    pass

class OtherFileMixin:
    def test_elsewhere(self):
        pass
"""


def method(cls: str, name: str) -> targets.FoundTest:
    """Make a test method of a class in CLASSES.

    Args:
        cls (str): Its class.
        name (str): Its name.

    Returns:
        targets.FoundTest: The test (its lines don't matter here).
    """
    return targets.FoundTest(name, cls, 1, 2)


def test_classes_are_found_with_bases_methods_and_test_flag() -> None:
    """Check each top-level class's bases (without module), methods and __test__ = False."""
    classes = targets.find_classes(CLASSES)
    assert classes["CheckCase"].bases == ("CheckMixin", "TestCase")
    assert classes["TestZarrDir"].methods == frozenset({"test_redefined"})
    assert classes["TestOff"].off and not classes["TestZarrDict"].off
    assert targets.find_classes("class broken(:\n") == {}


def test_a_base_class_test_runs_in_the_test_classes_that_inherit_it() -> None:
    """Check a base method gets an ID per collected subclass that doesn't redefine it."""
    classes = targets.find_classes(CLASSES)
    path = "t/test_io.py"
    assert targets.node_ids(path, method("DatasetIOBase", "test_roundtrip"), classes, "pytest") == [
        "t/test_io.py::TestZarrDict::test_roundtrip",
        "t/test_io.py::TestZarrDir::test_roundtrip",
    ]
    assert targets.node_ids(path, method("ZarrBase", "test_redefined"), classes, "pytest") == [
        "t/test_io.py::TestZarrDict::test_redefined"
    ]
    assert targets.node_ids(path, method("CheckMixin", "test_check"), classes, "pytest") == [
        "t/test_io.py::CheckCase::test_check"
    ]
    assert targets.node_ids(path, method("TestZarrDir", "test_redefined"), classes, "pytest") == [
        "t/test_io.py::TestZarrDir::test_redefined"
    ]


def test_a_test_only_other_files_run_gets_no_id() -> None:
    """Check a mixin whose users live in other files gives nothing to run."""
    classes = targets.find_classes(CLASSES)
    found = targets.node_ids(
        "t/test_io.py", method("OtherFileMixin", "test_elsewhere"), classes, "pytest"
    )
    assert found == []


def test_django_runs_test_cases_and_classes_with_imported_bases() -> None:
    """Check Django labels go to TestCases, and not to mixins."""
    source = (
        "from django.test import SimpleTestCase\n\n"
        "class FilterMixin:\n    def test_a(self):\n        pass\n\n"
        "class FilterTests(FilterMixin, SimpleTestCase):\n    pass\n\n"
        "class MoreTests(BaseTests):\n    pass\n\n"
        "class Helper(object):\n    def test_b(self):\n        pass\n"
    )
    classes = targets.find_classes(source)
    path = "tests/template_tests/test_filters.py"
    assert targets.node_ids(path, method("FilterMixin", "test_a"), classes, "django") == [
        "template_tests.test_filters.FilterTests.test_a"
    ]
    assert targets.node_ids(path, method("MoreTests", "test_c"), classes, "django") == [
        "template_tests.test_filters.MoreTests.test_c"
    ]
    assert targets.node_ids(path, method("Helper", "test_b"), classes, "django") == []
