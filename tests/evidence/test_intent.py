"""Tests for the signs that a PR changed a behavior on purpose (evidence/intent.py).

Checks which added lines count: test lines naming the function, doc or changelog lines naming
it or the error, and warnings added inside the function; and which don't.

How to run (from the project folder):
    pytest tests/evidence/test_intent.py
"""

from pr_reviewer.evidence.intent import is_doc_file, signs
from tests.reviewer.helpers import make_pr

CODE = """\
12 |     def average(self):
13 |         warnings.warn("average is deprecated", DeprecationWarning)
14 |         return self.total / len(self.items)"""


def test_signs_come_from_tests_docs_and_new_warnings() -> None:
    """Check each kind of sign, and lines that aren't signs."""
    pr = make_pr(
        files=[
            (
                "pkg/tests/test_cart.py",
                "@@ -1 +1,3 @@\n+def test_empty():\n"
                "+    assert Cart([]).average() == 0\n+    x = 1",
            ),
            ("doc/changes.rst", "@@ -1 +1,2 @@\n+average now raises ZeroDivisionError\n+other"),
            (
                "pkg/cart.py",
                "@@ -12 +12,3 @@\n"
                '+        warnings.warn("average is deprecated", DeprecationWarning)\n'
                '+        warnings.warn("elsewhere")',
            ),
            ("pkg/other.py", "@@ -1 +1 @@\n+average = 3"),
        ]
    )
    found = signs(pr, "pkg/cart.py", ["average", "Cart"], "ZeroDivisionError", CODE)
    assert found == [
        "pkg/tests/test_cart.py (test): + assert Cart([]).average() == 0",
        "doc/changes.rst (docs): + average now raises ZeroDivisionError",
        'pkg/cart.py (warning): + warnings.warn("average is deprecated", DeprecationWarning)',
    ]


def test_signs_from_change_notes_removed_warnings_and_short_class_names() -> None:
    """Check notes like "changed to", removed deprecations and their tests, and a test that
    calls a private method through its short-named class."""
    pr = make_pr(
        files=[
            (
                "pkg/solver.py",
                "@@ -1,3 +1,1 @@\n"
                '+    The default of `solver` changed to "highs" in version 1.4.\n'
                "-        warnings.warn(\n"
                '-            "The default will change", FutureWarning)\n'
                "-        x = 1",
            ),
            (
                "pkg/tests/test_solver.py",
                "@@ -1,2 +1,1 @@\n"
                "+    assert Ci(x).series(x, -oo) == y\n"
                '-    with pytest.warns(FutureWarning, match="will change"):\n'
                "-        fit()",
            ),
        ]
    )
    found = signs(pr, "pkg/solver.py", ["_eval_aseries", "Ci"], None, "")
    assert found == [
        "pkg/tests/test_solver.py (test): + assert Ci(x).series(x, -oo) == y",
        'pkg/solver.py (note): + The default of `solver` changed to "highs" in version 1.4.',
        "pkg/solver.py (removed warning): - warnings.warn(",
        'pkg/solver.py (removed warning): - "The default will change", FutureWarning)',
        'pkg/tests/test_solver.py (removed test): - with pytest.warns(FutureWarning, match="will '
        'change"):',
    ]


def test_signs_take_one_of_each_kind_first() -> None:
    """Check a long run of test lines can't push a changelog note out of the list."""
    tests = "".join(f"+    assert average({n}) == {n}\n" for n in range(20))
    pr = make_pr(
        files=[
            ("pkg/tests/test_cart.py", f"@@ -1 +1,20 @@\n{tests}"),
            ("CHANGES.rst", "@@ -1 +1 @@\n+average now returns 0 for an empty cart"),
        ]
    )
    found = signs(pr, "pkg/cart.py", ["average"], None, "")
    assert len(found) == 12
    assert found[-1] == "CHANGES.rst (docs): + average now returns 0 for an empty cart"


def test_no_signs_without_a_matching_line() -> None:
    """Check a PR whose tests don't name the function gives no signs."""
    pr = make_pr(files=[("tests/test_x.py", "@@ -1 +1 @@\n+def test_x():\n+    pass")])
    assert signs(pr, "pkg/cart.py", ["average"], None, "") == []


def test_doc_files_are_told_by_folder_or_suffix() -> None:
    """Check what counts as documentation or a changelog."""
    assert is_doc_file("doc/whats_new/v1.3.rst")
    assert is_doc_file("docs/api/cart.py")
    assert is_doc_file("CHANGES.md")
    assert is_doc_file("changelog.d/123.bugfix")
    assert not is_doc_file("pkg/cart.py")
