"""Tests for the survey's pure functions (evidence/survey.py): no Docker needed.

Checks finding the function around a line, showing code around a claim within a size, reading
diff hunks and picking the nearest, collecting the names a PR adds, choosing where a test goes,
listing the conftest.py files and fixtures it can use, picking an example test, and building
materials without a survey from the PR's own diff.

How to run (from the project folder):
    pytest tests/evidence/test_survey.py
"""

from pr_reviewer.core.schemas import Claim
from pr_reviewer.evidence.inputs import Candidate
from pr_reviewer.evidence.survey import (
    added_names,
    build_materials,
    choose_test_file,
    code_view,
    conftests,
    defined_names,
    enclosing_function,
    example_test,
    fallback_materials,
    fixtures,
    near_hunks,
    parse_hunks,
    place_test,
    short_names,
)
from tests.reviewer.helpers import make_pr

SOURCE = """\
import math


class Cart:
    def __init__(self, items):
        self.items = items

    @property
    def total(self):
        return sum(self.items)

    def average(self):
        def helper(n):
            return n / len(self.items)

        return helper(self.total)


LIMIT = 3
"""

DIFF = """\
diff --git a/pkg/cart.py b/pkg/cart.py
index 1..2 100644
--- a/pkg/cart.py
+++ b/pkg/cart.py
@@ -5,3 +5,3 @@ class Cart:
     def __init__(self, items):
-        self.items = list(items)
+        self.items = items

@@ -12,4 +12,6 @@ class Cart:
     def average(self):
+        def helper(n):
+            return n / len(self.items)

-        return self.total / len(self.items)
+        return helper(self.total)
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+def fresh_name():
+    return 1
"""

LISTING = [
    "pkg/__init__.py",
    "pkg/cart.py",
    "pkg/tests/__init__.py",
    "pkg/tests/conftest.py",
    "pkg/tests/test_cart.py",
    "pkg/tests/test_other.py",
    "pkg/sub/deep.py",
    "tests/test_cart.py",
    "conftest.py",
]

CONFTEST = '''\
import pytest


@pytest.fixture
def cart(tmp_path):
    """A cart with three items.

    More words.
    """
    return None


@pytest.fixture(scope="module")
def empty():
    return None


def not_a_fixture():
    pass
'''

TESTS = """\
import pytest
from pkg.cart import Cart


def test_total():
    assert Cart([1]).total == 1


def test_average():
    assert Cart([2, 4]).average() == 3
"""


def candidate(line: int = 14, file: str = "pkg/cart.py", finding: str = "f2") -> Candidate:
    """Build a candidate claiming a problem at a line.

    Args:
        line (int): The claimed line.
        file (str): The claimed file.
        finding (str): Its finding ID.

    Returns:
        Candidate: The candidate.
    """
    claim = Claim(
        file=file,
        start_line=line,
        end_line=line,
        category="F.4",
        summary="average divides by zero for an empty cart.",
        expected_failure="ZeroDivisionError",
        confidence=0.8,
    )
    return Candidate("o__r-1", finding, claim)


def test_the_innermost_function_holds_the_line() -> None:
    """Check methods get their class in their name, nested functions win, decorators count."""
    assert enclosing_function(SOURCE, 14).qualname == "Cart.average.helper"  # type: ignore[union-attr]
    assert enclosing_function(SOURCE, 16).qualname == "Cart.average"  # type: ignore[union-attr]
    span = enclosing_function(SOURCE, 8)
    assert span is not None and (span.qualname, span.start, span.end) == ("Cart.total", 8, 10)
    assert enclosing_function(SOURCE, 19) is None
    assert enclosing_function("def broken(:\n", 1) is None


def test_code_around_a_claim_is_numbered_and_fits() -> None:
    """Check the whole function is shown with line numbers, or the lines near the claim."""
    name, text = code_view(SOURCE, 16, 16)
    assert name == "Cart.average"
    assert text.splitlines()[0] == "12 |     def average(self):"
    assert "16 |         return helper(self.total)" in text
    name, outside = code_view(SOURCE, 19, 19)
    assert name == "" and "19 | LIMIT = 3" in outside and " 1 | import math" in outside
    _, small = code_view(SOURCE, 16, 16, max_chars=80)
    assert "16 |" in small and len(small) <= 80


def test_hunks_are_split_by_file_and_the_nearest_come_first() -> None:
    """Check hunk line ranges, and that the nearest hunk is kept when room is short."""
    hunks = parse_hunks(DIFF)
    assert sorted(hunks) == ["pkg/cart.py", "pkg/new.py"]
    first, second = hunks["pkg/cart.py"]
    assert (first.start, first.end, second.start, second.end) == (5, 7, 12, 17)
    both = near_hunks(hunks["pkg/cart.py"], 15, 15)
    assert both.startswith("--- pkg/cart.py\n@@ -5,3") and "+        return helper" in both
    near = near_hunks(hunks["pkg/cart.py"], 15, 15, max_chars=len(second.text) + 20)
    assert "def average" in near and "__init__" not in near
    assert near_hunks([], 1, 1) == ""


def test_the_names_a_pr_adds_include_new_modules() -> None:
    """Check names on added lines, and the module name of an added file."""
    names = added_names(
        DIFF.replace("@@ -5,3 +5,3 @@", "@@ -6 +6 @@").replace(
            "@@ -12,4 +12,6 @@", "@@ -13,0 +13,2 @@"
        )
    )
    assert {"helper", "fresh_name", "new"} <= names
    assert "list" not in names, "a removed line's names don't count"


def test_names_a_line_defines() -> None:
    """Check functions with their arguments, classes, assignments and imported names."""
    assert defined_names("    def total(self, items: list[int], *, scale=1, **kw):") == {
        "total",
        "self",
        "items",
        "scale",
        "kw",
    }
    assert defined_names("class Cart(Base):") == {"Cart"}
    assert defined_names("        self.limit = 3") == {"limit"}
    assert defined_names("LIMIT: int = 3") == {"LIMIT"}
    assert defined_names("from pkg.cart import Cart, total as sum_total") == {"Cart", "sum_total"}
    assert defined_names("    return x.shape == y") == set()
    assert defined_names("    if a == b:") == set()


def test_the_test_goes_next_to_the_closest_test_file() -> None:
    """Check hints win (whatever their name), then a test named after the module, then the
    nearest folder, a tests folder counting as nearer."""
    assert choose_test_file(LISTING, "pkg/cart.py", ["pkg/tests/test_other.py"]) == (
        "pkg/tests/test_other.py"
    )
    assert choose_test_file(
        [*LISTING, "testing/approx.py"], "pkg/cart.py", ["testing/approx.py"]
    ) == ("testing/approx.py")
    assert choose_test_file(
        ["src/a/b.py", "bench/test_x.py", "testing/test_y.py"], "src/a/b.py", []
    ) == ("testing/test_y.py")
    assert choose_test_file(LISTING, "pkg/cart.py", ["gone/test_x.py"]) == "pkg/tests/test_cart.py"
    assert choose_test_file(LISTING, "pkg/sub/deep.py", []) == "pkg/tests/test_cart.py"
    assert choose_test_file(["pkg/cart.py"], "pkg/cart.py", []) is None
    assert place_test("pkg/tests/test_cart.py", "f3") == "pkg/tests/test_pr_reviewer_f3.py"
    assert place_test("test_top.py", "f3") == "test_pr_reviewer_f3.py"
    assert place_test(None, "f3") is None


def test_the_fixtures_come_from_every_conftest_above_the_test() -> None:
    """Check the conftest.py files pytest loads, nearest first, and the fixtures they define."""
    assert conftests(LISTING, "pkg/tests/test_pr_reviewer_f1.py") == [
        "pkg/tests/conftest.py",
        "conftest.py",
    ]
    assert fixtures(CONFTEST) == ["cart(tmp_path): A cart with three items.", "empty()"]
    assert fixtures("def broken(:") == []


def test_the_example_test_prefers_one_using_the_changed_function() -> None:
    """Check the example holds the imports and the test that uses the function's name."""
    text = example_test("pkg/tests/test_cart.py", TESTS, ["average", "Cart"])
    assert text.startswith("# pkg/tests/test_cart.py\nimport pytest\nfrom pkg.cart import Cart")
    assert "def test_average" in text and "def test_total" not in text
    first = example_test("pkg/tests/test_cart.py", TESTS, ["nothing"])
    assert "def test_total" in first
    assert len(example_test("t.py", TESTS, [], max_chars=60)) <= 60
    assert short_names("Cart.average") == ["average", "Cart"]
    assert short_names("Cart.__init__") == ["Cart"]


def test_materials_hold_the_code_the_hunks_and_the_test_place() -> None:
    """Check a candidate's materials put the survey's pieces together."""
    hunks = parse_hunks(DIFF)
    materials = build_materials(
        candidate(),
        {"pkg/cart.py": SOURCE},
        hunks,
        "pkg/tests/test_cart.py",
        TESTS,
        ["cart(tmp_path)"],
    )
    assert materials.function == "Cart.average.helper"
    assert "14 |" in materials.code and materials.hunks.startswith("--- pkg/cart.py")
    assert materials.test_path == "pkg/tests/test_pr_reviewer_f2.py"
    assert "def test_average" in materials.example, "no test names helper, one names average"
    assert materials.fixtures == ("cart(tmp_path)",)
    unread = build_materials(candidate(), {"pkg/cart.py": None}, {}, None, None, [])
    assert (unread.code, unread.test_path) == ("", None)


def test_fallback_materials_use_the_prs_own_diff() -> None:
    """Check materials without a survey: the PR's own hunks, no code and no test place."""
    patch = "@@ -1,2 +1,3 @@\n def average(cart):\n+    total = sum(cart)\n     return total"
    pr = make_pr(files=[("pkg/cart.py", patch)])
    materials = fallback_materials(pr, candidate(line=2))
    assert materials.hunks == f"--- pkg/cart.py\n{patch}"
    assert (materials.code, materials.test_path) == ("", None)
