"""Tests for the in-sandbox retrieval script (context/probe.py), run here on small fake repos.

The probe normally runs inside an indexer sandbox, with git and Jedi. Here its parts run on a
folder of plain files (no git, no Jedi), which covers reading diffs, module names, which calls
`ast` resolves and which it leaves to Jedi, matching callers and tests (and refusing ambiguous
ones), the caps and their order, docstrings cut at 10 lines, and where changed tests go. The
git and Jedi parts are tested in a real indexer sandbox (test_retrieve_docker.py).

How to run (from the project folder):
    pytest tests/context/test_probe.py
"""

from pathlib import Path

import pytest

from pr_reviewer.context.probe import (
    Settings,
    collect,
    is_test_file,
    module_of,
    parse,
    parse_diff,
    touched_functions,
    walk_defs,
)
from pr_reviewer.reviewer.budget import is_test_file as budget_is_test_file

CORE = '''\
def helper(x):
    """Add one."""
    return x + 1


class Cart:
    def __init__(self, items):
        self.items = items

    def total(self):
        return sum(self.items)

    def average(self):
        return self.total() / len(self.items)
'''

USE = """\
from pkg import helper
from pkg.core import Cart


def report(items):
    cart = Cart(items)
    return cart.average()


def bump(x):
    return helper(x)
"""

TEST = """\
from pkg.core import Cart


def test_average():
    assert Cart([1, 3]).average() == 2
"""


def write(root: Path, files: dict[str, str]) -> str:
    """Write a fake repo.

    Args:
        root (Path): Its folder.
        files (dict[str, str]): File contents by path.

    Returns:
        str: The folder, as the probe takes it.
    """
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return str(root)


@pytest.fixture
def repo(tmp_path: Path) -> str:
    """Make a small package with a re-export, a caller and a test.

    Args:
        tmp_path (Path): pytest's temporary folder.

    Returns:
        str: The repo's folder.
    """
    files = {
        "pkg/__init__.py": "from .core import helper\n",
        "pkg/core.py": CORE,
        "pkg/use.py": USE,
        "tests/test_core.py": TEST,
    }
    return write(tmp_path, files)


def changed(path: str, *lines: int) -> dict[str, dict[str, set[int]]]:
    """Describe changed lines the way parse_diff() does.

    Args:
        path (str): The file.
        *lines (int): Its changed lines.

    Returns:
        dict[str, dict[str, set[int]]]: The changes.
    """
    return {path: {"lines": set(lines), "added": set(lines)}}


def kinds(pieces: list[dict[str, object]]) -> list[tuple[object, object]]:
    """List each piece's kind and function.

    Args:
        pieces (list[dict[str, object]]): The pieces.

    Returns:
        list[tuple[object, object]]: (kind, qualname) per piece, in order.
    """
    return [(piece["kind"], piece["qualname"]) for piece in pieces]


def test_a_changed_method_brings_its_callee_caller_and_test(repo: str) -> None:
    """Check the four kinds of pieces, in packing order, for a one-line change."""
    pieces, found, stats, _ = collect(repo, changed("pkg/core.py", 14), Settings(), False)
    assert [(c.item.qualname, c.fully_added) for c in found] == [("Cart.average", False)]
    assert kinds(pieces) == [
        ("changed", "Cart.average"),
        ("called", "Cart.total"),
        ("caller", "report"),
        ("test", "test_average"),
        ("file", ""),
    ]
    # self.total() is found by ast; len() is a builtin, outside the repo.
    assert (stats["resolved_ast"], stats["external"]) == (1, 1)
    assert pieces[2]["target"] == "Cart.average"
    assert pieces[2]["whole"] is True


def test_a_re_exported_name_is_left_to_jedi(repo: str) -> None:
    """Check helper, imported from the package that re-exports it, needs Jedi."""
    pieces, _, stats, _ = collect(repo, changed("pkg/use.py", 11), Settings(), False)
    assert stats["unresolved"] == 1
    assert ("called", "helper") not in kinds(pieces)


def test_a_method_name_shared_by_two_classes_is_refused_for_unknown_objects(
    tmp_path: Path,
) -> None:
    """Check cart.average() isn't matched when another class has an average() too."""
    box = "class Box:\n    def average(self):\n        return 0\n"
    root = write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/core.py": CORE, "pkg/use.py": USE, "pkg/box.py": box},
    )
    pieces, _, stats, _ = collect(root, changed("pkg/core.py", 14), Settings(), False)
    assert ("caller", "report") not in kinds(pieces)
    assert stats["ambiguous_skipped"] >= 1


def test_callers_are_capped_closest_first(tmp_path: Path) -> None:
    """Check the cap keeps the same file's callers first, then the same folder's."""
    lib = "def target():\n    return 1\n\n\ndef near():\n    return target()\n"
    files = {"pkg/__init__.py": "", "pkg/lib.py": lib, "other/__init__.py": ""}
    for number in range(3):
        files[f"pkg/same_{number}.py"] = (
            f"from pkg.lib import target\n\n\ndef in_pkg_{number}():\n    return target()\n"
        )
        files[f"other/far_{number}.py"] = (
            f"from pkg.lib import target\n\n\ndef far_{number}():\n    return target()\n"
        )
    root = write(tmp_path, files)
    pieces, _, stats, _ = collect(root, changed("pkg/lib.py", 2), Settings(callers=4), False)
    callers = [piece["qualname"] for piece in pieces if piece["kind"] == "caller"]
    assert callers == ["near", "in_pkg_0", "in_pkg_1", "in_pkg_2"]
    assert (stats["callers_found"], stats["callers_kept"]) == (7, 4)


def test_long_docstrings_are_cut_after_ten_lines(tmp_path: Path) -> None:
    """Check a long called function shows its signature and 10 docstring lines only."""
    doc = "\n".join(f"    line {n}" for n in range(15))
    body = "\n".join(f"    x{n} = {n}" for n in range(30))
    lib = f'def big(a, b):\n    """Big.\n{doc}\n    """\n{body}\n    return a\n'
    caller = "from lib import big\n\n\ndef uses():\n    return big(1, 2)\n"
    root = write(tmp_path, {"lib.py": lib, "main.py": caller})
    pieces, _, _, _ = collect(root, changed("main.py", 5), Settings(), False)
    called = next(piece for piece in pieces if piece["kind"] == "called")
    text = str(called["text"])
    assert called["whole"] is False
    assert "line 8" in text and "line 9" not in text
    assert "[... rest of the docstring]" in text
    assert "x0 = 0" not in text


def test_edited_tests_go_after_related_tests_and_new_functions_are_skipped(
    repo: str, tmp_path: Path
) -> None:
    """Check an edited test is a changed_test piece after the tests, and an added one isn't."""
    test_path = Path(repo) / "tests" / "test_core.py"
    test_path.write_text(TEST + "\n\ndef test_new():\n    assert True\n", encoding="utf-8")
    changes = {
        "pkg/core.py": {"lines": {14}, "added": {14}},
        "tests/test_core.py": {"lines": {5, 8, 9}, "added": {5, 8, 9}},
    }
    pieces, found, stats, _ = collect(repo, changes, Settings(), False)
    assert [c.item.qualname for c in found] == ["Cart.average", "test_average", "test_new"]
    order = [kind for kind, _ in kinds(pieces)]
    assert order.index("changed_test") > order.index("caller")
    assert ("changed_test", "test_new") not in kinds(pieces)
    assert stats["changed_fully_added"] == 1


def test_a_piece_bigger_than_the_budget_is_skipped(repo: str) -> None:
    """Check pieces that could never fit are dropped and counted."""
    pieces, _, stats, _ = collect(repo, changed("pkg/core.py", 14), Settings(max_piece_chars=60))
    assert all(len(str(piece["text"])) <= 60 for piece in pieces)
    assert stats["too_big"] >= 1


DIFF = """\
diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -3,2 +3,3 @@ def f():
--- not a header, a removed line starting with dashes
-x = 1
+y = 2
+++ not a header either
+z = 3
@@ -10,0 +12 @@ def g():
+inserted = True
diff --git a/old.py b/new.py
--- a/old.py
+++ b/new.py
@@ -5 +5 @@
-a
+b
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-x
+y
"""


def test_diffs_are_read_by_hunk_counts() -> None:
    """Check lines that look like headers inside a hunk don't confuse the reader."""
    new = parse_diff(DIFF, "new")
    assert sorted(new) == ["new.py", "pkg/a.py"]
    assert new["pkg/a.py"]["added"] == {3, 4, 5, 12}
    old = parse_diff(DIFF, "old")
    assert sorted(old) == ["old.py", "pkg/a.py"]
    assert old["pkg/a.py"]["lines"] == {3, 4}
    # The insertion has no line on the old side: it happened after line 10.
    assert old["pkg/a.py"]["between"] == {10}


def test_an_insertion_touches_a_function_only_from_inside() -> None:
    """Check code added right after a function's last line doesn't count as changing it."""
    source = "def f():\n    a = 1\n    return a\n\n\ndef g():\n    return 2\n"
    tree = parse(source)
    assert tree is not None
    defs = walk_defs(tree, "m.py")
    after_f = {"lines": set(), "added": set(), "between": {3}}
    inside_f = {"lines": set(), "added": set(), "between": {2}}
    assert touched_functions(defs, after_f) == []
    assert [d.qualname for d in touched_functions(defs, inside_f)] == ["f"]
    changed_g = {"lines": {7}, "added": {7}, "between": set()}
    assert [d.qualname for d in touched_functions(defs, changed_g)] == ["g"]


def test_module_names_start_at_the_top_package(tmp_path: Path) -> None:
    """Check a lib/ or src/ layout gives the importable name, not the folder path."""
    root = write(
        tmp_path,
        {"lib/mpl/__init__.py": "", "lib/mpl/axes/__init__.py": "", "lib/mpl/axes/_base.py": ""},
    )
    assert module_of(root, "lib/mpl/axes/_base.py") == ("mpl.axes._base", False, "lib")
    assert module_of(root, "lib/mpl/__init__.py") == ("mpl", True, "lib")
    assert module_of(root, "setup.py") == ("setup", False, "")


@pytest.mark.parametrize(
    "path",
    ["pkg/tests/test_a.py", "test_x.py", "a_test.py", "conftest.py", "pkg/a.py", "testing/u.py"],
)
def test_the_probe_and_the_budget_agree_on_test_files(path: str) -> None:
    """Check the probe's copy of the test-file rule matches the reviewer's."""
    assert is_test_file(path) == budget_is_test_file(path)
