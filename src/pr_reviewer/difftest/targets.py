"""Finding the tests a change added or edited, to run only those.

A test file can hold thousands of tests, but a PR or a fix usually adds or edits a few. Those
are the ones whose outcome is known, so the validation runs only them:
1. changed_lines() reads `git diff -U0` and gives, per file, the line numbers that are new or
   changed in the newer version;
2. find_tests() lists a file's test functions and methods with their line ranges, using
   Python's own parser (ast). Parsing only reads the text; nothing in the file runs;
3. changed_tests() keeps the tests whose lines (decorators included) were touched.
A test becomes a pytest node ID ("pkg/tests/test_a.py::TestA::test_b") or, for Django's
runner, a label ("app.tests.TestA.test_b").

A method of a base class or mixin (such as `ZarrBase.test_x`) never runs under its own
class's name: the runner only collects test classes, and the method runs in each of them that
inherits it. node_ids() gives those, from the classes in the same file. A test that only runs
through classes in other files gets no ID, so the caller can note it instead of asking the
runner for a test it can't find.

Imported by other modules; the file isn't run directly.
"""

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from pr_reviewer.difftest.outcomes import Runner
from pr_reviewer.setup_agent.selection import django_labels

# A hunk header of a unified diff: "@@ -10,2 +12,3 @@".
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class FoundTest:
    """A test function or method in a file.

    Attributes:
        name (str): The function's name, such as "test_b".
        cls (str | None): Its class, such as "TestA", or None for a module-level test.
        start (int): Its first line, decorators included.
        end (int): Its last line.
    """

    name: str
    cls: str | None
    start: int
    end: int


@dataclass(frozen=True)
class FoundClass:
    """A class at the top level of a test file.

    Attributes:
        name (str): Its name, such as "TestA".
        bases (tuple[str, ...]): Its bases' names, such as ("Base", "TestCase") for
            `class TestA(Base, unittest.TestCase)`.
        methods (frozenset[str]): The functions it defines itself.
        off (bool): Whether it sets `__test__ = False`, which tells pytest to skip it.
    """

    name: str
    bases: tuple[str, ...]
    methods: frozenset[str]
    off: bool


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Read which lines a diff adds or changes in the newer version of each file.

    A pure deletion marks the lines on both sides of where it was, so a test that only lost
    a line still counts as changed.

    Args:
        diff (str): The output of `git diff -U0 <old> <new>`.

    Returns:
        dict[str, set[int]]: Line numbers in the newer version, by file path. Deleted files
        are left out.
    """
    lines: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = target[2:] if target.startswith("b/") else None
            if current is not None:
                lines.setdefault(current, set())
            continue
        hunk = _HUNK.match(line)
        if hunk and current is not None:
            start, count = int(hunk.group(1)), int(hunk.group(2) or "1")
            if count == 0:
                lines[current].update((start, start + 1))
            else:
                lines[current].update(range(start, start + count))
    return lines


def _is_test(name: str) -> bool:
    """Tell whether a function name is a test's.

    Args:
        name (str): The function's name.

    Returns:
        bool: True for names starting with "test".
    """
    return name.startswith("test")


def find_tests(source: str) -> list[FoundTest]:
    """List a test file's test functions and methods, with their line ranges.

    Args:
        source (str): The file's text.

    Returns:
        list[FoundTest]: Module-level test functions, and test methods of classes, in
        file order. Empty when the file isn't valid Python for this parser.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[FoundTest] = []

    def add(node: ast.FunctionDef | ast.AsyncFunctionDef, cls: str | None) -> None:
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        found.append(FoundTest(node.name, cls, start, node.end_lineno or node.lineno))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_test(node.name):
            add(node, None)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and _is_test(item.name):
                    add(item, node.name)
    return found


def changed_tests(source: str, lines: Iterable[int]) -> list[FoundTest]:
    """Keep the tests a change touched.

    Args:
        source (str): The test file's newer text.
        lines (Iterable[int]): Its added or changed line numbers.

    Returns:
        list[FoundTest]: The tests with at least one touched line, in file order.
    """
    touched = set(lines)
    return [
        test
        for test in find_tests(source)
        if any(test.start <= line <= test.end for line in touched)
    ]


def node_id(path: str, test: FoundTest, runner: Runner) -> str:
    """Give a test's ID for the runner.

    Args:
        path (str): The test file, relative to the repo root.
        test (FoundTest): The test.
        runner (Runner): "pytest" or "django".

    Returns:
        str: A pytest node ID such as "tests/test_a.py::TestA::test_b", or a Django label
        such as "app.tests.TestA.test_b" ("" when the file isn't under Django's tests/).
    """
    parts = [test.cls, test.name] if test.cls else [test.name]
    if runner == "django":
        labels = django_labels([path])
        return ".".join([labels[0], *parts]) if labels else ""
    return "::".join([path, *parts])


def _base_name(node: ast.expr) -> str:
    """Give a base class's name as written, without its module.

    Args:
        node (ast.expr): The base, such as `Base` or `unittest.TestCase`.

    Returns:
        str: Such as "Base" or "TestCase"; "" for anything else, such as `Generic[T]`.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _turns_off_pytest(node: ast.ClassDef) -> bool:
    """Tell whether a class body sets `__test__ = False`.

    Args:
        node (ast.ClassDef): The class.

    Returns:
        bool: True when it does.
    """
    for item in node.body:
        if (
            isinstance(item, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__test__" for t in item.targets)
            and isinstance(item.value, ast.Constant)
            and item.value.value is False
        ):
            return True
    return False


def find_classes(source: str) -> dict[str, FoundClass]:
    """List the classes at the top level of a test file.

    Args:
        source (str): The file's text.

    Returns:
        dict[str, FoundClass]: The classes by name, in file order. Empty when the file isn't
        valid Python for this parser.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    classes: dict[str, FoundClass] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = frozenset(
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            )
            bases = tuple(name for name in map(_base_name, node.bases) if name)
            classes[node.name] = FoundClass(node.name, bases, methods, _turns_off_pytest(node))
    return classes


def _is_test_case(name: str, classes: Mapping[str, FoundClass], seen: frozenset[str]) -> bool:
    """Tell whether a class is a unittest TestCase, as far as the file shows.

    Args:
        name (str): The class's name.
        classes (Mapping[str, FoundClass]): The file's classes.
        seen (frozenset[str]): Classes already on the way here, so a loop can't recurse forever.

    Returns:
        bool: True when it, or a base in the file, inherits from a class named "...TestCase".
    """
    found = classes.get(name)
    if found is None or name in seen:
        return False
    return any(
        base.endswith("TestCase") or _is_test_case(base, classes, seen | {name})
        for base in found.bases
    )


def _collected(name: str, classes: Mapping[str, FoundClass], runner: Runner) -> bool:
    """Tell whether the runner collects a class's tests under the class's own name.

    pytest collects classes named "Test..." (unless they define __init__ or set
    `__test__ = False`) and unittest TestCases. Django's runner collects TestCases only; a
    base defined in another file is taken to be one unless it's named "...Mixin".

    Args:
        name (str): The class's name.
        classes (Mapping[str, FoundClass]): The file's classes.
        runner (Runner): "pytest" or "django".

    Returns:
        bool: True when the class's tests run under its name.
    """
    found = classes[name]
    test_case = _is_test_case(name, classes, frozenset())
    if runner == "django":
        imported = [base for base in found.bases if base not in classes and base != "object"]
        return test_case or any(not base.endswith("Mixin") for base in imported)
    if found.off:
        return False
    return test_case or (name.startswith("Test") and "__init__" not in found.methods)


def running_classes(
    test: FoundTest, classes: Mapping[str, FoundClass], runner: Runner
) -> list[str]:
    """Give the classes in the file whose runs include a test method.

    They're the method's own class and every class in the file that inherits the method
    without redefining it, kept when the runner collects them.

    Args:
        test (FoundTest): The test method.
        classes (Mapping[str, FoundClass]): The file's classes.
        runner (Runner): "pytest" or "django".

    Returns:
        list[str]: The classes, in file order. Empty when only other files' classes run it.
    """
    if test.cls is None or test.cls not in classes:
        return [test.cls] if test.cls else []
    inherit = {test.cls}
    # A subclass always comes after its bases in the file, so one pass finds them all.
    for name, found in classes.items():
        redefines = test.name in found.methods
        if name not in inherit and not redefines and any(b in inherit for b in found.bases):
            inherit.add(name)
    return [name for name in classes if name in inherit and _collected(name, classes, runner)]


def node_ids(
    path: str, test: FoundTest, classes: Mapping[str, FoundClass], runner: Runner
) -> list[str]:
    """Give every ID a test runs under: one per class whose runs include it.

    Args:
        path (str): The test file, relative to the repo root.
        test (FoundTest): The test.
        classes (Mapping[str, FoundClass]): The file's classes, from find_classes().
        runner (Runner): "pytest" or "django".

    Returns:
        list[str]: The IDs, such as ["t.py::TestZarrDict::test_x", "t.py::TestZarrDir::test_x"]
        for a method of a ZarrBase class both inherit from. Empty when no class in the file
        runs it, or when Django can't name the file.
    """
    if test.cls is None:
        found = node_id(path, test, runner)
        return [found] if found else []
    ids = [
        node_id(path, replace(test, cls=name), runner)
        for name in running_classes(test, classes, runner)
    ]
    return [found for found in ids if found]
