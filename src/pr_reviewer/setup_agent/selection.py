"""Picking the repo's existing tests that relate to a PR, to check an environment with.

The whole test suite of a repo like sympy takes far too long, and most of it has nothing to do
with the PR. So an environment is checked with the test files near the code the PR changes,
as they exist at the base:
1. test files the PR itself edits (their old versions);
2. test files named after a changed module, such as test_ridge.py for _ridge.py, preferring
   ones in the same package;
3. if still none, test files in the nearest tests folder of a changed module;
4. then test files in a top-level tests/ folder that mirrors the package, as pylint has
   (pylint/checkers/classes/x.py is tested in tests/checkers/);
5. last, test files in a top-level tests/ folder named after the module or its package, the
   way Django lays out its tests.
Test files are test_*.py, *_test.py, tests.py, and unittest_*.py (pylint's name for them).
At most MAX_FILES files are picked, so a check stays within its time limit.

A PR that changes no Python code (only config, CI or docs) has no related tests; it's checked
with the smallest few files of the repo's main tests folder instead (fallback_tests()). Small
files are the quickest and lightest check: some big ones need more memory than a sandbox has
(matplotlib's test_agg.py has a test that uses about 2 GB).

For Django's own test runner, files become test labels, such as "admin_views.tests".

Imported by other modules; the file isn't run directly.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath

# Most test files run to check an environment.
MAX_FILES = 5


def is_test_file(path: str) -> bool:
    """Tell whether a path is a test module.

    Args:
        path (str): A file path in the repo, such as "pkg/tests/test_a.py".

    Returns:
        bool: True for test_*.py, unittest_*.py, *_test.py and tests.py files, not
        conftest.py or __init__.py.
    """
    name = PurePosixPath(path).name
    if not name.endswith(".py") or name in ("conftest.py", "__init__.py"):
        return False
    return (
        name.startswith(("test_", "unittest_")) or name.endswith("_test.py") or name == "tests.py"
    )


def _tested_name(path: str) -> str:
    """Give the module name a test file is named after.

    Args:
        path (str): A test file, such as "tests/checkers/unittest_classes.py".

    Returns:
        str: Such as "classes".
    """
    stem = PurePosixPath(path).stem
    return stem.removeprefix("unittest_").removeprefix("test_").removesuffix("_test")


def _module_names(path: str) -> set[str]:
    """Give the names a source file's test file might be named after.

    Args:
        path (str): A source file, such as "sklearn/linear_model/_ridge.py".

    Returns:
        set[str]: Such as {"_ridge", "ridge"}.
    """
    stem = PurePosixPath(path).stem
    return {stem, stem.lstrip("_")} - {"", "__init__"}


def _shared_prefix(a: str, b: str) -> int:
    """Count the leading folders two paths share.

    Args:
        a (str): A path.
        b (str): Another path.

    Returns:
        int: How many leading folders are the same.
    """
    count = 0
    for x, y in zip(PurePosixPath(a).parts[:-1], PurePosixPath(b).parts[:-1], strict=False):
        if x != y:
            break
        count += 1
    return count


def related_tests(changed: Iterable[str], repo_files: Sequence[str]) -> list[str]:
    """Pick the existing test files that relate to a PR's changes.

    Args:
        changed (Iterable[str]): Files the PR's reviewed commits change.
        repo_files (Sequence[str]): Python files in the repo at the base.

    Returns:
        list[str]: Up to MAX_FILES test files, most related first. Empty when none relate.
    """
    existing = set(repo_files)
    tests = [f for f in repo_files if is_test_file(f)]
    picked: list[str] = []

    def add(paths: Iterable[str]) -> None:
        for path in paths:
            if path not in picked and len(picked) < MAX_FILES:
                picked.append(path)

    changed = list(dict.fromkeys(changed))
    sources = [f for f in changed if f.endswith(".py") and not is_test_file(f)]
    add(f for f in changed if is_test_file(f) and f in existing)
    for source in sources:
        names = _module_names(source)
        matches = [t for t in tests if _tested_name(t) in names]
        add(sorted(matches, key=lambda t: -_shared_prefix(t, source)))
    if picked:
        return picked
    for source in sources:
        folder = PurePosixPath(source).parent
        while str(folder) not in (".", ""):
            nearby = sorted(t for t in tests if PurePosixPath(t).parent == folder / "tests")
            if nearby:
                add(nearby[:3])
                break
            folder = folder.parent
    if picked:
        return picked
    # A top-level tests/ folder mirroring the package, deepest match first. Files named
    # after the module or its folder go first.
    for source in sources:
        path = PurePosixPath(source)
        names = _module_names(source) | {path.parent.name}
        inner = path.parts[1:-1]  # the folders below the top-level package
        for depth in range(len(inner), 0, -1):
            mirror = PurePosixPath("tests", *inner[:depth])
            here = [t for t in tests if PurePosixPath(t).parent == mirror]
            if here:

                def named_first(test: str, names: set[str] = names) -> tuple[bool, str]:
                    return (not any(name in _tested_name(test) for name in names), test)

                add(sorted(here, key=named_first)[:3])
                break
    if picked:
        return picked
    # Last, a top-level tests/ folder named after the module or its package, as Django has
    # (django/contrib/admin/options.py is tested in tests/admin_views/ and similar).
    for source in sources:
        path = PurePosixPath(source)
        names = _module_names(source) | {path.parent.name}
        add(
            sorted(
                t
                for t in tests
                if PurePosixPath(t).parts[0] == "tests"
                and len(PurePosixPath(t).parts) > 2
                and any(name and name in PurePosixPath(t).parts[1] for name in names)
            )[:3]
        )
    return picked


def main_test_folder(repo_files: Sequence[str]) -> str | None:
    """Find the repo's main tests folder: the one holding the most test files.

    Args:
        repo_files (Sequence[str]): Python files in the repo at the base.

    Returns:
        str | None: The folder, such as "lib/matplotlib/tests", or "." for the top. None when
        the repo has no test files.
    """
    folders = Counter(str(PurePosixPath(f).parent) for f in repo_files if is_test_file(f))
    if not folders:
        return None
    return max(folders, key=lambda folder: (folders[folder], -len(folder)))


def tree_sizes(listing: str, folder: str) -> dict[str, int]:
    """Read file sizes from what `git ls-tree -l` printed for one folder.

    Args:
        listing (str): Its output: a line per entry, such as
            "100644 blob <id>    1234<TAB>test_a.py". Folders show "tree" and no size.
        folder (str): The folder listed, such as "lib/matplotlib/tests", or "." for the top.

    Returns:
        dict[str, int]: Each file's size in bytes, by its path in the repo.
    """
    sizes = {}
    for line in listing.splitlines():
        info, _, name = line.partition("\t")
        fields = info.split()
        if len(fields) != 4 or fields[1] != "blob" or not fields[3].isdigit():
            continue
        sizes[name if folder == "." else f"{folder}/{name}"] = int(fields[3])
    return sizes


def fallback_tests(
    repo_files: Sequence[str], sizes: Mapping[str, int], count: int = 3
) -> list[str]:
    """Pick a few of the repo's tests, for a PR that changes no Python code.

    They are the smallest test files of the repo's main tests folder (main_test_folder()).

    Args:
        repo_files (Sequence[str]): Python files in the repo at the base.
        sizes (Mapping[str, int]): Sizes in bytes of the main folder's files (tree_sizes()).
            A file with no size counts as the biggest.
        count (int): Most files to pick.

    Returns:
        list[str]: Up to `count` test files, in path order. Empty when the repo has none.
    """
    main = main_test_folder(repo_files)
    tests = [f for f in repo_files if is_test_file(f) and str(PurePosixPath(f).parent) == main]
    smallest = sorted(tests, key=lambda t: (sizes.get(t, math.inf), t))[:count]
    return sorted(smallest)


def django_labels(test_files: Sequence[str]) -> list[str]:
    """Turn test files under tests/ into labels for Django's runtests.py.

    Args:
        test_files (Sequence[str]): Such as ["tests/admin_views/tests.py"].

    Returns:
        list[str]: Such as ["admin_views.tests"]. Files outside tests/ are left out.
    """
    labels = []
    for path in test_files:
        parts = PurePosixPath(path).with_suffix("").parts
        if len(parts) > 1 and parts[0] == "tests":
            labels.append(".".join(parts[1:]))
    return labels
