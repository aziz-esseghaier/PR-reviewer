"""Signs that a PR changed some behavior on purpose, for the intent check on a proof.

A test proves that the PR changed what some code does. That's a bug only if the change wasn't
meant: a PR may change a default or start raising an error on bad input deliberately. The PR
usually leaves a trace when it does, and our code looks for it in the PR's own diff, without a
model call:
- lines its tests add that mention the changed function or its class (such as an assert on
  the new value, or `pytest.raises` of the new error). The class counts even when its name is
  short, because tests often call a private method through the public API (`Ci(x).series`
  for `Ci._eval_aseries`);
- lines its docs or changelog add that mention the function or the error;
- warnings (deprecations) added to the changed function;
- notes the claim's file adds about a change, such as "changed to" or `versionchanged` in a
  docstring;
- deprecation warnings the claim's file removes, and warning checks its tests remove (a
  finished deprecation).
Only a proof with signs goes to the intent check, which decides whether the change was
deliberate.

The PR's text is untrusted: the signs go to the model as data, never as instructions.

Imported by other modules; the file isn't run directly.
"""

import re
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.setup_agent.selection import is_test_file

# Most signs kept, and most characters of each.
MAX_SIGNS = 12
SIGN_CHARS = 200

# Words in a doc or changelog file's path.
_DOC_PARTS = ("doc", "docs", "changelog", "changes", "whatsnew", "release", "news", "history")

# Suffixes of doc files.
_DOC_SUFFIXES = (".rst", ".md", ".txt")

# An added line that warns, such as `warnings.warn(..., DeprecationWarning)`.
_WARNS = re.compile(r"warn\(|DeprecationWarning|FutureWarning|PendingDeprecationWarning|deprecat")

# A note that says something changed on purpose, such as "The default changed to 'highs' in
# version 1.4" or ".. versionchanged:: 1.4".
_CHANGE_NOTE = re.compile(
    r"versionchanged|versionadded|deprecated::|changed (?:to|from)\b|now defaults?\b|"
    r"default (?:is|was) now\b|no longer\b",
    re.IGNORECASE,
)

# The order kinds of signs are taken in when there are more than MAX_SIGNS: one of each kind
# first, so a long list of test lines can't hide a changelog note.
_KINDS = ("test", "docs", "note", "warning", "removed warning", "removed test")


def added_lines(pr: Instance) -> list[tuple[str, str]]:
    """List the lines a PR's reviewed commits add, with their file.

    Args:
        pr (Instance): The PR.

    Returns:
        list[tuple[str, str]]: (file, line) for every added line, in commit order, without
        the leading "+".
    """
    return _changed_lines(pr, "+")


def removed_lines(pr: Instance) -> list[tuple[str, str]]:
    """List the lines a PR's reviewed commits remove, with their file.

    Args:
        pr (Instance): The PR.

    Returns:
        list[tuple[str, str]]: (file, line) for every removed line, in commit order, without
        the leading "-".
    """
    return _changed_lines(pr, "-")


def _changed_lines(pr: Instance, mark: str) -> list[tuple[str, str]]:
    """List the lines a PR's reviewed commits add or remove, with their file.

    Args:
        pr (Instance): The PR.
        mark (str): "+" for added lines, "-" for removed ones.

    Returns:
        list[tuple[str, str]]: (file, line) for every such line, in commit order, without
        the mark.
    """
    header = mark * 3
    found = []
    for commit in pr.pr_commits:
        for patch in commit.diff:
            for line in patch.patch.splitlines():
                if line.startswith(mark) and not line.startswith(header):
                    found.append((patch.file, line[1:]))
    return found


def names_for_tests(names: Sequence[str]) -> list[str]:
    """Pick the names whose use in a test line counts as a sign.

    Args:
        names (Sequence[str]): The changed function's names, such as ["_eval_aseries", "Ci"].

    Returns:
        list[str]: Names longer than 2 characters, and class names (capitalized) of 2, which
        tests call to reach a private method.
    """
    return [name for name in names if len(name) > 2 or (len(name) == 2 and name[0].isupper())]


def is_doc_file(path: str) -> bool:
    """Tell whether a file is documentation or a changelog.

    Args:
        path (str): The file, such as "doc/whats_new/v1.3.rst".

    Returns:
        bool: True for .rst, .md and .txt files, and files under a docs or changelog folder.
    """
    lower = path.lower()
    parts = PurePosixPath(lower).parts
    if any(part.startswith(_DOC_PARTS) for part in parts[:-1]):
        return True
    name = parts[-1] if parts else ""
    return name.endswith(_DOC_SUFFIXES) or name.startswith(("changelog", "changes", "news"))


def _mentions(line: str, names: Iterable[str]) -> bool:
    """Tell whether a line uses any of some names as a whole word.

    Args:
        line (str): The line.
        names (Iterable[str]): The names.

    Returns:
        bool: True when one appears.
    """
    return any(re.search(rf"\b{re.escape(name)}\b", line) for name in names)


def signs(
    pr: Instance,
    claim_file: str,
    names: Sequence[str],
    exception: str | None,
    function_code: str,
) -> list[str]:
    """Find signs in a PR's diff that a change to a function was deliberate.

    Args:
        pr (Instance): The PR.
        claim_file (str): The file of the claim the test proves.
        names (Sequence[str]): The changed function's names, such as ["total", "Cart"].
        exception (str | None): The claimed exception, such as "ValueError".
        function_code (str): The changed function's code, to spot warnings added inside it.

    Returns:
        list[str]: One line per sign, such as "tests/test_cart.py (test): + assert
        total([]) == 0", at most MAX_SIGNS, taking one of each kind in turn.
    """
    wanted = [name for name in names if len(name) > 2]
    tested = names_for_tests(names)
    error = [exception.rsplit(".", 1)[-1]] if exception else []
    by_kind: dict[str, list[str]] = {kind: [] for kind in _KINDS}
    for file, line in added_lines(pr):
        text = line.strip()
        if not text:
            continue
        kind = ""
        if is_test_file(file):
            kind = "test" if _mentions(text, tested) else ""
        elif is_doc_file(file) and _mentions(text, [*wanted, *error]):
            kind = "docs"
        elif file == claim_file and _WARNS.search(text) and text in function_code:
            kind = "warning"
        elif file == claim_file and _CHANGE_NOTE.search(text):
            kind = "note"
        if kind:
            by_kind[kind].append(f"{file} ({kind}): + {text}"[:SIGN_CHARS])
    for file, line in removed_lines(pr):
        text = line.strip()
        if not text or not _WARNS.search(text):
            continue
        kind = ""
        if is_test_file(file):
            kind = "removed test"
        elif file == claim_file:
            kind = "removed warning"
        if kind:
            by_kind[kind].append(f"{file} ({kind}): - {text}"[:SIGN_CHARS])
    return _take_in_turn(by_kind, MAX_SIGNS)


def _take_in_turn(by_kind: dict[str, list[str]], limit: int) -> list[str]:
    """Take signs one kind at a time, in _KINDS order, until the limit.

    Args:
        by_kind (dict[str, list[str]]): The signs of each kind, in diff order.
        limit (int): Most signs taken.

    Returns:
        list[str]: The signs taken, grouped by kind in _KINDS order.
    """
    taken: dict[str, int] = dict.fromkeys(by_kind, 0)
    total = 0
    while total < limit and any(taken[kind] < len(by_kind[kind]) for kind in by_kind):
        for kind in _KINDS:
            if total < limit and taken[kind] < len(by_kind[kind]):
                taken[kind] += 1
                total += 1
    return [sign for kind in _KINDS for sign in by_kind[kind][: taken[kind]]]
