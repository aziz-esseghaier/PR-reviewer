"""Signs that a PR changed some behavior on purpose, for the intent check on a proof.

A test proves that the PR changed what some code does. That's a bug only if the change wasn't
meant: a PR may change a default or start raising an error on bad input deliberately. The PR
usually leaves a trace when it does, and our code looks for it in the PR's own diff, without a
model call:
- lines its tests add that mention the changed function (such as an assert on the new value,
  or `pytest.raises` of the new error);
- lines its docs or changelog add that mention the function or the error;
- warnings (deprecations) added to the changed function.
Only a proof with signs goes to the verifier, which decides whether the change was deliberate.

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


def added_lines(pr: Instance) -> list[tuple[str, str]]:
    """List the lines a PR's reviewed commits add, with their file.

    Args:
        pr (Instance): The PR.

    Returns:
        list[tuple[str, str]]: (file, line) for every added line, in commit order, without
        the leading "+".
    """
    found = []
    for commit in pr.pr_commits:
        for patch in commit.diff:
            for line in patch.patch.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    found.append((patch.file, line[1:]))
    return found


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
        total([]) == 0", at most MAX_SIGNS.
    """
    wanted = [name for name in names if len(name) > 2]
    error = [exception.rsplit(".", 1)[-1]] if exception else []
    found: list[str] = []
    for file, line in added_lines(pr):
        text = line.strip()
        if not text:
            continue
        kind = ""
        if is_test_file(file) and _mentions(text, wanted):
            kind = "test"
        elif is_doc_file(file) and _mentions(text, [*wanted, *error]):
            kind = "docs"
        elif file == claim_file and _WARNS.search(text) and text in function_code:
            kind = "warning"
        if kind:
            sign = f"{file} ({kind}): + {text}"
            found.append(sign[:SIGN_CHARS])
        if len(found) >= MAX_SIGNS:
            break
    return found
