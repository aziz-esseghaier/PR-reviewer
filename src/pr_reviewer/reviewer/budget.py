"""Fitting a PR's review input into a size limit, by leaving out whole file diffs.

A free-tier model takes only so many tokens a minute (Groq: 8,000, the reply included), so a
big PR can't be sent whole. fit_pr_details() keeps the PR's text exactly as SWR-Bench writes
it (benchmark/review_input.py) when it fits. When it doesn't:
- if only a long description is in the way, every diff is kept and the description is
  shortened to the room left, with a note where it was cut;
- otherwise the description gets at most a fifth of the room, long commit messages are
  shortened, and file diffs are kept whole or left out, never cut in the middle, so the model
  never reviews half a change. Python source files go first, then Python test files, then
  other files (docs, configs), each group in the PR's own order. A diff too big for the room
  left is skipped and smaller ones after it still get in. The left-out files are listed by
  name at the end, so the model knows they changed.

Sizes are counted in characters. The caller turns its token budget into characters at the
LLM client's own conservative rate (core/limits.py), so a fitted request never goes over.

Imported by other modules; the file isn't run directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from pr_reviewer.benchmark.dataset import Commit, FilePatch, Instance
from pr_reviewer.benchmark.review_input import render_pr_details

# The list of left-out files starts with this line.
LEFT_OUT_NOTE = "**Left out to fit the size limit** (these changed files' diffs aren't shown):"

# Where a shortened description or commit message was cut.
CUT_NOTE = " [... cut to fit the size limit]"

# Longest commit message kept when a PR doesn't fit, in characters.
MESSAGE_CHARS = 500

# Share of the budget the description may take when a PR doesn't fit.
DESCRIPTION_SHARE = 0.2

# Folder names that hold tests.
TEST_FOLDERS = frozenset({"test", "tests", "testing"})


@dataclass(frozen=True)
class FittedInput:
    """A PR's review input, fitted into a size limit.

    Attributes:
        text (str): The text to send: SWR-Bench's PR details, trimmed if needed.
        left_out (tuple[str, ...]): The file diffs left out, such as "pkg/io.py (commit
            1a2b3c4)", in the PR's order. Empty when every diff is in.
        cut (bool): Whether the description or a commit message was shortened.
        full_chars (int): Length of the untrimmed text.
    """

    text: str
    left_out: tuple[str, ...]
    cut: bool
    full_chars: int

    @property
    def trimmed(self) -> bool:
        """Whether anything was left out or shortened.

        Returns:
            bool: True when the text isn't the full PR.
        """
        return bool(self.left_out) or self.cut


def is_test_file(path: str) -> bool:
    """Tell whether a file holds tests, from its path.

    Args:
        path (str): The file's path in the repo, such as "pkg/tests/test_io.py".

    Returns:
        bool: True for files in a tests folder, and for test_*.py, *_test.py and conftest.py.
    """
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""
    return (
        any(part in TEST_FOLDERS for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _rank(path: str) -> int:
    """Rank a changed file by how much its diff matters for finding bugs.

    Args:
        path (str): The file's path.

    Returns:
        int: 0 for Python source, 1 for Python tests, 2 for anything else.
    """
    if not path.endswith(".py"):
        return 2
    return 1 if is_test_file(path) else 0


def _shorten(text: str, limit: int) -> str:
    """Cut a text to a length, marking the cut.

    Args:
        text (str): The text.
        limit (int): Most characters to keep, the note included.

    Returns:
        str: The text itself when it fits, or its start followed by CUT_NOTE.
    """
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(CUT_NOTE))] + CUT_NOTE


def _block(patch: FilePatch) -> str:
    """Write one file's diff the way SWR-Bench does.

    Args:
        patch (FilePatch): The file and its diff.

    Returns:
        str: The file's path, then its diff in a code fence.
    """
    return f"{patch.file}\n```\n{patch.patch}\n```\n"


def _render(
    pr: Instance,
    description: str,
    commits: Sequence[tuple[Commit, str]],
    kept: set[tuple[int, int]],
    left_out: Sequence[str],
) -> str:
    """Write the PR details with only some file diffs, in SWR-Bench's layout.

    Args:
        pr (Instance): The PR.
        description (str): Its description, maybe shortened.
        commits (Sequence[tuple[Commit, str]]): Each reviewed commit with its message, maybe
            shortened.
        kept (set[tuple[int, int]]): The diffs to show, as (commit index, file index).
        left_out (Sequence[str]): Names of the diffs not shown, listed at the end.

    Returns:
        str: The text.
    """
    changes = ""
    for number, (commit, message) in enumerate(commits):
        block = f"Commit: {commit.sha}\nCommit Message: {message}\nCommit Code Changes: \n"
        for index, patch in enumerate(commit.diff):
            if (number, index) in kept:
                block += _block(patch)
        changes += f"{block}\n"
    text = (
        "**Pull Request Details**: \n"
        f"- **Title**: {pr.pr_title} \n"
        f"- **Description**: {description} \n"
        f"- **Code Changes**: {changes.strip()} \n"
    )
    if left_out:
        text += "\n" + LEFT_OUT_NOTE + "\n" + "".join(f"- {name}\n" for name in left_out)
    return text


def fit_pr_details(pr: Instance, max_chars: int) -> FittedInput:
    """Write a PR's review input in at most max_chars characters.

    Args:
        pr (Instance): The PR.
        max_chars (int): The most characters the text may have.

    Returns:
        FittedInput: The text, with what was left out or shortened. When the PR fits, the text
        is exactly SWR-Bench's (render_pr_details()).
    """
    full = render_pr_details(pr)
    if len(full) <= max_chars:
        return FittedInput(full, (), False, len(full))
    diffs = [
        (number, index, patch)
        for number, commit in enumerate(pr.pr_commits)
        for index, patch in enumerate(commit.diff)
    ]
    # A long description with diffs that fit: keep every diff, and give the description the
    # room that's left.
    every = {(number, index) for number, index, _ in diffs}
    whole = [(commit, commit.message) for commit in pr.pr_commits]
    description_room = max_chars - len(_render(pr, "", whole, every, []))
    if description_room >= max_chars * DESCRIPTION_SHARE:
        description = _shorten(pr.pr_statement, description_room)
        return FittedInput(_render(pr, description, whole, every, []), (), True, len(full))
    description = _shorten(pr.pr_statement, int(max_chars * DESCRIPTION_SHARE))
    commits = [(commit, _shorten(commit.message, MESSAGE_CHARS)) for commit in pr.pr_commits]
    cut = description != pr.pr_statement or any(
        message != commit.message for commit, message in commits
    )

    def name(number: int, patch: FilePatch) -> str:
        return f"{patch.file} (commit {pr.pr_commits[number].sha[:7]})"

    # Keep room for the list of left-out files as if every diff were left out, so the list
    # always fits.
    list_room = len(f"\n{LEFT_OUT_NOTE}\n") + sum(len(f"- {name(n, p)}\n") for n, _, p in diffs)
    room = max_chars - len(_render(pr, description, commits, set(), [])) - list_room
    kept: set[tuple[int, int]] = set()
    for number, index, patch in sorted(diffs, key=lambda item: (_rank(item[2].file), item[:2])):
        size = len(_block(patch))
        if size <= room:
            kept.add((number, index))
            room -= size
    left_out = [name(n, p) for n, i, p in diffs if (n, i) not in kept]
    text = _render(pr, description, commits, kept, left_out)
    # Only a PR whose description and commit messages alone are too big gets here.
    if len(text) > max_chars:
        text, cut = _shorten(text, max_chars), True
    return FittedInput(text, tuple(left_out), cut, len(full))
