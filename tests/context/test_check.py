"""Tests for the retrieval check (context/check.py).

Checks each place a known issue's fix can be found (in the diff, in the context, retrieved but
left out, not found), the cases that can't be placed (fix in tests only, no fix commit, fix
unreadable), that a trimmed PR's left-out diffs don't count as shown, and the summary counts.

How to run (from the project folder):
    pytest tests/context/test_check.py
"""

import json

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.candidates import ChangedFunction, FixDef, FixFunctions
from pr_reviewer.context.check import check_pr, files_in_diff, summary_markdown
from pr_reviewer.context.pack import pack_context
from pr_reviewer.reviewer.budget import FittedInput
from tests.benchmark.test_dataset import make_row
from tests.context.helpers import candidates, piece


def make_issue_pr(fixes: list[str | None]) -> Instance:
    """Make a 2022 PR with one known issue per fix commit given.

    Args:
        fixes (list[str | None]): Each issue's fix commit, or None for an issue without one.

    Returns:
        Instance: The PR, whose commit changes pkg/a.py.
    """
    row = make_row(1, year=2022, issues=tuple("F.2 Logic" for _ in fixes))
    row["pr_commits"][0]["diff"] = [{"file": "pkg/a.py", "patch": "@@ -1 +1 @@\n-a\n+b"}]
    for issue, fix in zip(row["changes"], fixes, strict=True):
        if fix is None:
            issue["change_resolve_info"] = None
        else:
            issue["change_resolve_info"] = {"commit_sha": fix, "resolution_explanation": "x"}
    return Instance.model_validate_json(json.dumps(row))


def fix_of(*defs: tuple[str, str, bool], error: str = "") -> FixFunctions:
    """Describe what a fix commit changed.

    Args:
        *defs (tuple[str, str, bool]): Each changed function's file, qualname and test flag.
        error (str): Why git couldn't read it, if it couldn't.

    Returns:
        FixFunctions: The description.
    """
    return FixFunctions(
        defs=tuple(FixDef(file=f, qualname=q, test=t) for f, q, t in defs),
        outside_lines=0,
        new_files=(),
        error=error,
    )


def test_each_issue_gets_the_best_place_its_fix_was_seen() -> None:
    """Check in_diff, in_context, left_out, not_found and the unplaced cases."""
    shas = [str(n) * 40 for n in range(1, 8)]
    pr = make_issue_pr([*shas, None])
    changed = [ChangedFunction(file="pkg/a.py", qualname="f", start=1, end=9, fully_added=False)]
    shown = piece("caller", qualname="caller_fn", file="pkg/b.py")
    hidden = piece("test", 5000, qualname="test_x", file="tests/test_a.py")
    found = candidates(shown, hidden, changed=[c.model_dump() for c in changed])
    fitted = FittedInput("text", (), False, 4)
    packed = pack_context(found.pieces, 400)
    fixes = {
        shas[0]: fix_of(("pkg/a.py", "f", False)),
        shas[1]: fix_of(("pkg/b.py", "caller_fn", False), ("pkg/c.py", "z", False)),
        shas[2]: fix_of(("tests/test_a.py", "test_x", False)),
        shas[3]: fix_of(("pkg/d.py", "nowhere", False)),
        shas[4]: fix_of(("tests/test_a.py", "test_y", True)),
        shas[5]: fix_of(error="git failed"),
        # shas[6] wasn't read at all.
    }
    statuses = [check.status for check in check_pr(pr, found, fitted, packed, fixes)]
    assert statuses == [
        "in_diff",
        "in_context",
        "left_out",
        "not_found",
        "no_function",
        "fix_unreadable",
        "fix_unreadable",
        "no_fix",
    ]


def test_a_left_out_diff_isnt_shown() -> None:
    """Check a trimmed PR's left-out file diffs don't count as seen."""
    pr = make_issue_pr(["1" * 40])
    sha = pr.pr_commits[0].sha[:7]
    trimmed = FittedInput("text", (f"pkg/a.py (commit {sha})",), False, 99)
    assert files_in_diff(pr, trimmed) == set()
    assert files_in_diff(pr, FittedInput("text", (), False, 4)) == {"pkg/a.py"}


def test_the_summary_counts_by_place_and_kind() -> None:
    """Check the summary's counts and its line about issues outside the diff."""
    pr = make_issue_pr(["1" * 40, "2" * 40])
    found = candidates(piece("caller", qualname="g"))
    packed = pack_context(found.pieces, 0)
    fixes = {"1" * 40: fix_of(("pkg/a.py", "g", False)), "2" * 40: fix_of(("x.py", "h", False))}
    checks = check_pr(pr, found, FittedInput("t", (), False, 1), packed, fixes)
    text = summary_markdown(checks, "run", prs=1)
    assert "| left_out | 1 | 1 | 0 |" in text
    assert "| not_found | 1 | 1 | 0 |" in text
    assert (
        "Of the 2 issues whose fix changed code outside the PR's diff, the context held 0" in text
    )
