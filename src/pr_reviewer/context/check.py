"""The retrieval check: did the model see the code each known issue's fix changed?

Each known issue in SWR-Bench names the later commit that fixed it. The functions that commit
changed (outside test files) show where the problem was. For each issue, the check finds the
best place the model could have seen one of them:
- in_diff: a function the PR itself changed, shown in its diff;
- in_context: a function (or whole file) in the context the request held;
- left_out: found by retrieval, but left out for lack of room;
- not_found: retrieval never found it.
Issues it can't place: no_function (the fix changed only tests, lines outside functions, or
new files), no_fix (SWR-Bench records no fix commit), fix_unreadable (git couldn't read it).

It tells "retrieval missed it" (not_found, left_out) apart from "the model didn't use it"
(in_diff, in_context, but the review missed the issue). Fix commits are used only here, to
measure; they're never review input. No model is called.

Imported by other modules; the file isn't run directly.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.candidates import Candidates, FixFunctions
from pr_reviewer.context.pack import PackedContext
from pr_reviewer.reviewer.budget import FittedInput

Status = Literal[
    "in_diff", "in_context", "left_out", "not_found", "no_function", "no_fix", "fix_unreadable"
]

# The placed statuses, best first: an issue takes the best one among its fix's functions.
PLACED: tuple[Status, ...] = ("in_diff", "in_context", "left_out", "not_found")
STATUSES: tuple[Status, ...] = (*PLACED, "no_function", "no_fix", "fix_unreadable")

# What each status means, for the summary.
MEANING: dict[Status, str] = {
    "in_diff": "in the PR's diff (a function the PR changed)",
    "in_context": "in the context the request held",
    "left_out": "found by retrieval, but left out for lack of room",
    "not_found": "not found by retrieval",
    "no_function": "the fix changed no function outside tests (tests, module-level lines or "
    "new files only)",
    "no_fix": "SWR-Bench records no fix commit",
    "fix_unreadable": "git couldn't read the fix commit",
}


@dataclass(frozen=True)
class IssueCheck:
    """Where the model could have seen one known issue's code.

    Attributes:
        instance_id (str): The PR.
        number (int): The issue's position among the PR's known issues, from 1.
        category (str): Its SWR-Bench category, such as "F.2".
        fix (str): The fix commit's ID, or "".
        status (Status): The best place one of the fix's functions was seen.
        functions (tuple[str, ...]): The fix's functions outside tests, as "file:qualname".
    """

    instance_id: str
    number: int
    category: str
    fix: str
    status: Status
    functions: tuple[str, ...]

    @property
    def functional(self) -> bool:
        """Whether it's a functional (F) issue."""
        return self.category.startswith("F")


def files_in_diff(pr: Instance, fitted: FittedInput) -> set[str]:
    """List the files with at least one diff in the PR's fitted text.

    Args:
        pr (Instance): The PR.
        fitted (FittedInput): Its fitted text, with the diffs left out.

    Returns:
        set[str]: The files the model saw a diff of.
    """
    left_out = set(fitted.left_out)
    return {
        patch.file
        for commit in pr.pr_commits
        for patch in commit.diff
        if f"{patch.file} (commit {commit.sha[:7]})" not in left_out
    }


def check_pr(
    pr: Instance,
    candidates: Candidates,
    fitted: FittedInput,
    packed: PackedContext,
    fixes: dict[str, FixFunctions],
) -> list[IssueCheck]:
    """Check each known issue of a PR.

    Args:
        pr (Instance): The PR.
        candidates (Candidates): Its candidate context.
        fitted (FittedInput): Its fitted PR text.
        packed (PackedContext): The context its request held.
        fixes (dict[str, FixFunctions]): What each of its fix commits changed.

    Returns:
        list[IssueCheck]: One per known issue, in the PR's order.
    """
    shown_files = files_in_diff(pr, fitted)
    in_diff = {(c.file, c.qualname) for c in candidates.changed if c.file in shown_files}
    in_context = {(p.file, p.qualname) for p in packed.included if p.kind != "file"}
    whole_in_context = {p.file for p in packed.included if p.kind == "file"}
    found = {(p.file, p.qualname) for p in candidates.pieces if p.kind != "file"}
    whole_found = {p.file for p in candidates.pieces if p.kind == "file"}

    def place(key: tuple[str, str]) -> Status:
        if key in in_diff:
            return "in_diff"
        if key in in_context or key[0] in whole_in_context:
            return "in_context"
        if key in found or key[0] in whole_found:
            return "left_out"
        return "not_found"

    checks: list[IssueCheck] = []
    for number, issue in enumerate(pr.changes, start=1):
        category = issue.change_type.split()[0]
        fix = issue.change_resolve_info.commit_sha if issue.change_resolve_info else ""
        status: Status
        functions: tuple[str, ...] = ()
        if not fix:
            status = "no_fix"
        elif fix not in fixes or fixes[fix].error:
            status = "fix_unreadable"
        else:
            keys = [(d.file, d.qualname) for d in fixes[fix].defs if not d.test]
            functions = tuple(f"{file}:{qualname}" for file, qualname in keys)
            if not keys:
                status = "no_function"
            else:
                status = min((place(key) for key in keys), key=PLACED.index)
        checks.append(IssueCheck(pr.instance_id, number, category, fix, status, functions))
    return checks


def summary_markdown(checks: Sequence[IssueCheck], name: str, prs: int) -> str:
    """Describe the retrieval check, as Markdown.

    Args:
        checks (Sequence[IssueCheck]): Every issue's check.
        name (str): The review run it was checked for.
        prs (int): How many PRs with known issues were checked.

    Returns:
        str: The text of retrieval_check.md.
    """
    counts = Counter(check.status for check in checks)
    functional = Counter(check.status for check in checks if check.functional)
    evolvability = Counter(check.status for check in checks if not check.functional)
    lines = [
        f"# Retrieval check: {name}",
        "",
        f"{len(checks)} known issues in {prs} PRs. For each, the functions its fix commit "
        "changed (outside tests) show where the problem was; the check finds the best place "
        "the model could have seen one of them. No model is called; fix commits are used only "
        "to measure.",
        "",
        "| Where | All | F | E | Meaning |",
        "|---|---|---|---|---|",
    ]
    for status in STATUSES:
        lines.append(
            f"| {status} | {counts[status]} | {functional[status]} | {evolvability[status]} | "
            f"{MEANING[status]} |"
        )
    outside = [check for check in checks if check.status in PLACED[1:]]
    if outside:
        reached = sum(check.status == "in_context" for check in outside)
        lines += [
            "",
            f"Of the {len(outside)} issues whose fix changed code outside the PR's diff, "
            f"the context held {reached} ({reached / len(outside):.0%}); "
            f"{sum(c.status == 'left_out' for c in outside)} were found but didn't fit, and "
            f"{sum(c.status == 'not_found' for c in outside)} weren't found.",
        ]
    lines += ["", "## Issues", "", "| PR | # | Category | Where | Fix's functions |"]
    lines.append("|---|---|---|---|---|")
    for check in checks:
        shown = ", ".join(f"`{f}`" for f in check.functions[:3])
        if len(check.functions) > 3:
            shown += f" and {len(check.functions) - 3} more"
        lines.append(
            f"| {check.instance_id} | {check.number} | {check.category} | {check.status} | "
            f"{shown} |"
        )
    return "\n".join(lines) + "\n"
