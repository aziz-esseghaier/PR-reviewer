"""Turn a live GitHub pull request into the Instance the benchmark's code works on.

Every stage (setup, context, static tools, reviewer, evidence agent) takes an Instance: a
SWR-Bench PR with its commits and their diffs. Building the same object from GitHub's answers
lets the Action run the measured pipeline unchanged. A live PR has no known issues, so its
`changes` are empty, like a clean benchmark PR.

The PR's base is the parent of its first commit, as for benchmark PRs (sandbox/checkout.py),
and its reviewed state is its last commit. A PR with merge commits is refused: merging the
target branch into the PR would make its diff include changes it didn't make.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from pr_reviewer.benchmark.dataset import Commit, CommitRef, FilePatch, Instance
from pr_reviewer.integrations.github import check_repo


class LivePrError(Exception):
    """Raised when a pull request can't be reviewed, such as one with merge commits."""


def live_instance_id(repo: str, number: int) -> str:
    """Name a live PR the way SWR-Bench names its PRs.

    Args:
        repo (str): The repo, such as "octo/demo".
        number (int): The pull request's number.

    Returns:
        str: Such as "octo__demo-7".
    """
    owner, name = check_repo(repo).split("/")
    return f"{owner}__{name}-{int(number)}"


def is_fork(pull: Mapping[str, Any]) -> bool:
    """Tell whether a pull request comes from another repo (a fork).

    Args:
        pull (Mapping[str, Any]): GitHub's pull request object.

    Returns:
        bool: True when its head branch isn't in the base repo, or its repo was deleted.
    """
    head_repo = (pull.get("head") or {}).get("repo") or {}
    base_repo = (pull.get("base") or {}).get("repo") or {}
    return not head_repo or head_repo.get("full_name") != base_repo.get("full_name")


def _date(text: str) -> dt.datetime:
    """Read one of GitHub's timestamps, such as "2026-11-22T10:00:00Z".

    Args:
        text (str): The timestamp.

    Returns:
        dt.datetime: The time, in UTC.
    """
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(dt.UTC)


def commit_of(detail: Mapping[str, Any]) -> Commit:
    """Turn GitHub's commit object into a reviewed commit with its diff.

    Files without a patch (binary files, or files too big for GitHub to show) are left out,
    as there's nothing to review in them.

    Args:
        detail (Mapping[str, Any]): GitHub's commit object, from the single-commit endpoint.

    Returns:
        Commit: Its ID, date, message and changed files.

    Raises:
        LivePrError: If it's a merge commit.
    """
    if len(detail.get("parents") or []) > 1:
        raise LivePrError(
            f"Commit {str(detail.get('sha'))[:10]} is a merge commit; rebase the pull request "
            "on its target branch instead of merging it in"
        )
    info = detail["commit"]
    files = [
        FilePatch(file=item["filename"], patch=item["patch"])
        for item in detail.get("files") or []
        if item.get("patch")
    ]
    return Commit(
        sha=detail["sha"],
        date=_date(info["committer"]["date"]),
        message=info.get("message") or "",
        diff=tuple(files),
    )


def build_instance(
    repo: str,
    pull: Mapping[str, Any],
    commits: Sequence[Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Any]],
) -> Instance:
    """Build the Instance of a live pull request from GitHub's answers.

    Args:
        repo (str): The repo, such as "octo/demo".
        pull (Mapping[str, Any]): GitHub's pull request object.
        commits (Sequence[Mapping[str, Any]]): Its commits, oldest first (pulls/N/commits).
        details (Mapping[str, Mapping[str, Any]]): Each commit's full object, by ID, with its
            files.

    Returns:
        Instance: The PR, reviewed at its last commit, with no known issues.

    Raises:
        LivePrError: If it has no commits, has a merge commit, or its commits aren't in date
            order (the reviewed state couldn't be told apart).
    """
    if not commits:
        raise LivePrError("The pull request has no commits")
    reviewed = [commit_of(details[item["sha"]]) for item in commits]
    parents = details[commits[0]["sha"]].get("parents") or []
    if not parents:
        raise LivePrError("The pull request's first commit has no parent, so it has no base")
    number = int(pull["number"])
    try:
        return Instance(
            instance_id=live_instance_id(repo, number),
            repo=repo,
            pr_title=pull.get("title") or "",
            pr_statement=pull.get("body") or "",
            change_introduced=False,
            base_commit=parents[0]["sha"],
            # Reviewed as of its last commit, so every commit counts as made by then.
            created_at=max(commit.date for commit in reviewed),
            changes=(),
            pr_commits=tuple(reviewed),
            all_commits=tuple(CommitRef(sha=c.sha, date=c.date) for c in reviewed),
        )
    except ValidationError as exc:
        error = exc.errors()[0]
        raise LivePrError(f"The pull request can't be reviewed: {error['msg']}") from None
