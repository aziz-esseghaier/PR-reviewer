"""Checkouts: put a sandbox's own working copy of a repo at a PR's base or PR state.

A PR has two sides:
- base: the commit just before the PR's first reviewed commit (its first parent), as
  SWR-Bench's own scripts use. Only git knows it: the dataset lists the PR's commits but not
  their parents. The dataset's base_commit field can be a different, later commit.
- pr: the PR's last reviewed commit, the state the reviewer sees. Never a later fix.

The working copy lives in the sandbox at /work/<owner>__<name>. It borrows the history from
the read-only shared copy (`git clone --shared`, see sandbox/repos.py), so making it takes
seconds and downloads nothing, and checkouts work with the network off. After each checkout,
git is asked which commit it's at, so a wrong state is caught instead of tested.

Imported by other modules; the file isn't run directly.
"""

from typing import Literal

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.sandbox.repos import RepoError, git, repo_dir
from pr_reviewer.sandbox.runner import Sandbox

# Which side of a PR to check out.
Side = Literal["base", "pr"]

# Where a sandbox keeps its working copies.
WORK_DIR = "/work"

# Seconds one git command on a working copy may take. Checking out a big repo takes seconds.
GIT_TIMEOUT = 600


class CheckoutError(RepoError):
    """Raised when a PR's commit can't be found or checked out."""


def working_copy(sandbox: Sandbox, repo: str) -> str:
    """Make the sandbox's working copy of a repo, unless it already has one.

    Args:
        sandbox (Sandbox): The sandbox.
        repo (str): The GitHub repo, such as "psf/requests".

    Returns:
        str: The working copy's folder inside the sandbox, such as "/work/psf__requests".

    Raises:
        RepoError: If the shared copy is missing (run `pr-reviewer sandbox repos` first) or
            the clone fails.
    """
    path = f"{WORK_DIR}/{repo.replace('/', '__')}"
    if sandbox.run(["test", "-d", f"{path}/.git"], timeout=60, label="exists").ok:
        return path
    shared = repo_dir(repo)
    if not sandbox.run(["test", "-d", shared], timeout=60, label="exists").ok:
        raise RepoError(
            f"No shared copy of {repo} in the repo volume. Download it with: pr-reviewer "
            "sandbox repos"
        )
    git(
        sandbox,
        ["clone", "--shared", "--no-checkout", "--quiet", shared, path],
        timeout=GIT_TIMEOUT,
        label="clone",
    )
    return path


def base_of(sandbox: Sandbox, path: str, pr: Instance) -> str:
    """Find a PR's base: the commit just before its first reviewed commit.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy (or shared copy) to ask.
        pr (Instance): The PR.

    Returns:
        str: The base's full commit ID.

    Raises:
        CheckoutError: If the first reviewed commit is missing, or has no parent.
    """
    first = pr.pr_commits[0].sha
    result = sandbox.run(
        ["git", "-C", path, "rev-parse", "--verify", "--quiet", f"{first}^1"],
        timeout=GIT_TIMEOUT,
        label="base",
    )
    if not result.ok:
        raise CheckoutError(f"{pr.instance_id}: can't find the commit before {first[:10]}")
    return result.output.strip()


def commit_at(sandbox: Sandbox, path: str, pr: Instance, side: Side) -> str:
    """Give the commit ID of one side of a PR.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy.
        pr (Instance): The PR.
        side (Side): "base" or "pr".

    Returns:
        str: The full commit ID.

    Raises:
        CheckoutError: If the base can't be found.
    """
    return base_of(sandbox, path, pr) if side == "base" else pr.head_commit


def checkout(sandbox: Sandbox, pr: Instance, side: Side) -> tuple[str, str]:
    """Put the sandbox's working copy of the PR's repo at one side of the PR.

    Files of the other side that git tracks are replaced; files git doesn't track (such as
    build output) are left alone.

    Args:
        sandbox (Sandbox): The sandbox.
        pr (Instance): The PR.
        side (Side): "base" or "pr".

    Returns:
        tuple[str, str]: The working copy's folder, and the commit ID it's now at.

    Raises:
        RepoError: If there's no shared copy of the repo.
        CheckoutError: If the commit is missing, the checkout fails, or git ends up at a
            different commit.
    """
    path = working_copy(sandbox, pr.repo)
    sha = commit_at(sandbox, path, pr, side)
    return path, _switch(sandbox, pr, path, sha, side)


def checkout_commit(sandbox: Sandbox, pr: Instance, sha: str) -> tuple[str, str]:
    """Put the working copy at one of the PR's own commits, such as a later fix.

    Args:
        sandbox (Sandbox): The sandbox.
        pr (Instance): The PR.
        sha (str): The full ID of one of the PR's commits (its all_commits).

    Returns:
        tuple[str, str]: The working copy's folder, and the commit ID it's now at.

    Raises:
        CheckoutError: If the commit isn't one of the PR's, or the checkout fails.
    """
    if sha not in {commit.sha for commit in pr.all_commits}:
        raise CheckoutError(f"{pr.instance_id}: {sha[:10]} isn't one of the PR's commits")
    path = working_copy(sandbox, pr.repo)
    return path, _switch(sandbox, pr, path, sha, sha[:10])


def _switch(sandbox: Sandbox, pr: Instance, path: str, sha: str, name: str) -> str:
    """Check out a commit in the working copy and make sure git is there.

    Args:
        sandbox (Sandbox): The sandbox.
        pr (Instance): The PR, for messages.
        path (str): The working copy.
        sha (str): The full commit ID.
        name (str): What the commit is, for messages and log names, such as "base".

    Returns:
        str: The commit ID git is now at.

    Raises:
        CheckoutError: If the checkout fails, or git ends up at a different commit.
    """
    result = sandbox.run(
        [
            "git",
            "-C",
            path,
            "-c",
            "advice.detachedHead=false",
            "checkout",
            "--quiet",
            "--force",
            "--detach",
            sha,
        ],
        timeout=GIT_TIMEOUT,
        label=f"checkout-{name}",
    )
    if not result.ok:
        raise CheckoutError(
            f"{pr.instance_id}: checking out {name} ({sha[:10]}) failed: "
            f"{result.describe()}: {result.output.strip()[-300:]}"
        )
    head = git(sandbox, ["-C", path, "rev-parse", "HEAD"], timeout=GIT_TIMEOUT, label="head")
    if head != sha:
        raise CheckoutError(
            f"{pr.instance_id}: asked for {sha[:10]} but git is at {head[:10]} after checkout"
        )
    return sha
