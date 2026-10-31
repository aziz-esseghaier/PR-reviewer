"""Shared repo copies: one git copy of each SWR-Bench repo, kept in a Docker volume.

Each repo is cloned once from GitHub as a bare repository (the history only, no files checked
out) into the volume named in configs/sandbox.yaml, at /repos/<owner>__<name>.git. The PRs'
commits come from GitHub's pull/<N>/head refs, as SWR-Bench's own scripts get them. A reviewed
commit that isn't there, for example because the author rewrote the branch later, is then
fetched by its ID. All this git work happens in a container, never on the Windows drive.

Only the sandbox that downloads keeps the volume writable, and it runs nothing but git. Every
other sandbox attaches it read-only and makes its own working copy from it
(sandbox/checkout.py), so PR code can never change it.

Imported by other modules; the file isn't run directly.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.sandbox.runner import REPOS_MOUNT, RunResult, Sandbox, SandboxError

# How many PR refs one fetch asks for.
REFS_PER_FETCH = 100

# Where commits fetched by their ID are kept, so git never cleans them away.
FETCHED_REFS = "refs/swr-bench/commits"


class RepoError(SandboxError):
    """Raised when a git command on a shared copy fails."""


def repo_dir(repo: str) -> str:
    """Give the folder of a repo's shared copy inside a sandbox.

    Args:
        repo (str): The GitHub repo, such as "psf/requests".

    Returns:
        str: Such as "/repos/psf__requests.git".
    """
    return f"{REPOS_MOUNT}/{repo.replace('/', '__')}.git"


def github_url(repo: str) -> str:
    """Give the address a repo is cloned from.

    Args:
        repo (str): The GitHub repo, such as "psf/requests".

    Returns:
        str: Such as "https://github.com/psf/requests.git".
    """
    return f"https://github.com/{repo}.git"


def pr_number(pr: Instance) -> int:
    """Read a PR's number from its instance ID.

    Args:
        pr (Instance): The PR, such as psf__requests-2317.

    Returns:
        int: Its number on GitHub, such as 2317.
    """
    return int(pr.instance_id.rsplit("-", 1)[1])


def git(sandbox: Sandbox, args: Sequence[str], *, timeout: float, label: str = "git") -> str:
    """Run git in a sandbox and return what it printed.

    Args:
        sandbox (Sandbox): The sandbox to run it in.
        args (Sequence[str]): git's arguments, such as ["-C", "/repos/x.git", "fetch", ...].
        timeout (float): Seconds it may run.
        label (str): A short word for the log file's name.

    Returns:
        str: Its output, with surrounding blank space removed.

    Raises:
        RepoError: If git fails, runs out of time or memory.
    """
    result = sandbox.run(["git", *args], timeout=timeout, label=label)
    if not result.ok:
        raise RepoError(f"git {' '.join(args[:4])} ... failed: {_tail(result)}")
    return result.output.strip()


def missing_commits(
    sandbox: Sandbox, git_dir: str, shas: Iterable[str], *, timeout: float
) -> list[str]:
    """List the commits a git copy doesn't have.

    Args:
        sandbox (Sandbox): The sandbox to look in.
        git_dir (str): The git copy, such as a shared copy or a working copy.
        shas (Iterable[str]): Commit IDs to look for.
        timeout (float): Seconds the lookup may take.

    Returns:
        list[str]: The IDs not found, in the order given.

    Raises:
        RepoError: If the lookup itself fails.
    """
    wanted = list(dict.fromkeys(shas))
    if not wanted:
        return []
    # One shell loop instead of one command per commit. The IDs are checked hex strings, and
    # they're passed as arguments, never pasted into the script.
    script = 'for c do git -C "$0" cat-file -e "$c^{commit}" 2>/dev/null || echo "$c"; done'
    result = sandbox.run(["sh", "-c", script, git_dir, *wanted], timeout=timeout, label="lookup")
    if not result.ok:
        raise RepoError(f"Looking up commits in {git_dir} failed: {_tail(result)}")
    found_missing = set(result.output.split())
    return [sha for sha in wanted if sha in found_missing]


@dataclass(frozen=True)
class RepoUpdate:
    """What updating one repo's shared copy did.

    Attributes:
        repo (str): The GitHub repo.
        cloned (bool): Whether the copy was downloaded now, rather than already there.
        refs_fetched (int): PR refs fetched now.
        refs_missing (tuple[int, ...]): PRs whose ref GitHub didn't have.
        fetched_by_id (tuple[str, ...]): Commits that weren't in their PR's ref, fetched by ID.
        missing (tuple[str, ...]): Reviewed commits still missing: their PRs can't be checked
            out.
        base_fields_missing (tuple[str, ...]): Commits named in the dataset's base_commit
            field that are missing. Only needed for the comparison in the commit check.
    """

    repo: str
    cloned: bool
    refs_fetched: int
    refs_missing: tuple[int, ...]
    fetched_by_id: tuple[str, ...]
    missing: tuple[str, ...]
    base_fields_missing: tuple[str, ...]


def update_repo(
    sandbox: Sandbox,
    repo: str,
    prs: Sequence[Instance],
    *,
    timeout: float,
    source_url: Callable[[str], str] = github_url,
) -> RepoUpdate:
    """Make sure a repo's shared copy has every commit some PRs need.

    Clones the repo if its copy isn't there yet, fetches each PR's pull/<N>/head ref it lacks,
    then fetches by ID any reviewed commit (or base_commit field) still missing. Running it
    again only fetches what's new. Needs a sandbox with the network on and the repo volume
    writable.

    Args:
        sandbox (Sandbox): The downloading sandbox.
        repo (str): The GitHub repo, such as "psf/requests".
        prs (Sequence[Instance]): The repo's PRs whose commits are needed.
        timeout (float): Seconds each download may take.
        source_url (Callable[[str], str]): Gives the address to clone from. Tests use a local
            repo instead of GitHub.

    Returns:
        RepoUpdate: What was downloaded, and what is still missing.

    Raises:
        RepoError: If cloning fails, or another git command fails.
    """
    path = repo_dir(repo)
    cloned = False
    if not sandbox.run(["test", "-d", path], timeout=60, label="exists").ok:
        partial = f"{path}.partial"
        sandbox.run(["rm", "-rf", partial], timeout=600, label="cleanup")
        git(sandbox, ["clone", "--bare", "--quiet", source_url(repo), partial], timeout=timeout)
        # The copy only ever grows; git must never prune commits fetched by their ID.
        git(sandbox, ["-C", partial, "config", "gc.auto", "0"], timeout=60)
        if not sandbox.run(["mv", partial, path], timeout=600, label="move").ok:
            raise RepoError(f"Can't move {partial} into place")
        cloned = True

    have = set(
        git(
            sandbox,
            ["-C", path, "for-each-ref", "--format=%(refname)", "refs/pull/"],
            timeout=600,
        ).split()
    )
    numbers = sorted({pr_number(pr) for pr in prs} - _ref_numbers(have))
    refs_missing: list[int] = []
    for start in range(0, len(numbers), REFS_PER_FETCH):
        batch = numbers[start : start + REFS_PER_FETCH]
        if not _fetch_refs(sandbox, path, batch, timeout):
            # One unknown ref fails the whole fetch, so ask for each one alone.
            refs_missing += [n for n in batch if not _fetch_refs(sandbox, path, [n], timeout)]

    reviewed = [commit.sha for pr in prs for commit in pr.pr_commits]
    fields = [pr.base_commit for pr in prs]
    fetched: list[str] = []
    for sha in missing_commits(sandbox, path, [*reviewed, *fields], timeout=600):
        refspec = f"{sha}:{FETCHED_REFS}/{sha}"
        result = sandbox.run(
            ["git", "-C", path, "fetch", "--quiet", "origin", refspec],
            timeout=timeout,
            label="fetch-id",
        )
        if result.ok:
            fetched.append(sha)
    still_missing = set(missing_commits(sandbox, path, [*reviewed, *fields], timeout=600))
    return RepoUpdate(
        repo=repo,
        cloned=cloned,
        refs_fetched=len(numbers) - len(refs_missing),
        refs_missing=tuple(refs_missing),
        fetched_by_id=tuple(fetched),
        missing=tuple(sha for sha in dict.fromkeys(reviewed) if sha in still_missing),
        base_fields_missing=tuple(sha for sha in dict.fromkeys(fields) if sha in still_missing),
    )


def copy_sizes(sandbox: Sandbox) -> dict[str, int]:
    """Measure each shared copy in the repo volume.

    Args:
        sandbox (Sandbox): Any sandbox; the volume is attached to all of them.

    Returns:
        dict[str, int]: Size in MB by GitHub repo, such as {"psf/requests": 12}.
    """
    script = f'for d in {REPOS_MOUNT}/*.git; do [ -d "$d" ] && du -sm "$d"; done; true'
    result = sandbox.run(["sh", "-c", script], timeout=600, label="sizes")
    sizes: dict[str, int] = {}
    for line in result.output.splitlines():
        size, _, folder = line.partition("\t")
        name = folder.rsplit("/", 1)[-1].removesuffix(".git")
        if size.isdigit() and "__" in name:
            sizes[name.replace("__", "/", 1)] = int(size)
    return sizes


def _ref_numbers(refs: Iterable[str]) -> set[int]:
    """Read the PR numbers out of refs/pull/<N>/head ref names.

    Args:
        refs (Iterable[str]): Ref names.

    Returns:
        set[int]: The PR numbers whose head ref is among them.
    """
    numbers = set()
    for ref in refs:
        parts = ref.split("/")
        is_head = len(parts) == 4 and parts[:2] == ["refs", "pull"] and parts[3] == "head"
        if is_head and parts[2].isdigit():
            numbers.add(int(parts[2]))
    return numbers


def _fetch_refs(sandbox: Sandbox, path: str, numbers: Sequence[int], timeout: float) -> bool:
    """Fetch some PRs' pull/<N>/head refs into a shared copy.

    Args:
        sandbox (Sandbox): The downloading sandbox.
        path (str): The shared copy.
        numbers (Sequence[int]): The PR numbers.
        timeout (float): Seconds the fetch may take.

    Returns:
        bool: True when every ref was fetched.
    """
    refspecs = [f"+refs/pull/{n}/head:refs/pull/{n}/head" for n in numbers]
    result = sandbox.run(
        ["git", "-C", path, "fetch", "--quiet", "origin", *refspecs],
        timeout=timeout,
        label="fetch-prs",
    )
    return result.ok


def _tail(result: RunResult, lines: int = 5) -> str:
    """Give the last lines of a command's output, for an error message.

    Args:
        result (RunResult): The command's result.
        lines (int): How many lines to keep.

    Returns:
        str: How it ended, then its last output lines.
    """
    last = " | ".join(result.output.strip().splitlines()[-lines:])
    return f"{result.describe()}: {last}" if last else result.describe()
