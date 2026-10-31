"""Commit check: can every dev PR be checked out at its base and PR state, with the network off?

Each repo's PRs are checked in one sandbox whose network is off; connection attempts to
github.com and to the proxy must fail first, or the check stops. For each PR:
1. every reviewed commit must be in the shared copy;
2. its base is found: the commit just before its first reviewed commit;
3. the dataset's base_commit field is compared with the base. It can be the base itself,
   later on the same line of history (other people's commits came in between), earlier, on
   another line, or missing;
4. merge commits among its reviewed commits are counted. A merge can bring in other people's
   changes;
5. the files that differ between base and PR state are compared with the files the PR's own
   commits changed. Extra files mean base vs PR state holds more than the PR's own changes;
6. both sides are checked out, and a command runs at each: it counts the Python files.

A PR is usable when all its reviewed commits are there and both checkouts worked. The
results go to results/commit-check/ and to the pr_checkouts table, which later milestones read
to know which commits to install.

Imported by other modules; the file isn't run directly.
"""

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import psycopg
from docker import DockerClient
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.source import DATASET, SOURCE_COMMIT, SOURCE_REPO
from pr_reviewer.benchmark.split import SPLIT_SHA256
from pr_reviewer.sandbox.checkout import GIT_TIMEOUT, base_of, checkout, working_copy
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.repos import RepoError, git, missing_commits
from pr_reviewer.sandbox.runner import Sandbox, SandboxError, can_reach

# Where the dataset's base_commit field is, seen from the base.
Relation = Literal["same", "later", "earlier", "elsewhere", "missing"]

RELATION_TEXT: dict[Relation, str] = {
    "same": "the base itself",
    "later": "later on the same line (other commits came in between)",
    "earlier": "earlier on the same line",
    "elsewhere": "on another line of history",
    "missing": "not in the repo",
}

# Counts the Python files in the current folder, leaving out git's own folder.
_COUNT_PY = (
    "import pathlib; "
    "print(sum(1 for p in pathlib.Path('.').rglob('*.py') if '.git' not in p.parts))"
)

# Called after each PR with its check.
CheckProgress = Callable[["CommitCheck"], None]


@dataclass(frozen=True)
class CommitCheck:
    """What the commit check found for one PR.

    Attributes:
        instance_id (str): The PR.
        repo (str): Its GitHub repo.
        pr_sha (str): Its last reviewed commit: the state the reviewer sees.
        base_sha (str | None): The commit just before its first reviewed commit. None when it
            couldn't be found.
        base_field (str): The dataset's base_commit field.
        base_field_relation (Relation | None): Where that field is, seen from the base. None
            when the base wasn't found.
        commits_apart (int | None): How many commits the field and the base are apart, when
            one comes after the other on the same line.
        missing_commits (tuple[str, ...]): Reviewed commits not in the shared copy.
        merge_commits (int): Merge commits among the reviewed commits.
        extra_files (tuple[str, ...] | None): Files that differ between base and PR state but
            that no reviewed commit changed. None when not measured.
        python_files (dict[str, int]): Python files found at each side checked out.
        problem (str | None): Why the PR isn't usable. None when it is.
    """

    instance_id: str
    repo: str
    pr_sha: str
    base_sha: str | None = None
    base_field: str = ""
    base_field_relation: Relation | None = None
    commits_apart: int | None = None
    missing_commits: tuple[str, ...] = ()
    merge_commits: int = 0
    extra_files: tuple[str, ...] | None = None
    python_files: dict[str, int] = field(default_factory=dict)
    problem: str | None = None

    @property
    def usable(self) -> bool:
        """Whether both sides of the PR can be checked out."""
        return self.problem is None

    @property
    def only_pr_changes(self) -> bool | None:
        """Whether base vs PR state changes only files the PR's own commits changed."""
        return None if self.extra_files is None else not self.extra_files


def is_ancestor(sandbox: Sandbox, path: str, older: str, newer: str) -> bool:
    """Tell whether one commit comes before another on the same line of history.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy.
        older (str): The commit that may come first.
        newer (str): The commit that may come after it.

    Returns:
        bool: True when `older` is an ancestor of `newer`.

    Raises:
        RepoError: If git can't answer, for example because a commit is missing.
    """
    result = sandbox.run(
        ["git", "-C", path, "merge-base", "--is-ancestor", older, newer],
        timeout=GIT_TIMEOUT,
        label="ancestor",
    )
    if result.exit_code not in (0, 1) or result.timed_out:
        raise RepoError(f"Can't compare {older[:10]} and {newer[:10]}: {result.describe()}")
    return result.exit_code == 0


def base_field_relation(
    sandbox: Sandbox, path: str, base: str, base_field: str
) -> tuple[Relation, int | None]:
    """Say where the dataset's base_commit field is, seen from the base.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy.
        base (str): The base: the commit just before the first reviewed commit.
        base_field (str): The dataset's base_commit field.

    Returns:
        tuple[Relation, int | None]: The relation, and how many commits apart they are when
        one comes after the other.
    """
    if base_field == base:
        return "same", 0
    if missing_commits(sandbox, path, [base_field], timeout=GIT_TIMEOUT):
        return "missing", None
    orders: tuple[tuple[Relation, str, str], ...] = (
        ("later", base, base_field),
        ("earlier", base_field, base),
    )
    for relation, older, newer in orders:
        if is_ancestor(sandbox, path, older, newer):
            count = git(
                sandbox,
                ["-C", path, "rev-list", "--count", f"{older}..{newer}"],
                timeout=GIT_TIMEOUT,
                label="count",
            )
            return relation, int(count)
    return "elsewhere", None


def count_merges(sandbox: Sandbox, path: str, shas: Sequence[str]) -> int:
    """Count the merge commits (commits with more than one parent) among some commits.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy.
        shas (Sequence[str]): The commits.

    Returns:
        int: How many of them are merges.
    """
    output = git(
        sandbox,
        ["-C", path, "rev-list", "--no-walk", "--min-parents=2", *shas],
        timeout=GIT_TIMEOUT,
        label="merges",
    )
    return len(output.split())


def extra_files(sandbox: Sandbox, path: str, base: str, pr: Instance) -> tuple[str, ...]:
    """List files that differ between base and PR state but that no reviewed commit changed.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy.
        base (str): The base commit.
        pr (Instance): The PR, with the files each reviewed commit changed.

    Returns:
        tuple[str, ...]: The extra files, sorted. Empty when base vs PR state is exactly the
        PR's own changes.
    """
    output = git(
        sandbox,
        ["-C", path, "diff", "--name-only", base, pr.head_commit],
        timeout=GIT_TIMEOUT,
        label="diff",
    )
    changed = set(output.splitlines())
    own = {patch.file for commit in pr.pr_commits for patch in commit.diff}
    return tuple(sorted(changed - own))


def check_pr(sandbox: Sandbox, pr: Instance) -> CommitCheck:
    """Check one PR: its commits, its base, and a checkout and command at each side.

    Args:
        sandbox (Sandbox): A sandbox with the network off and the repo volume attached.
        pr (Instance): The PR.

    Returns:
        CommitCheck: What was found. A git problem is recorded as the PR's problem.

    Raises:
        SandboxError: If the sandbox itself stops working.
    """
    found: dict[str, Any] = {"base_field": pr.base_commit}
    try:
        path = working_copy(sandbox, pr.repo)
        reviewed = [commit.sha for commit in pr.pr_commits]
        missing = missing_commits(sandbox, path, reviewed, timeout=GIT_TIMEOUT)
        if missing:
            return CommitCheck(
                pr.instance_id,
                pr.repo,
                pr.head_commit,
                missing_commits=tuple(missing),
                problem=f"{len(missing)} of {len(reviewed)} reviewed commits missing",
                **found,
            )
        base = found["base_sha"] = base_of(sandbox, path, pr)
        relation, apart = base_field_relation(sandbox, path, base, pr.base_commit)
        found.update(base_field_relation=relation, commits_apart=apart)
        found["merge_commits"] = count_merges(sandbox, path, reviewed)
        found["extra_files"] = extra_files(sandbox, path, base, pr)
        counts: dict[str, int] = {}
        found["python_files"] = counts
        for side in ("base", "pr"):
            path, _ = checkout(sandbox, pr, side)
            result = sandbox.run(
                ["python3", "-c", _COUNT_PY], timeout=GIT_TIMEOUT, label="count-py", workdir=path
            )
            if not result.ok:
                raise RepoError(f"The command at the {side} side failed: {result.describe()}")
            counts[side] = int(result.output.strip())
    except RepoError as exc:
        return CommitCheck(pr.instance_id, pr.repo, pr.head_commit, problem=str(exc), **found)
    return CommitCheck(pr.instance_id, pr.repo, pr.head_commit, **found)


def check_prs(
    client: DockerClient,
    config: SandboxConfig,
    image: str,
    prs: Sequence[Instance],
    *,
    log_dir: Path | None = None,
    on_result: CheckProgress | None = None,
) -> list[CommitCheck]:
    """Check many PRs, one sandbox per repo, all with the network off.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.
        image (str): The sandbox image's name and tag.
        prs (Sequence[Instance]): The PRs to check.
        log_dir (Path | None): Folder for the sandboxes' command logs. None keeps none.
        on_result (CheckProgress | None): Called after each PR with its check.

    Returns:
        list[CommitCheck]: One check per PR, in the order of `prs`.

    Raises:
        SandboxError: If a sandbox can reach the internet although its network is off, or
            stops working.
    """
    by_repo: dict[str, list[Instance]] = defaultdict(list)
    for pr in prs:
        by_repo[pr.repo].append(pr)
    checks: dict[str, CommitCheck] = {}
    for repo_prs in by_repo.values():
        with Sandbox(client, config, image, purpose="check", log_dir=log_dir) as sandbox:
            # Off means no way out at all: neither the internet directly nor the proxy.
            if can_reach(sandbox) or can_reach(sandbox, config.proxy.container, config.proxy.port):
                raise SandboxError("A sandbox reached the network although its network is off")
            for pr in repo_prs:
                check = check_pr(sandbox, pr)
                checks[pr.instance_id] = check
                if on_result is not None:
                    on_result(check)
    return [checks[pr.instance_id] for pr in prs]


def summary_markdown(checks: Sequence[CommitCheck], config: dict[str, Any]) -> str:
    """Write the commit check's results as readable tables.

    Args:
        checks (Sequence[CommitCheck]): One check per PR.
        config (dict[str, Any]): The run's config.json contents.

    Returns:
        str: The Markdown text of summary.md.
    """
    usable = [c for c in checks if c.usable]
    repos = sorted({c.repo for c in checks})
    relations = Counter(c.base_field_relation for c in checks if c.base_field_relation)
    later = [c.commits_apart for c in checks if c.base_field_relation == "later"]
    merges = [c for c in checks if c.merge_commits]
    extra = [c for c in checks if c.extra_files]
    lines = [
        "# Commit check: dev set",
        "",
        f"- {len(checks)} PRs in {len(repos)} repos, checked {config['date']} in sandboxes "
        f"from {config['image']}",
        "- Network off in every sandbox: connections to github.com and to the proxy failed "
        "before each repo's checks, as they should",
        "- Base: the commit just before each PR's first reviewed commit (found with git). PR "
        "state: its last reviewed commit",
        "",
        "## Usable PRs",
        "",
        "A PR is usable when all its reviewed commits are in the shared copy and both sides "
        "were checked out and ran a command.",
        "",
        "| | PRs |",
        "|---|---|",
        f"| Usable | {len(usable)} |",
        f"| Not usable | {len(checks) - len(usable)} |",
        "",
        "## The dataset's base_commit field, seen from the base",
        "",
        "| base_commit field is ... | PRs |",
        "|---|---|",
    ]
    for relation, text in RELATION_TEXT.items():
        lines.append(f"| {text} | {relations.get(relation, 0)} |")
    if later:
        lines += [
            "",
            f"When it's later, it's {statistics.median(n for n in later if n is not None):g} "
            f"commits after the base in the middle case, and up to "
            f"{max(n for n in later if n is not None)}.",
        ]
    lines += [
        "",
        "## Merges and extra changes",
        "",
        f"- PRs with a merge among their reviewed commits: {len(merges)}",
        f"- PRs where base vs PR state also changes files no reviewed commit changed: {len(extra)}",
        "",
        "## By repo",
        "",
        "| Repo | PRs | Usable | base_commit field = base | With a merge | With extra files |",
        "|---|---|---|---|---|---|",
    ]
    for repo in repos:
        rows = [c for c in checks if c.repo == repo]
        same = sum(c.base_field_relation == "same" for c in rows)
        merged = sum(bool(c.merge_commits) for c in rows)
        extra_rows = sum(bool(c.extra_files) for c in rows)
        lines.append(
            f"| {repo} | {len(rows)} | {sum(c.usable for c in rows)} | {same} | {merged} | "
            f"{extra_rows} |"
        )
    flagged = [c for c in checks if not c.usable or c.merge_commits or c.extra_files]
    if flagged:
        lines += ["", "## PRs to look at", ""]
        for c in flagged:
            notes = []
            if c.problem:
                notes.append(f"not usable: {c.problem}")
            if c.merge_commits:
                notes.append(f"{c.merge_commits} merge commit(s)")
            if c.extra_files:
                notes.append(f"{len(c.extra_files)} extra file(s), such as {c.extra_files[0]}")
            lines.append(f"- {c.instance_id}: {'; '.join(notes)}")
    return "\n".join(lines) + "\n"


def check_config(date: str, image: str, config: SandboxConfig, prs: int) -> dict[str, Any]:
    """Describe a commit-check run, for config.json.

    Args:
        date (str): When it ran, as YYYY-MM-DD.
        image (str): The sandbox image's name and tag.
        config (SandboxConfig): The sandbox settings.
        prs (int): How many PRs were checked.

    Returns:
        dict[str, Any]: The config.
    """
    return {
        "name": "commit-check",
        "date": date,
        "split": "dev",
        "prs": prs,
        "image": image,
        "limits": config.limits.model_dump(),
        "base": "the first parent of the first reviewed commit, as SWR-Bench's scripts use",
        "swr_bench": {
            "repo": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "dataset_sha256": DATASET.sha256,
            "split_sha256": SPLIT_SHA256,
        },
    }


def write_check_results(
    folder: Path, checks: Sequence[CommitCheck], config: dict[str, Any]
) -> None:
    """Write the commit check's folder: config.json, summary.md and raw/checks.jsonl.

    Args:
        folder (Path): The folder, such as results/commit-check. Created if missing; earlier
            files are replaced.
        checks (Sequence[CommitCheck]): One check per PR.
        config (dict[str, Any]): The run's config, from check_config().
    """
    raw = folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (folder / "summary.md").write_text(
        summary_markdown(checks, config), encoding="utf-8", newline="\n"
    )
    with (raw / "checks.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for check in checks:
            file.write(json.dumps(asdict(check)) + "\n")


def save_checks(conn: psycopg.Connection[TupleRow], checks: Sequence[CommitCheck]) -> None:
    """Write the checks to pr_checkouts, replacing earlier rows of the same PRs.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection. bench_prs must hold the PRs.
        checks (Sequence[CommitCheck]): One check per PR.
    """
    with conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO pr_checkouts (instance_id, usable, base_sha, pr_sha,
                                      base_field_relation, merge_commits, only_pr_changes,
                                      problem, details, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (instance_id) DO UPDATE SET
                usable = EXCLUDED.usable, base_sha = EXCLUDED.base_sha,
                pr_sha = EXCLUDED.pr_sha, base_field_relation = EXCLUDED.base_field_relation,
                merge_commits = EXCLUDED.merge_commits,
                only_pr_changes = EXCLUDED.only_pr_changes, problem = EXCLUDED.problem,
                details = EXCLUDED.details, checked_at = EXCLUDED.checked_at
            """,
            [
                (
                    c.instance_id,
                    c.usable,
                    c.base_sha,
                    c.pr_sha,
                    c.base_field_relation,
                    c.merge_commits,
                    c.only_pr_changes,
                    c.problem,
                    Jsonb(asdict(c)),
                )
                for c in checks
            ],
        )
