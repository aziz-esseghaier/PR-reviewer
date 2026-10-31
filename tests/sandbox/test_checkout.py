"""
Tests for the shared repo copies, checkouts and commit check (sandbox/repos.py,
sandbox/checkout.py and sandbox/commit_check.py).

A small "upstream" repo is built inside a container (never on the Windows drive), shaped like
the cases the real PRs have:
- main: commits A, B, C, D, one after the other.
- PR 1: P1 and P2, branched from B. Its base_commit field is C, later than its base B.
- PR 2: Q1, branched from C, then M, which merges D from main into the PR. Field C.
- PR 3: X, branched from D. No ref holds X any more, as after a force-push. Field D.
- PR 4: a commit that never existed.

Then its shared copy is made the way the real ones are (PR refs, then missing commits by ID),
and the PRs are checked out and checked. These tests start sandboxes, so they're skipped when
Docker isn't running or the image isn't built.

How to run (from the project folder):
    pytest tests/sandbox/test_checkout.py
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pytest
from docker import DockerClient

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.commit_check import check_prs
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.repos import RepoError, RepoUpdate, pr_number, repo_dir, update_repo
from pr_reviewer.sandbox.runner import Sandbox
from tests.benchmark.test_dataset import make_row

pytestmark = pytest.mark.docker

REPO = "owner/toy"
UPSTREAM = "/repos/_upstream/toy"
NEVER_EXISTED = "f" * 40

# Builds the upstream repo and prints each commit's ID as NAME=sha.
BUILD_UPSTREAM = f"""
set -e
rm -rf /repos/_upstream {repo_dir(REPO)}
mkdir -p {UPSTREAM} && cd {UPSTREAM}
git init -q -b main
git config user.email test@example.com && git config user.name Test
commit() {{ echo "$2" > "$1"; git add "$1"; git commit -q -m "$3"; git rev-parse HEAD; }}
A=$(commit a.py 'a = 1' A)
B=$(commit b.py 'b = 1' B)
git checkout -q -b pr1
P1=$(commit p.py 'p = 1' P1)
P2=$(commit p.py 'p = 2' P2)
git checkout -q main
C=$(commit c.py 'c = 1' C)
git checkout -q -b pr2
Q1=$(commit q.py 'q = 1' Q1)
git checkout -q main
D=$(commit d.py 'd = 1' D)
git checkout -q pr2
git merge -q --no-edit main
M=$(git rev-parse HEAD)
git checkout -q -b pr3 main
X=$(commit x.py 'x = 1' X)
git checkout -q main
git update-ref refs/pull/1/head $P2
git update-ref refs/pull/2/head $M
git update-ref refs/pull/3/head $D
git branch -q -D pr1 pr2 pr3
git config uploadpack.allowAnySHA1InWant true
echo A=$A B=$B C=$C D=$D P1=$P1 P2=$P2 Q1=$Q1 M=$M X=$X
"""


def toy_pr(number: int, commits: Sequence[tuple[str, Sequence[str]]], base_field: str) -> Instance:
    """Make a clean PR of the toy repo whose reviewed commits are real toy commits.

    Args:
        number (int): The PR number.
        commits (Sequence[tuple[str, Sequence[str]]]): Each reviewed commit's ID, and the
            files the dataset says it changed.
        base_field (str): The dataset's base_commit field.

    Returns:
        Instance: The PR.
    """
    row = make_row(number, issues=(), repo=REPO, commits=len(commits))
    # pr_commits and all_commits share these entries, so both get the real IDs.
    for entry, (sha, files) in zip(row["pr_commits"], commits, strict=True):
        entry["sha"] = sha
        entry["diff"] = [{"file": name, "patch": "@@"} for name in files]
    row["base_commit"] = base_field
    return Instance.model_validate_json(json.dumps(row))


@dataclass(frozen=True)
class Toy:
    """The toy repo's commits, PRs, and what making its shared copy did.

    Attributes:
        sha (dict[str, str]): Commit IDs by name, such as sha["B"].
        prs (dict[int, Instance]): The four PRs by number.
        update (RepoUpdate): The result of making the shared copy.
    """

    sha: dict[str, str]
    prs: dict[int, Instance]
    update: RepoUpdate


@pytest.fixture(scope="module")
def writer(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> Iterator[Sandbox]:
    """Start a sandbox that may write the test repo volume, as the downloading one does.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.

    Yields:
        Sandbox: The sandbox, network off.
    """
    with Sandbox(
        docker, test_config, sandbox_image, purpose="test-writer", repos_writable=True
    ) as box:
        yield box


@pytest.fixture(scope="module")
def toy(writer: Sandbox) -> Toy:
    """Build the upstream repo and make its shared copy from it.

    Args:
        writer (Sandbox): A sandbox that may write the repo volume.

    Returns:
        Toy: The commits, PRs and update result.
    """
    built = writer.run(["sh", "-c", BUILD_UPSTREAM], timeout=120)
    assert built.ok, built.output
    sha = dict(pair.split("=") for pair in built.output.split())
    prs = {
        1: toy_pr(1, [(sha["P1"], ["p.py"]), (sha["P2"], ["p.py"])], sha["C"]),
        2: toy_pr(2, [(sha["Q1"], ["q.py"]), (sha["M"], [])], sha["C"]),
        3: toy_pr(3, [(sha["X"], ["x.py"])], sha["D"]),
        4: toy_pr(4, [(NEVER_EXISTED, ["n.py"])], sha["B"]),
    }
    update = update_repo(
        writer, REPO, list(prs.values()), timeout=120, source_url=lambda _: f"file://{UPSTREAM}"
    )
    return Toy(sha, prs, update)


def test_pr_numbers_come_from_the_instance_id() -> None:
    """Check the PR number is read from the end of the instance ID."""
    assert pr_number(toy_pr(4170, [("a" * 40, [])], "b" * 40)) == 4170
    assert repo_dir("psf/requests") == "/repos/psf__requests.git"


def test_the_shared_copy_gets_pr_refs_and_lost_commits(toy: Toy) -> None:
    """Check the copy has the PR refs, fetches the commit no ref holds, and names the rest."""
    update = toy.update
    assert update.cloned
    assert update.refs_fetched == 3
    assert update.refs_missing == (4,)
    assert update.fetched_by_id == (toy.sha["X"],)
    assert update.missing == (NEVER_EXISTED,)
    assert update.base_fields_missing == ()


def test_updating_again_downloads_nothing_new(writer: Sandbox, toy: Toy) -> None:
    """Check that a second update keeps the copy and has nothing left to fetch."""
    again = update_repo(
        writer,
        REPO,
        list(toy.prs.values()),
        timeout=120,
        source_url=lambda _: f"file://{UPSTREAM}",
    )
    assert not again.cloned
    assert again.refs_fetched == 0
    assert again.fetched_by_id == ()
    assert again.missing == (NEVER_EXISTED,)


def test_checkout_puts_each_side_on_disk(sandbox: Sandbox, toy: Toy) -> None:
    """Check the base side is the commit before the PR, and the PR side its last commit."""
    path, base = checkout(sandbox, toy.prs[1], "base")
    assert base == toy.sha["B"]
    assert sandbox.run(["ls", path], timeout=30).output.split() == ["a.py", "b.py"]
    path, head = checkout(sandbox, toy.prs[1], "pr")
    assert head == toy.sha["P2"]
    assert sandbox.run(["cat", f"{path}/p.py"], timeout=30).output == "p = 2\n"


def test_checkout_needs_the_shared_copy(sandbox: Sandbox) -> None:
    """Check a repo without a shared copy gives a clear error naming the download command."""
    other = toy_pr(1, [("a" * 40, [])], "b" * 40).model_copy(update={"repo": "owner/none"})
    with pytest.raises(RepoError, match="pr-reviewer sandbox repos"):
        checkout(sandbox, other, "pr")


def test_commit_check_finds_each_case(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str, toy: Toy
) -> None:
    """Check the base, the base_commit field, merges, extra files and missing commits."""
    checks = {
        check.instance_id: check
        for check in check_prs(docker, test_config, sandbox_image, list(toy.prs.values()))
    }
    later = checks["owner__toy-1"]
    assert later.usable
    assert (later.base_sha, later.base_field_relation, later.commits_apart) == (
        toy.sha["B"],
        "later",
        1,
    )
    assert (later.merge_commits, later.extra_files) == (0, ())
    assert later.python_files == {"base": 2, "pr": 3}

    merged = checks["owner__toy-2"]
    assert merged.usable
    assert (merged.base_sha, merged.base_field_relation) == (toy.sha["C"], "same")
    assert merged.merge_commits == 1
    assert merged.extra_files == ("d.py",)
    assert merged.only_pr_changes is False

    lost = checks["owner__toy-3"]
    assert lost.usable
    assert (lost.base_sha, lost.base_field_relation) == (toy.sha["D"], "same")

    missing = checks["owner__toy-4"]
    assert not missing.usable
    assert missing.missing_commits == (NEVER_EXISTED,)
    assert missing.problem == "1 of 1 reviewed commits missing"
