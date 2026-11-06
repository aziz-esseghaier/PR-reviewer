"""Tests for running the probe in a real indexer sandbox (context/retrieve.py, probe.py).

Makes a small git repo inside an indexer sandbox (a base commit, a PR commit, a fix commit),
then runs the probe there as the real retrieval does: git finds the changed lines, Jedi
resolves a name re-exported through a package's __init__.py that `ast` can't, the answer comes
back as a file and passes the checks, and the fix mode lists the functions a fix changed.
Marked `docker`: skipped when Docker isn't running or the sandbox image isn't built (the
indexer image is built on it in seconds).

How to run (from the project folder):
    pytest tests/context/test_retrieve_docker.py
"""

from collections.abc import Iterator

import pytest
from docker import DockerClient

from pr_reviewer.context.candidates import Candidates, FixFunctions
from pr_reviewer.context.retrieve import ready_indexer, run_probe
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.runner import Sandbox

pytestmark = pytest.mark.docker

REPO = "/work/repo"

BASE_FILES = {
    "pkg/__init__.py": "from .core import helper\n",
    "pkg/core.py": (
        "def helper(x):\n"
        '    """Add one."""\n'
        "    return x + 1\n\n\n"
        "class Cart:\n"
        "    def __init__(self, items):\n"
        "        self.items = items\n\n"
        "    def average(self):\n"
        "        return sum(self.items) / len(self.items)\n"
    ),
    "pkg/use.py": "from pkg import helper\n\n\ndef bump(x):\n    return x\n",
    "tests/test_core.py": (
        "from pkg.core import Cart\n\n\ndef test_average():\n    assert Cart([2]).average() == 2\n"
    ),
}


@pytest.fixture
def indexer(docker: DockerClient, test_config: SandboxConfig, sandbox_image: str) -> str:
    """Give the indexer image, building it on the sandbox image if needed.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image (skips the test when it isn't built).

    Returns:
        str: The indexer image's name and tag.
    """
    return ready_indexer(docker, test_config)


@pytest.fixture
def box(docker: DockerClient, test_config: SandboxConfig, indexer: str) -> Iterator[Sandbox]:
    """Start an indexer sandbox, network off, and remove it after the test.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        indexer (str): The indexer image.

    Yields:
        Sandbox: The sandbox.
    """
    with Sandbox(docker, test_config, indexer, purpose="test-context") as started:
        yield started


def git(box: Sandbox, *args: str) -> str:
    """Run git in the test repo and give its output.

    Args:
        box (Sandbox): The sandbox.
        *args (str): The git arguments.

    Returns:
        str: What git printed, stripped.
    """
    command = ["git", "-C", REPO, "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
    result = box.run(command, timeout=60, label="git")
    assert result.ok, result.output
    return result.output.strip()


def commit(box: Sandbox, files: dict[str, str], message: str) -> str:
    """Write files into the test repo and commit them.

    Args:
        box (Sandbox): The sandbox.
        files (dict[str, str]): File contents by path in the repo.
        message (str): The commit message.

    Returns:
        str: The new commit's ID.
    """
    box.write_files(REPO, files)
    git(box, "add", "-A")
    git(box, "commit", "-q", "-m", message)
    return git(box, "rev-parse", "HEAD")


def test_the_probe_runs_with_git_and_jedi(box: Sandbox) -> None:
    """Check changed lines from git, a re-export resolved by Jedi, and the fix mode."""
    made = box.run(["mkdir", "-p", f"{REPO}/pkg", f"{REPO}/tests"], timeout=60)
    assert made.ok
    git(box, "init", "-q")
    base = commit(box, BASE_FILES, "base")
    pr_files = {**BASE_FILES, "pkg/use.py": BASE_FILES["pkg/use.py"].replace("x\n", "helper(x)\n")}
    head = commit(box, pr_files, "pr")
    request = {"root": REPO, "base": base, "head": head, "max_piece_chars": 12_000}
    answer = run_probe(box, "retrieve", request, timeout=300)
    found = Candidates.model_validate({**answer, "instance_id": "psf__requests-1"})
    assert [(c.file, c.qualname) for c in found.changed] == [("pkg/use.py", "bump")]
    assert found.stats["jedi_available"] == 1
    assert found.stats["resolved_jedi"] == 1
    called = [p for p in found.pieces if p.kind == "called"]
    assert [(p.file, p.qualname) for p in called] == [("pkg/core.py", "helper")]

    # A fix that changes Cart.average and adds a test after test_average: only Cart.average
    # counts as changed (code added after a function's last line isn't a change to it).
    fix_files = dict(pr_files)
    fix_files["pkg/core.py"] = pr_files["pkg/core.py"].replace(
        "/ len(self.items)", "/ max(1, len(self.items))"
    )
    fix_files["tests/test_core.py"] = pr_files["tests/test_core.py"] + (
        "\n\ndef test_empty():\n    assert Cart([]).average() == 0\n"
    )
    fix = commit(box, fix_files, "fix")
    fixes = run_probe(box, "fixdefs", {"root": REPO, "commits": [fix]}, timeout=300)["fixes"]
    result = FixFunctions.model_validate(fixes[fix])
    assert [(d.file, d.qualname, d.test) for d in result.defs] == [
        ("pkg/core.py", "Cart.average", False)
    ]
    assert result.error == ""
