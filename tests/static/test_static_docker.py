"""Tests for running the static tools in a real sandbox (static/tools.py).

Makes a small git repo inside a sandbox with the tools volume attached (a base commit, then a
PR commit that adds code each tool flags), with an environment at /work/venv as a prepared PR
image has, then finds the changed lines and runs the four tools as a real run does: each
tool finishes, its JSON is read back, and only warnings on changed lines are kept. Marked
`docker`: skipped when Docker isn't running or the sandbox or tools image isn't built
(`pr-reviewer static run` builds the tools image).

How to run (from the project folder):
    pytest tests/static/test_static_docker.py
"""

from collections.abc import Iterator

import pytest
from docker import DockerClient

from pr_reviewer.sandbox.config import Limits, SandboxConfig
from pr_reviewer.sandbox.image import image_exists
from pr_reviewer.sandbox.runner import Sandbox
from pr_reviewer.static.tools import (
    TOOLS_DIR,
    ReadyTools,
    StaticSettings,
    analyze,
    changed_lines,
    ready_tools,
    tools_image,
)
from pr_reviewer.static.warnings import on_changed_lines

pytestmark = pytest.mark.docker

REPO = "/work/repo"

# Enough for semgrep and mypy, which the test config's small limits aren't.
LIMITS = Limits(cpus=2, memory_gb=2, processes=512, lifetime_minutes=20)

BASE = "import os\n\n\ndef total(items):\n    return sum(items)\n"

# The PR keeps the unused `import os` (ruff F401, on an unchanged line) and adds code that
# each tool flags.
HEAD = (
    BASE
    + "\n\n"
    + "import subprocess\n\n\n"
    + "def count(items: list[int]) -> int:\n"
    + '    return "many"\n\n\n'
    + "def run(cmd):\n"
    + "    subprocess.call(cmd, shell=True)\n"
    + "    return totl\n\n\n"
    + "def same(x):\n"
    + "    return x == x\n"
)


@pytest.fixture
def tools(docker: DockerClient, test_config: SandboxConfig, sandbox_image: str) -> ReadyTools:
    """Give the tools and their volume, or skip until the tools image is built.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image (skips the test when it isn't built).

    Returns:
        ReadyTools: The tools image, its volume and the versions.
    """
    if not image_exists(docker, tools_image(test_config).tag):
        pytest.skip("Build the tools image first: pr-reviewer static run")
    return ready_tools(docker, test_config)


@pytest.fixture
def box(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str, tools: ReadyTools
) -> Iterator[Sandbox]:
    """Start a sandbox with the tools attached, network off, and remove it after the test.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.
        tools (ReadyTools): The tools and their volume.

    Yields:
        Sandbox: The sandbox.
    """
    with Sandbox(
        docker,
        test_config,
        sandbox_image,
        purpose="test-static",
        limits=LIMITS,
        volumes={tools.volume: TOOLS_DIR},
    ) as started:
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


def commit(box: Sandbox, text: str, message: str) -> str:
    """Write pkg/core.py into the test repo and commit it.

    Args:
        box (Sandbox): The sandbox.
        text (str): The file's content.
        message (str): The commit message.

    Returns:
        str: The new commit's ID.
    """
    box.write_files(REPO, {"pkg/__init__.py": "", "pkg/core.py": text})
    git(box, "add", "-A")
    git(box, "commit", "-q", "-m", message)
    return git(box, "rev-parse", "HEAD")


def test_every_tool_runs_and_only_changed_lines_are_kept(box: Sandbox, tools: ReadyTools) -> None:
    """Check the versions, the changed lines, each tool's warning, and the line filter."""
    assert "ruff 0.17.4" in tools.versions and "semgrep-rules" in tools.versions
    assert box.run(["mkdir", "-p", f"{REPO}/pkg"], timeout=60).ok
    venv = box.run(["uv", "venv", "--python", "/usr/bin/python3", "/work/venv"], timeout=120)
    assert venv.ok, venv.output
    git(box, "init", "-q")
    base = commit(box, BASE, "base")
    head = commit(box, HEAD, "pr")
    changed = changed_lines(box, REPO, base, head)
    assert list(changed) == ["pkg/core.py"] and changed["pkg/core.py"][0] == 6
    runs, warnings = analyze(box, REPO, changed, "3.11", StaticSettings())
    assert [(run.tool, run.status) for run in runs] == [
        ("ruff", "ok"),
        ("mypy", "ok"),
        ("bandit", "ok"),
        ("semgrep", "ok"),
    ], [run.detail for run in runs]
    assert runs[1].mode == "silent"
    found = {(w.tool, w.rule) for w in warnings}
    assert {("ruff", "F401"), ("ruff", "F821"), ("mypy", "return-value"), ("bandit", "B602")} <= (
        found
    )
    assert any(w.tool == "semgrep" for w in warnings), found
    kept = on_changed_lines(warnings, changed)
    assert ("ruff", "F401") not in {(w.tool, w.rule) for w in kept}
    assert {w.tool for w in kept} == {"ruff", "mypy", "bandit", "semgrep"}
