"""
End-to-end tests of a setup attempt (setup_agent/attempt.py) in real sandboxes.

A tiny package with three tests is built inside a container, with a PR that adds a function
and a test. Its shared copy is made like the real ones, then:
- the probe and the standard recipe install it for real: uv downloads Python and pytest
  through the proxy, the tests run at the base and at the PR state, and the environment is
  saved as an image;
- a recipe that asks for a package that doesn't exist fails, named as a resolution failure;
- when Docker can't save the image, the attempt still passes, without one.
These need Docker, the images built, and an internet connection (PyPI and GitHub, through the
proxy), and take a few minutes.

How to run (from the project folder):
    pytest tests/setup_agent/test_setup_docker.py
"""

import json
from collections.abc import Iterator

import pytest
from docker import DockerClient

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.repos import update_repo
from pr_reviewer.sandbox.runner import Sandbox, SandboxError
from pr_reviewer.setup_agent.attempt import TryContext, try_recipe
from pr_reviewer.setup_agent.facts import probe
from pr_reviewer.setup_agent.standard import standard_recipe
from tests.benchmark.test_dataset import make_row

pytestmark = pytest.mark.docker

REPO = "owner/toypkg"
UPSTREAM = "/repos/_upstream/toypkg"

# Builds the package's repo: a base commit, then a PR commit; prints both IDs.
BUILD = f"""
set -e
rm -rf {UPSTREAM} /repos/owner__toypkg.git
mkdir -p {UPSTREAM}/toypkg {UPSTREAM}/tests && cd {UPSTREAM}
git init -q -b main && git config user.email t@example.com && git config user.name T
cat > setup.py <<'EOF'
from setuptools import setup
setup(name="toypkg", version="1.0", packages=["toypkg"], python_requires=">=3.8")
EOF
cat > toypkg/__init__.py <<'EOF'
def add(a, b):
    return a + b
EOF
cat > tests/test_toypkg.py <<'EOF'
from toypkg import add

def test_one():
    assert add(1, 1) == 2

def test_two():
    assert add(2, 2) == 4

def test_three():
    assert add(0, 0) == 0
EOF
git add -A && git commit -q -m base && BASE=$(git rev-parse HEAD)
cat >> toypkg/__init__.py <<'EOF'

def sub(a, b):
    return a - b
EOF
git commit -q -am "Add sub" && PR=$(git rev-parse HEAD)
git update-ref refs/pull/1/head $PR
echo BASE=$BASE PR=$PR
"""


@pytest.fixture(scope="module")
def toy_pr(docker: DockerClient, test_config: SandboxConfig, sandbox_image: str) -> Instance:
    """Build the package's repo and its shared copy, and describe its PR.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.

    Returns:
        Instance: The PR, whose one reviewed commit adds sub().
    """
    with Sandbox(
        docker, test_config, sandbox_image, purpose="test-writer", repos_writable=True
    ) as box:
        built = box.run(["sh", "-c", BUILD], timeout=120)
        assert built.ok, built.output
        sha = dict(pair.split("=") for pair in built.output.split())
        row = make_row(1, year=2022, issues=(), repo=REPO)
        row["pr_commits"][0]["sha"] = sha["PR"]
        row["pr_commits"][0]["diff"] = [{"file": "toypkg/__init__.py", "patch": "@@"}]
        pr = Instance.model_validate_json(json.dumps(row))
        update_repo(box, REPO, [pr], timeout=120, source_url=lambda _: f"file://{UPSTREAM}")
    return pr


@pytest.fixture
def ctx(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> Iterator[TryContext]:
    """Give the attempt context, and remove environment images the test saved.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.

    Yields:
        TryContext: The context.
    """
    yield TryContext(docker, test_config, sandbox_image, None)
    for image in docker.images.list(filters={"label": f"pr-reviewer.env={REPO}"}):
        if image.id:
            docker.images.remove(image.id, force=True)


def test_standard_recipe_builds_a_working_environment(
    ctx: TryContext, toy_pr: Instance, sandbox_image: str
) -> None:
    """Check the whole path: probe, standard recipe, install, tests at both sides, image."""
    with Sandbox(ctx.client, ctx.config, sandbox_image, purpose="test") as box:
        path, _ = checkout(box, toy_pr, "base")
        facts = probe(box, path)
    assert facts.python_requires == ">=3.8"
    recipe = standard_recipe(toy_pr, facts)
    assert recipe.install_steps == ("uv pip install -e .", "uv pip install pytest")
    attempt = try_recipe(ctx, toy_pr, recipe, "standard", ["tests/test_toypkg.py"], save_image=True)
    assert attempt.ok, attempt.detail
    assert attempt.base is not None and attempt.base["passed"] == 3
    assert attempt.pr_side is not None and attempt.pr_side["passed"] == 3
    assert attempt.image is not None and attempt.image.startswith(
        "pr-reviewer-env:owner__toypkg-2022-"
    )


def test_missing_package_is_a_resolution_failure(ctx: TryContext, toy_pr: Instance) -> None:
    """Check a recipe asking for a package that doesn't exist stops at install, named so."""
    recipe = EnvRecipe(
        repo=REPO,
        python_version="3.9",
        install_steps=("uv pip install no-such-package-pr-reviewer-test==9.9",),
        test_command="python -m pytest",
    )
    attempt = try_recipe(ctx, toy_pr, recipe, "agent", ["tests/test_toypkg.py"])
    assert not attempt.ok
    assert (attempt.stage, attempt.problem) == ("install", "dependency_resolution")
    assert "no-such-package-pr-reviewer-test" in attempt.detail


def test_a_failed_save_keeps_a_passing_attempt(
    ctx: TryContext, toy_pr: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that when Docker can't save the image, the attempt passes without one."""

    def stuck(*args: object, **kwargs: object) -> str:
        raise SandboxError("Can't save it as an image: Read timed out.")

    monkeypatch.setattr(Sandbox, "save_image", stuck)
    recipe = EnvRecipe(
        repo=REPO,
        python_version="3.9",
        install_steps=("uv pip install -e .", "uv pip install pytest"),
        test_command="python -m pytest",
    )
    attempt = try_recipe(ctx, toy_pr, recipe, "standard", ["tests/test_toypkg.py"], save_image=True)
    assert attempt.ok, attempt.detail
    assert attempt.image is None
    assert attempt.detail.startswith("Passed, but not saved as an image: Can't save")
