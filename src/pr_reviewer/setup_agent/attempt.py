"""Trying one recipe on one PR: install and test at the base, then again at the PR state.

An attempt runs in a fresh sandbox, so nothing from an earlier attempt can help or hurt it:
1. check out the PR's base, install the recipe (network on), run the related tests (network
   off);
2. check out the PR state, reinstall the repo's steps (a changed C file must be recompiled),
   and run the same tests again.
It passes only when both test runs pass. Then the sandbox can be saved as an environment
image, so later PRs with the same recipe start from it instead of installing from scratch. If
Docker can't save it, the attempt still passes, without an image.

The attempt's result says where it stopped and why, in our code's words (failure categories
come from the logs), plus the end of the failing output for the setup agent to read.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docker import DockerClient

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.repos import RepoError
from pr_reviewer.sandbox.runner import Sandbox, SandboxError
from pr_reviewer.setup_agent.install import environment, install
from pr_reviewer.setup_agent.testrun import CheckOutcome, existing, run_tests

# The name of saved environment images.
ENV_IMAGE = "pr-reviewer-env"


@dataclass(frozen=True)
class TryContext:
    """What every attempt needs: Docker, the sandbox settings and where logs go.

    Attributes:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings, with the time limits.
        base_image (str): The sandbox image, for attempts without an environment image.
        log_dir (Path | None): Folder for the sandboxes' command logs.
    """

    client: DockerClient
    config: SandboxConfig
    base_image: str
    log_dir: Path | None


@dataclass(frozen=True)
class Attempt:
    """How trying one recipe on one PR went.

    Attributes:
        source (str): Where the recipe came from: "cache", "standard" or "agent".
        recipe (dict[str, Any]): The recipe, as EnvRecipe data.
        stage (str): Where it stopped: "checkout", "install", "base_tests", "pr_install",
            "pr_tests", or "done" when everything passed.
        problem (str | None): The failure category, or None when it passed.
        detail (str): The end of the failing output, for the setup agent's model. When it
            passed, why no image was saved, if one was asked for and couldn't be.
        base (dict[str, Any] | None): The test counts at the base, when tests ran.
        pr_side (dict[str, Any] | None): The test counts at the PR state, when tests ran.
        minutes (float): How long it took.
        image (str | None): The environment image saved by it, if any.
        log_dir (str | None): The folder with every command's log.
    """

    source: str
    recipe: dict[str, Any]
    stage: str
    problem: str | None
    detail: str
    base: dict[str, Any] | None
    pr_side: dict[str, Any] | None
    minutes: float
    image: str | None
    log_dir: str | None

    @property
    def ok(self) -> bool:
        """Whether the environment passed at both sides."""
        return self.problem is None

    def describe(self) -> str:
        """Say in a few words how the attempt ended.

        Returns:
            str: Such as "passed (base 120/124, PR 121/124)" or "base_tests: tests_failing".
        """
        if self.ok and self.base and self.pr_side:
            return f"passed (base {_counts(self.base)}, PR {_counts(self.pr_side)})"
        return f"{self.stage}: {self.problem}"


def _counts(outcome: dict[str, Any]) -> str:
    """Write test counts as "passed/ran".

    Args:
        outcome (dict[str, Any]): CheckOutcome data.

    Returns:
        str: Such as "120/124".
    """
    ran = outcome["passed"] + outcome["failed"] + outcome["errors"]
    return f"{outcome['passed']}/{ran}"


def recipe_fingerprint(recipe: EnvRecipe) -> str:
    """Give a short fingerprint of what a recipe does, to tell recipes apart.

    Args:
        recipe (EnvRecipe): The recipe.

    Returns:
        str: 12 hex characters. Checks made with the recipe don't change it.
    """
    what = {
        "python": recipe.python_version,
        "steps": list(recipe.install_steps),
        "test": recipe.test_command,
        "env": dict(sorted(recipe.env.items())),
    }
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def env_image_tag(recipe: EnvRecipe, year: int) -> str:
    """Name the environment image of a recipe.

    Args:
        recipe (EnvRecipe): The recipe.
        year (int): The year it's for.

    Returns:
        str: The tag, such as "scikit-learn__scikit-learn-2022-3f1a9c0e2b7d".
    """
    return f"{recipe.repo.replace('/', '__')}-{year}-{recipe_fingerprint(recipe)}"


def runner_of(recipe: EnvRecipe) -> str:
    """Tell which test runner a recipe's test command uses.

    Args:
        recipe (EnvRecipe): The recipe.

    Returns:
        str: "django" for Django's runtests.py, otherwise "pytest".
    """
    return "django" if "runtests.py" in recipe.test_command else "pytest"


def try_recipe(
    ctx: TryContext,
    pr: Instance,
    recipe: EnvRecipe,
    source: str,
    tests: Sequence[str],
    *,
    start_image: str | None = None,
    save_image: bool = False,
) -> Attempt:
    """Try a recipe on a PR in a fresh sandbox: install and test at both sides.

    Args:
        ctx (TryContext): Docker, settings and log folder.
        pr (Instance): The PR.
        recipe (EnvRecipe): The recipe to try.
        source (str): Where it came from: "cache", "standard" or "agent".
        tests (Sequence[str]): The related test files, relative to the repo root.
        start_image (str | None): An environment image to start from. None means the plain
            sandbox image.
        save_image (bool): Save the sandbox as an environment image when the attempt passes.

    Returns:
        Attempt: Where it stopped and why, with the test counts.
    """
    started = time.monotonic()
    timeouts = ctx.config.timeouts
    env = environment(recipe, pr.created_at)
    runner = runner_of(recipe)
    data = recipe.model_dump(mode="json")

    def ended(
        stage: str,
        problem: str | None,
        detail: str = "",
        base: CheckOutcome | None = None,
        pr_side: CheckOutcome | None = None,
        image: str | None = None,
        log_dir: Path | None = None,
    ) -> Attempt:
        return Attempt(
            source=source,
            recipe=data,
            stage=stage,
            problem=problem,
            detail=detail[-1500:],
            base=asdict(base) if base else None,
            pr_side=asdict(pr_side) if pr_side else None,
            minutes=round((time.monotonic() - started) / 60, 2),
            image=image,
            log_dir=str(log_dir) if log_dir else None,
        )

    image = start_image or ctx.base_image
    try:
        with Sandbox(ctx.client, ctx.config, image, purpose="setup", log_dir=ctx.log_dir) as box:
            logs = box.log_dir
            try:
                path, _ = checkout(box, pr, "base")
            except RepoError as exc:
                return ended("checkout", "checkout_failed", str(exc), log_dir=logs)
            done = install(box, recipe, path, pr.created_at, timeout=timeouts.install_minutes * 60)
            if not done.ok:
                detail = f"Step `{done.step}` failed:\n{done.detail}"
                return ended("install", done.problem, detail, log_dir=logs)
            test_seconds = timeouts.test_minutes * 60
            files = existing(box, path, tests)
            base = run_tests(
                box, recipe.test_command, path, files, runner, timeout=test_seconds, env=env
            )
            if not base.ok:
                return ended("base_tests", base.problem, base.detail, base, log_dir=logs)
            path, _ = checkout(box, pr, "pr")
            done = install(
                box,
                recipe,
                path,
                pr.created_at,
                timeout=timeouts.install_minutes * 60,
                steps_only=True,
            )
            if not done.ok:
                detail = f"At the PR state, step `{done.step}` failed:\n{done.detail}"
                return ended("pr_install", done.problem, detail, base, log_dir=logs)
            files = existing(box, path, tests)
            pr_side = run_tests(
                box, recipe.test_command, path, files, runner, timeout=test_seconds, env=env
            )
            if not pr_side.ok:
                return ended("pr_tests", pr_side.problem, pr_side.detail, base, pr_side, None, logs)
            saved, note = None, ""
            if save_image:
                try:
                    # The download cache isn't needed later and can be large.
                    box.run(["sh", "-c", "uv cache clean >/dev/null 2>&1; true"], timeout=600)
                    saved = box.save_image(
                        ENV_IMAGE,
                        env_image_tag(recipe, pr.year),
                        {
                            "pr-reviewer.env": recipe.repo,
                            "pr-reviewer.recipe": recipe_fingerprint(recipe),
                        },
                    )
                except SandboxError as exc:
                    # The environment works; only the shortcut for later PRs is missing, and
                    # the next PR with this recipe saves it.
                    note = f"Passed, but not saved as an image: {exc}"
            return ended("done", None, note, base, pr_side, saved, logs)
    except (RepoError, SandboxError) as exc:
        return ended("sandbox", "sandbox_error", str(exc))
