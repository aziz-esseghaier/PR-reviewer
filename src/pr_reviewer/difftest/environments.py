"""Prepared PR environments: a Docker image of a PR's repo installed at each of its sides.

A differential run needs the repo installed at the PR's base and at its PR state. Installing
takes from a minute to half an hour (scikit-learn compiles), so it's done once per PR and side
and saved as an image:
- pr-reviewer-pr:<instance_id>-base: the working copy at the base, with the recipe installed;
- pr-reviewer-pr:<instance_id>-pr: the same, then switched to the PR state and reinstalled.
Every later run starts a sandbox from one of these in seconds.

The recipe is the one that passed the PR's setup (setup_agent), and preparing starts from the
environment image saved with it, so most dependencies are already there. Each image is labelled
with the recipe's fingerprint: when a PR's setup later passes with another recipe, its images
no longer match and are prepared again.

Imported by other modules; the file isn't run directly.
"""

import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import docker
import psycopg
from psycopg.rows import TupleRow

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.runner import DOCKER_ERRORS, Sandbox, SandboxError
from pr_reviewer.setup_agent.attempt import TryContext, recipe_fingerprint
from pr_reviewer.setup_agent.install import install

# The name of prepared PR images.
PR_IMAGE = "pr-reviewer-pr"

# The two sides a PR is prepared at.
PreparedSide = Literal["base", "pr"]
SIDES: tuple[PreparedSide, ...] = ("base", "pr")


class PrepareError(SandboxError):
    """Raised when a PR has no working environment, or preparing one fails."""


@dataclass(frozen=True)
class PrEnvironment:
    """What a PR's environment is built from.

    Attributes:
        instance_id (str): The PR.
        recipe (EnvRecipe): The recipe that passed its setup.
        start_image (str | None): The environment image saved with that recipe, if any.
    """

    instance_id: str
    recipe: EnvRecipe
    start_image: str | None


def pr_environment(conn: psycopg.Connection[TupleRow], pr: Instance) -> PrEnvironment:
    """Find the recipe that passed a PR's most recent setup.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        pr (Instance): The PR.

    Returns:
        PrEnvironment: Its recipe and environment image.

    Raises:
        PrepareError: If the PR was never set up, or its latest setup didn't pass.
    """
    row = conn.execute(
        """
        SELECT s.verified, r.recipe, r.image
        FROM setup_runs s LEFT JOIN env_recipes r ON r.id = s.recipe_id
        WHERE s.instance_id = %s
        ORDER BY s.finished_at DESC, s.id DESC
        LIMIT 1
        """,
        (pr.instance_id,),
    ).fetchone()
    if row is None:
        raise PrepareError(
            f"{pr.instance_id} has no environment yet. Set it up first: pr-reviewer setup run "
            f"--ids {pr.instance_id}"
        )
    verified, recipe, image = row
    if not verified or recipe is None:
        raise PrepareError(f"{pr.instance_id}'s latest setup didn't pass, so it has no environment")
    return PrEnvironment(pr.instance_id, EnvRecipe.model_validate(recipe), image)


def prepared_tag(pr: Instance, side: PreparedSide) -> str:
    """Name a PR's prepared image for one side.

    Args:
        pr (Instance): The PR.
        side (PreparedSide): "base" or "pr".

    Returns:
        str: Such as "pr-reviewer-pr:psf__requests-2317-base".
    """
    return f"{PR_IMAGE}:{pr.instance_id}-{side}"


def find_prepared(
    ctx: TryContext, pr: Instance, recipe: EnvRecipe
) -> dict[PreparedSide, str] | None:
    """Find a PR's prepared images, if both exist and match its recipe.

    Args:
        ctx (TryContext): Docker and the sandbox settings.
        pr (Instance): The PR.
        recipe (EnvRecipe): The recipe its images must have been prepared with.

    Returns:
        dict[PreparedSide, str] | None: Each side's image tag, or None when either is missing
        or was prepared with another recipe.
    """
    found: dict[PreparedSide, str] = {}
    for side in SIDES:
        tag = prepared_tag(pr, side)
        try:
            image = ctx.client.images.get(tag)
        except docker.errors.ImageNotFound:
            return None
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't look up {tag}: {exc}") from None
        if image.labels.get("pr-reviewer.recipe") != recipe_fingerprint(recipe):
            return None
        found[side] = tag
    return found


def _remove_old(ctx: TryContext, ids: list[str]) -> None:
    """Remove images replaced by newly prepared ones, when Docker allows it.

    Args:
        ctx (TryContext): Docker.
        ids (list[str]): The old images' IDs, newest side first.
    """
    for image_id in ids:
        with contextlib.suppress(*DOCKER_ERRORS):
            ctx.client.images.remove(image_id)


def prepare(
    ctx: TryContext,
    pr: Instance,
    env: PrEnvironment,
    on_step: Callable[[str], None] | None = None,
) -> dict[PreparedSide, str]:
    """Install a PR's repo at its base and at its PR state, and save each as an image.

    One sandbox does both: install at the base and save, then switch to the PR state,
    reinstall (a changed C file must be recompiled) and save again.

    Args:
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        env (PrEnvironment): Its recipe and the image to start from.
        on_step (Callable[[str], None] | None): Called with a line per finished step.

    Returns:
        dict[PreparedSide, str]: Each side's image tag.

    Raises:
        PrepareError: If an install fails.
        SandboxError: If Docker fails.
    """
    report = on_step or (lambda line: None)
    recipe = env.recipe
    timeout = ctx.config.timeouts.install_minutes * 60
    old_ids = []
    for side in reversed(SIDES):
        with contextlib.suppress(*DOCKER_ERRORS):
            old = ctx.client.images.get(prepared_tag(pr, side)).id
            if old:
                old_ids.append(old)
    labels = {
        "pr-reviewer.pr": pr.instance_id,
        "pr-reviewer.recipe": recipe_fingerprint(recipe),
    }
    tags: dict[PreparedSide, str] = {}
    image = env.start_image or ctx.base_image
    with Sandbox(ctx.client, ctx.config, image, purpose="prepare", log_dir=ctx.log_dir) as box:
        for side in SIDES:
            path, sha = checkout(box, pr, side)
            done = install(
                box, recipe, path, pr.created_at, timeout=timeout, steps_only=side == "pr"
            )
            if not done.ok:
                raise PrepareError(
                    f"{pr.instance_id}: installing at the {side} side ({sha[:10]}) failed at "
                    f"`{done.step}` ({done.problem}): {done.detail[-500:]}"
                )
            # The download cache isn't needed later and can be large.
            box.run(["sh", "-c", "uv cache clean >/dev/null 2>&1; true"], timeout=600)
            tag = prepared_tag(pr, side)
            name, _, version = tag.partition(":")
            box.save_image(name, version, {**labels, "pr-reviewer.side": side})
            tags[side] = tag
            report(f"prepared {tag}")
    _remove_old(ctx, [i for i in old_ids if i not in _ids(ctx, tags.values())])
    return tags


def _ids(ctx: TryContext, tags: Iterable[str]) -> set[str]:
    """Give the image IDs of some tags.

    Args:
        ctx (TryContext): Docker.
        tags (Iterable[str]): Image tags.

    Returns:
        set[str]: Their IDs; tags Docker can't find are left out.
    """
    ids = set()
    for tag in tags:
        with contextlib.suppress(*DOCKER_ERRORS):
            image_id = ctx.client.images.get(tag).id
            if image_id:
                ids.add(image_id)
    return ids


def ready_prepared(
    ctx: TryContext,
    pr: Instance,
    env: PrEnvironment,
    on_step: Callable[[str], None] | None = None,
) -> dict[PreparedSide, str]:
    """Give a PR's prepared images, preparing them first when needed.

    Args:
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        env (PrEnvironment): Its recipe and the image to start from.
        on_step (Callable[[str], None] | None): Called with a line per finished step.

    Returns:
        dict[PreparedSide, str]: Each side's image tag.
    """
    return find_prepared(ctx, pr, env.recipe) or prepare(ctx, pr, env, on_step)
