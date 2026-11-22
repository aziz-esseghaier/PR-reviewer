"""The `pr-reviewer action ...` commands: review a live pull request, as the GitHub Action does.

cli.py adds them to the `pr-reviewer` command. action.yml (in the project folder) runs them on
GitHub's machines; they work the same on any machine with Docker and the database.

How to run (from the project folder, with Docker and the database running, and GITHUB_TOKEN
and GROQ_API_KEY set):
    pr-reviewer action run --repo octo/demo --pr 7
        review the pull request with every stage, post a review made only of comments for the
        findings a generated test proved, and send the Slack message when SLACK_WEBHOOK_URL is
        set. A pull request from a fork is skipped.
    pr-reviewer action run --repo octo/demo --pr 7 --dry-run
        the same, but print the comments and the Slack message instead of sending them
    pr-reviewer action run ... --recipes <file>
        load the repo's working recipes from the file first, and save them back after
    pr-reviewer action images [--build]
        list the Docker images the reviewer needs (building any that's missing with --build);
        the image workflow uses it to push them to GitHub's registry
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pr_reviewer.core.limits import LimitError
from pr_reviewer.db.connection import connect
from pr_reviewer.integrations.comments import (
    ReviewPlan,
    anchor,
    commentable_lines,
    plan_review,
    posted_fingerprints,
)
from pr_reviewer.integrations.github import GitHub, GitHubError
from pr_reviewer.integrations.images import build_all, image_tags
from pr_reviewer.integrations.live_pr import LivePrError, build_instance, is_fork
from pr_reviewer.integrations.pipeline import LiveReview, LiveReviewError, review_live
from pr_reviewer.integrations.recipes import RecipeFileError, import_recipes, write_recipes
from pr_reviewer.integrations.slack import SlackError, send_slack, slack_message
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.runner import docker_client

# GitHub lists at most this many commits of a pull request.
MAX_COMMITS = 250


def _summary_markdown(review: LiveReview | None, plan: ReviewPlan | None, note: str) -> str:
    """Write the job summary GitHub shows on the run's page.

    Args:
        review (LiveReview | None): How the review went; None when it didn't start.
        plan (ReviewPlan | None): What was posted; None when nothing was.
        note (str): Why the run ended early, or "".

    Returns:
        str: Markdown. The stage lines sit in a code block, as they can quote tool output.
    """
    lines = ["## pr-reviewer", ""]
    if note:
        lines += [note, ""]
    if review is not None:
        lines += ["```", *review.notes, "```", ""]
        if review.stopped:
            lines.append(f"Stopped early: {review.stopped}.")
        lines.append(f"Candidates checked: {review.candidates}. Proven: {len(review.proven)}.")
    if plan is not None:
        lines.append(
            f"Posted {len(plan.posted)} comments ({len(plan.comments)} inline); "
            f"{plan.skipped} were already on the pull request."
        )
    return "\n".join(lines) + "\n"


def _write_summary(path: str | None, text: str) -> None:
    """Add the job summary to the file GitHub reads, when there is one.

    Args:
        path (str | None): The file (GitHub's GITHUB_STEP_SUMMARY), or None.
        text (str): The Markdown.
    """
    if path:
        with Path(path).open("a", encoding="utf-8", newline="\n") as file:
            file.write(text)


def _post(github: GitHub, args: argparse.Namespace, review: LiveReview, base: str) -> ReviewPlan:
    """Post the proven findings as one review, and tell Slack.

    Args:
        github (GitHub): The GitHub client.
        args (argparse.Namespace): Parsed arguments. Uses `repo`, `pr` and `dry_run`.
        review (LiveReview): The review, with its proven findings.
        base (str): The PR's base commit.

    Returns:
        ReviewPlan: What was (or, with --dry-run, would be) posted.
    """
    pr = review.pr
    files = github.pull_files(args.repo, args.pr)
    diff_lines = {item["filename"]: commentable_lines(item.get("patch") or "") for item in files}
    already = posted_fingerprints(
        c.get("body") or "" for c in github.review_comments(args.repo, args.pr)
    )
    plan = plan_review(review.proven, diff_lines, already, base, pr.head_commit, review.candidates)
    if not plan.posted:
        print("Nothing new to post: every proven finding is already on the pull request.")
        return plan
    lines = [anchor(item.claim, diff_lines) for item in plan.posted]
    message = slack_message(args.repo, args.pr, plan.posted, lines)
    if args.dry_run:
        print("--dry-run: the review would be:\n" + plan.body)
        for comment in plan.comments:
            print(f"\n--- {comment['path']}:{comment['line']}\n{comment['body']}")
        print("\n--dry-run: the Slack message would be:\n" + json.dumps(message, indent=2))
        return plan
    github.post_review(args.repo, args.pr, pr.head_commit, plan.body, plan.comments)
    print(f"Posted a review with {len(plan.posted)} proven findings.")
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if webhook:
        try:
            send_slack(webhook, message)
            print("Sent the Slack message.")
        except SlackError as exc:
            # The review is posted; a Slack problem shouldn't fail the job.
            print(f"Slack message not sent: {exc}", file=sys.stderr)
    return plan


def _run(args: argparse.Namespace) -> int:
    """Review a live pull request and post what a test proved.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `repo`, `pr`, `recipes`, `dry_run`
            and `summary`.

    Returns:
        int: Exit code: 0 when the review ran (with or without findings, or skipped as a
        fork), 1 when it couldn't.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Set GITHUB_TOKEN (the Action passes its own).", file=sys.stderr)
        return 1
    github = GitHub(token)
    review: LiveReview | None = None
    plan: ReviewPlan | None = None
    note = ""
    try:
        pull = github.pull(args.repo, args.pr)
        if is_fork(pull):
            note = "Skipped: the pull request comes from a fork, which gets no secrets."
        elif pull.get("state") != "open":
            note = "Skipped: the pull request isn't open."
        else:
            commits = github.pull_commits(args.repo, args.pr)
            if len(commits) >= MAX_COMMITS:
                raise LivePrError(f"The pull request has {MAX_COMMITS} commits or more")
            details = {c["sha"]: github.commit(args.repo, c["sha"]) for c in commits}
            pr = build_instance(args.repo, pull, commits, details)
            if pr.head_commit != pull["head"]["sha"]:
                raise LivePrError("The pull request changed while it was read; run again")
            print(f"Reviewing {pr.instance_id} at {pr.head_commit[:12]} ({len(commits)} commits)")
            if args.recipes:
                with connect() as conn:
                    loaded = import_recipes(conn, args.recipes, args.repo)
                print(f"Loaded {loaded} cached recipes")
            review = review_live(pr)
            if args.recipes:
                with connect() as conn:
                    saved = write_recipes(conn, args.recipes, args.repo)
                print(f"Saved {saved} recipes for the next run")
            if review.proven:
                plan = _post(github, args, review, pr.base_commit)
            else:
                print("No finding was proven by a test, so nothing is posted.")
    except (GitHubError, LivePrError, LiveReviewError, RecipeFileError) as exc:
        print(f"Review problem: {exc}", file=sys.stderr)
        _write_summary(args.summary, _summary_markdown(review, plan, f"Failed: {exc}"))
        return 1
    except LimitError as exc:
        print(f"Stopped by the model's quota: {exc}", file=sys.stderr)
        _write_summary(args.summary, _summary_markdown(review, plan, "Stopped by the quota."))
        return 1
    if note:
        print(note)
    _write_summary(args.summary, _summary_markdown(review, plan, note))
    return 0


def _images(args: argparse.Namespace) -> int:
    """List the Docker images the reviewer needs, building missing ones with --build.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `build`.

    Returns:
        int: Exit code, 0 on success.
    """
    config = load_sandbox_config()
    tags = build_all(docker_client(), config) if args.build else image_tags(config)
    print("\n".join(tags))
    return 0


def _pr_number(text: str) -> int:
    """Read a pull request number from the command line.

    Args:
        text (str): The number as typed.

    Returns:
        int: The number.

    Raises:
        argparse.ArgumentTypeError: If it isn't a positive whole number.
    """
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"not a pull request number: {text!r}")
    return int(text)


def add_commands(commands: Any, data_option: argparse.ArgumentParser) -> None:
    """Add `pr-reviewer action ...` and its subcommands to the command line.

    Args:
        commands (Any): The top-level subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option (unused:
            a live PR isn't in the dataset).
    """
    action = commands.add_parser("action", help="review a live GitHub pull request (the Action)")
    sub = action.add_subparsers(metavar="ACTION_COMMAND", required=True)
    run = sub.add_parser("run", help="review a pull request and post what a test proved")
    run.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        required="GITHUB_REPOSITORY" not in os.environ,
        help="the repo, such as octo/demo (default: GITHUB_REPOSITORY)",
    )
    run.add_argument("--pr", type=_pr_number, required=True, help="the pull request's number")
    run.add_argument(
        "--recipes",
        type=Path,
        default=None,
        help="a file of the repo's working recipes to load first and save after",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the comments and the Slack message instead of sending them",
    )
    run.add_argument(
        "--summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="file to add the job summary to (default: GITHUB_STEP_SUMMARY)",
    )
    run.set_defaults(handler=_run)
    images = sub.add_parser("images", help="list (or --build) the Docker images the review needs")
    images.add_argument("--build", action="store_true", help="build any image that's missing")
    images.set_defaults(handler=_images)
