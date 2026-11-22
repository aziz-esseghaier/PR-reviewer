"""Tests for `pr-reviewer action run` (integrations/commands.py), with a fake GitHub.

The review itself (pipeline.review_live) is replaced by a stub that returns one proven
finding, so these tests need no Docker, database or model.

How to run (from the project folder):
    pytest tests/integrations/test_commands.py
"""

import argparse
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import Claim
from pr_reviewer.integrations import commands
from pr_reviewer.integrations.comments import ProvenFinding
from pr_reviewer.integrations.images import image_tags
from pr_reviewer.integrations.pipeline import LiveReview
from pr_reviewer.sandbox.config import load_sandbox_config
from tests.integrations.fakes import HEAD, PATCH, REPO, pull, two_commits


class FakeGitHub:
    """Answers like GitHub for pull request 7, and records the reviews posted.

    Args:
        token (str): Ignored.
    """

    posted: list[dict[str, Any]] = []
    fork = False
    existing: list[str] = []

    def __init__(self, token: str) -> None:
        self.commits, self.details = two_commits()

    def pull(self, repo: str, number: int) -> dict[str, Any]:
        """Give the pull request (from a fork when the class's `fork` is set)."""
        return pull(fork=self.fork)

    def pull_commits(self, repo: str, number: int) -> list[dict[str, Any]]:
        """Give its two commits, oldest first."""
        return self.commits

    def commit(self, repo: str, sha: str) -> dict[str, Any]:
        """Give one commit's full object."""
        return self.details[sha]

    def pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        """Give the files it changes, with their patches."""
        return [{"filename": "src/tinystats/stats.py", "patch": PATCH}]

    def review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """Give the comments already on it (the class's `existing` bodies)."""
        return [{"body": body} for body in self.existing]

    def post_review(
        self, repo: str, number: int, commit_id: str, body: str, comments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Record the review instead of posting it."""
        FakeGitHub.posted.append({"commit_id": commit_id, "body": body, "comments": comments})
        return {"id": 1}


def proven() -> ProvenFinding:
    """Build the proven finding the stub review returns.

    Returns:
        ProvenFinding: A ZeroDivisionError proof on line 8.
    """
    claim = Claim(
        file="src/tinystats/stats.py",
        start_line=8,
        end_line=8,
        category="F.4",
        summary="mean() divides by zero on empty input.",
        expected_failure="ZeroDivisionError",
        confidence=0.9,
    )
    test = "def test_mean_of_nothing():\n    assert mean([]) == 0.0\n"
    return ProvenFinding("f1", claim, "executed", "tests/test_stats.py", test, "ZeroDivisionError")


@pytest.fixture
def action(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Swap in the fake GitHub, a stub review and a fake Slack, and give the run's arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the module's helpers.
        tmp_path (Path): pytest's temporary folder, for the job summary.

    Returns:
        dict[str, Any]: The arguments, what the review saw, and the Slack messages sent.
    """
    FakeGitHub.posted = []
    FakeGitHub.fork = False
    FakeGitHub.existing = []
    seen: dict[str, Any] = {"reviewed": [], "slack": []}

    def review(pr: Instance) -> LiveReview:
        seen["reviewed"].append(pr)
        return LiveReview(
            pr, candidates=4, tiers={1: 1, 4: 3}, proven=[proven()], notes=["gate: 1"]
        )

    monkeypatch.setattr(commands, "GitHub", FakeGitHub)
    monkeypatch.setattr(commands, "review_live", review)
    monkeypatch.setattr(commands, "send_slack", lambda url, payload: seen["slack"].append(payload))
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    summary = tmp_path / "summary.md"
    seen["args"] = argparse.Namespace(
        repo=REPO, pr=7, recipes=None, dry_run=False, summary=str(summary)
    )
    seen["summary"] = summary
    return seen


def test_a_proven_finding_is_posted_as_an_inline_comment_on_the_head(
    action: dict[str, Any],
) -> None:
    """Check the review is posted at the head commit, with the comment on line 8."""
    assert commands._run(action["args"]) == 0
    [review] = FakeGitHub.posted
    assert review["commit_id"] == HEAD
    assert [(c["path"], c["line"]) for c in review["comments"]] == [("src/tinystats/stats.py", 8)]
    assert "found 1 problem proven by an executed test, out of 4" in review["body"]
    assert "Posted 1 comments (1 inline)" in action["summary"].read_text(encoding="utf-8")
    assert action["slack"] == []


def test_slack_gets_a_message_when_its_webhook_is_set(
    action: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the Slack message is sent after the review is posted."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    assert commands._run(action["args"]) == 0
    [message] = action["slack"]
    assert (
        "pr-reviewer proved 1 bug in <https://github.com/octo/demo/pull/7|octo/demo#7>"
        in message["text"]
    )


def test_a_fork_is_skipped_before_anything_runs(
    action: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check a fork's pull request isn't reviewed, posted on, or sent to Slack."""
    FakeGitHub.fork = True
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    assert commands._run(action["args"]) == 0
    assert action["reviewed"] == [] and FakeGitHub.posted == [] and action["slack"] == []
    assert "Skipped: the pull request comes from a fork" in action["summary"].read_text(
        encoding="utf-8"
    )


def test_a_finding_already_on_the_pr_isnt_posted_again(action: dict[str, Any]) -> None:
    """Check a rerun posts nothing new, and Slack hears nothing."""
    commands._run(action["args"])
    FakeGitHub.existing = [c["body"] for c in FakeGitHub.posted[0]["comments"]]
    FakeGitHub.posted = []
    assert commands._run(action["args"]) == 0
    assert FakeGitHub.posted == []


def test_a_dry_run_prints_instead_of_posting(
    action: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Check --dry-run posts nothing and shows the comment and the Slack message."""
    action["args"].dry_run = True
    assert commands._run(action["args"]) == 0
    assert FakeGitHub.posted == []
    out = capsys.readouterr().out
    assert "--- src/tinystats/stats.py:8" in out and "the Slack message would be" in out


def test_without_a_token_nothing_runs(
    action: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the command refuses to start without GITHUB_TOKEN."""
    monkeypatch.delenv("GITHUB_TOKEN")
    assert commands._run(action["args"]) == 1
    assert action["reviewed"] == []


def test_the_image_list_names_the_four_images() -> None:
    """Check the Action pulls the sandbox, proxy, indexer and tools images, each tagged."""
    tags = image_tags(load_sandbox_config())
    names = [tag.split(":")[0] for tag in tags]
    assert names == [
        "pr-reviewer-sandbox",
        "pr-reviewer-proxy",
        "pr-reviewer-indexer",
        "pr-reviewer-tools",
    ]
    assert all(len(tag.split(":")[1]) == 12 for tag in tags)
