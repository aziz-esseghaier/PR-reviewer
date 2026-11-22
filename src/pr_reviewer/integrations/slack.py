"""Tell a Slack channel what the reviewer proved on a pull request.

The message goes to a Slack incoming webhook, whose address the Action keeps as a secret. Our
code builds it from fixed fields: the repo, the PR's number, a link to the PR (built from those
two, never taken from PR text), and per finding its file and line, category, error and a
shortened summary. The summary comes from the model, so it's escaped: Slack's control
characters (&, <, >) can't form links or mentions like <!channel>, "@" and "://" get a
zero-width space, and link previews are off.

It's sent only when a comment was posted, and never for a pull request from a fork (the
Action skips those before any of this runs).

Imported by other modules; the file isn't run directly.
"""

import re
from collections.abc import Sequence
from typing import Any

import httpx2

from pr_reviewer.integrations.comments import ZERO_WIDTH, ProvenFinding, shorten
from pr_reviewer.integrations.github import check_repo

# Where Slack's incoming webhooks live; any other address is refused.
WEBHOOK_PREFIX = "https://hooks.slack.com/"

# Longest summary shown per finding, and most findings listed.
SUMMARY_CHARS = 150
MAX_LISTED = 5

TIMEOUT_S = 15.0

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class SlackError(Exception):
    """Raised when the message can't be sent."""


def escape_slack(text: str, limit: int) -> str:
    """Make untrusted text safe in a Slack message, as one short line.

    Args:
        text (str): The text, such as a claim's summary.
        limit (int): Most characters kept before escaping.

    Returns:
        str: The text on one line, with &, < and > escaped as Slack asks, backticks removed,
        and mentions and addresses broken by a zero-width space.
    """
    plain = shorten(re.sub(r"\s+", " ", _CONTROL.sub(" ", text)).strip(), limit)
    plain = plain.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    plain = plain.replace("`", "'").replace("@", "@" + ZERO_WIDTH)
    return plain.replace("://", ":" + ZERO_WIDTH + "//")


def pr_link(repo: str, number: int) -> str:
    """Build the link to a pull request from its repo and number.

    Args:
        repo (str): The repo, such as "octo/demo".
        number (int): The pull request's number.

    Returns:
        str: Such as "https://github.com/octo/demo/pull/7".
    """
    return f"https://github.com/{check_repo(repo)}/pull/{int(number)}"


def slack_message(
    repo: str, number: int, proven: Sequence[ProvenFinding], lines: Sequence[int | None]
) -> dict[str, Any]:
    """Write the message about the findings posted on a pull request.

    Args:
        repo (str): The repo.
        number (int): The pull request's number.
        proven (Sequence[ProvenFinding]): The findings posted.
        lines (Sequence[int | None]): The line each one's comment sits on (None when it went
            in the review's summary), in the same order.

    Returns:
        dict[str, Any]: The webhook's JSON payload.
    """
    count = len(proven)
    noun = "bug" if count == 1 else "bugs"
    title = f"{check_repo(repo)}#{int(number)}"
    text = [f"pr-reviewer proved {count} {noun} in <{pr_link(repo, number)}|{title}>:"]
    for item, line in list(zip(proven, lines, strict=True))[:MAX_LISTED]:
        claim = item.claim
        where = escape_slack(f"{claim.file}:{line or claim.start_line}", 120)
        error = escape_slack(item.failure, 80)
        summary = escape_slack(claim.summary, SUMMARY_CHARS)
        text.append(f"• `{where}` ({claim.category}, `{error}`): {summary}")
    if count > MAX_LISTED:
        text.append(f"… and {count - MAX_LISTED} more on the pull request.")
    text.append("Each one has a generated test that passes before the PR and fails after it.")
    return {"text": "\n".join(text), "unfurl_links": False, "unfurl_media": False}


def send_slack(webhook: str, payload: dict[str, Any], http: httpx2.Client | None = None) -> None:
    """Post a message to a Slack incoming webhook.

    Args:
        webhook (str): The webhook's address, from the Action's secret. Never printed.
        payload (dict[str, Any]): The message (slack_message()).
        http (httpx2.Client | None): The HTTP client. Tests pass one with a fake transport.

    Raises:
        SlackError: If the address isn't a Slack webhook, or Slack refuses the message.
    """
    if not webhook.startswith(WEBHOOK_PREFIX):
        raise SlackError("The Slack webhook address must start with " + WEBHOOK_PREFIX)
    client = http or httpx2.Client(timeout=TIMEOUT_S)
    try:
        response = client.post(webhook, json=payload)
    except httpx2.HTTPError as exc:
        # The exception's text can hold the address, which is a secret.
        raise SlackError(f"Sending to Slack failed ({type(exc).__name__})") from None
    if response.status_code >= 400:
        raise SlackError(f"Slack answered {response.status_code}: {response.text[:200]}")
