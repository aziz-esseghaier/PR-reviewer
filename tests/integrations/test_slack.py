"""Tests for the Slack message (integrations/slack.py).

How to run (from the project folder):
    pytest tests/integrations/test_slack.py
"""

import json

import httpx2
import pytest

from pr_reviewer.core.schemas import Claim
from pr_reviewer.integrations.comments import ZERO_WIDTH, ProvenFinding
from pr_reviewer.integrations.slack import (
    SlackError,
    escape_slack,
    pr_link,
    send_slack,
    slack_message,
)


def finding(summary: str) -> ProvenFinding:
    """Build a proven finding with a given summary.

    Args:
        summary (str): The claim's summary.

    Returns:
        ProvenFinding: The finding.
    """
    claim = Claim(
        file="src/tinystats/stats.py",
        start_line=8,
        end_line=8,
        category="F.4",
        summary=summary,
        expected_failure="ZeroDivisionError",
        confidence=0.9,
    )
    return ProvenFinding(
        "f1", claim, "executed", "tests/test_stats.py", "def test(): pass", "ZeroDivisionError"
    )


def test_the_message_is_built_from_fixed_fields_with_a_link_to_the_pr() -> None:
    """Check the count, the PR link, the place, the category and the error."""
    message = slack_message("octo/demo", 7, [finding("mean() divides by zero.")], [8])
    text = message["text"]
    assert text.startswith(
        "pr-reviewer proved 1 bug in <https://github.com/octo/demo/pull/7|octo/demo#7>:"
    )
    assert "`src/tinystats/stats.py:8` (F.4, `ZeroDivisionError`): mean() divides by zero." in text
    assert message["unfurl_links"] is False and message["unfurl_media"] is False


def test_model_text_cant_mention_link_or_ping_the_channel() -> None:
    """Check Slack's control syntax and addresses in a summary are disarmed."""
    safe = escape_slack("<!channel> see <https://evil.example|here> @here `x`", 200)
    assert "<" not in safe and ">" not in safe
    assert "https://" not in safe
    assert f"@{ZERO_WIDTH}here" in safe and "`" not in safe
    message = slack_message("octo/demo", 7, [finding("<!channel> " + "long " * 55)], [None])
    lines = message["text"].splitlines()
    # The only link is ours; the summary is cut.
    assert message["text"].count("<") == 1
    assert len(lines[1]) < 260


def test_a_long_list_is_cut_after_five() -> None:
    """Check at most five findings are listed."""
    message = slack_message("octo/demo", 7, [finding(f"bug {n}") for n in range(7)], [8] * 7)
    assert "… and 2 more on the pull request." in message["text"]


def test_the_link_only_uses_a_checked_repo_name() -> None:
    """Check a repo name that could change the address is refused."""
    assert pr_link("octo/demo", 7) == "https://github.com/octo/demo/pull/7"
    with pytest.raises(Exception, match="Not a GitHub repo name"):
        pr_link("octo/demo/../../x", 7)


def test_only_slack_webhooks_get_the_message() -> None:
    """Check another address is refused, and the payload is posted as JSON."""
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, text="ok")

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    with pytest.raises(SlackError, match="must start with"):
        send_slack("https://evil.example/hook", {"text": "x"}, http=client)
    send_slack("https://hooks.slack.com/services/T/B/X", {"text": "hello"}, http=client)
    assert len(sent) == 1 and json.loads(sent[0].content) == {"text": "hello"}


def test_a_refused_message_raises_without_the_address() -> None:
    """Check Slack's refusal is reported, and the secret address isn't in the error."""
    client = httpx2.Client(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(404, text="no_service"))
    )
    with pytest.raises(SlackError) as error:
        send_slack("https://hooks.slack.com/services/T/B/SECRET", {"text": "x"}, http=client)
    assert "404" in str(error.value) and "SECRET" not in str(error.value)
