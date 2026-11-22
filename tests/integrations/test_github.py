"""Tests for the GitHub client (integrations/github.py), against a fake GitHub.

Checks that a review is always a plain comment, the headers, paging, and refusals.

How to run (from the project folder):
    pytest tests/integrations/test_github.py
"""

import json
from collections.abc import Callable

import httpx2
import pytest

from pr_reviewer.integrations.github import REVIEW_EVENT, GitHub, GitHubError

Handler = Callable[[httpx2.Request], httpx2.Response]


def client(handler: Handler) -> GitHub:
    """Build a GitHub client that talks to a fake.

    Args:
        handler (Handler): Answers each request.

    Returns:
        GitHub: The client.
    """
    return GitHub("t0ken", http=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_a_review_is_always_a_plain_comment() -> None:
    """Check the review's event is COMMENT, never APPROVE or REQUEST_CHANGES."""
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json={"id": 1})

    comments = [{"path": "a.py", "line": 3, "side": "RIGHT", "body": "x"}]
    client(handler).post_review("octo/demo", 7, "c" * 40, "summary", comments)
    assert REVIEW_EVENT == "COMMENT"
    request = sent[0]
    assert request.method == "POST"
    assert request.url.path == "/repos/octo/demo/pulls/7/reviews"
    payload = json.loads(request.content)
    assert payload["event"] == "COMMENT" and payload["comments"] == comments
    assert request.headers["Authorization"] == "Bearer t0ken"


def test_lists_are_read_page_by_page() -> None:
    """Check a full page asks for the next one, and a short page ends the list."""
    pages: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        page = request.url.params["page"]
        pages.append(page)
        count = 100 if page == "1" else 3
        return httpx2.Response(200, json=[{"sha": f"{page}-{n}"} for n in range(count)])

    commits = client(handler).pull_commits("octo/demo", 7)
    assert len(commits) == 103 and pages == ["1", "2"]


def test_a_refusal_is_reported_without_the_token() -> None:
    """Check an error status raises, and the token isn't in the message."""
    github = client(lambda request: httpx2.Response(403, text="Resource not accessible"))
    with pytest.raises(GitHubError, match="403") as error:
        github.pull("octo/demo", 7)
    assert "t0ken" not in str(error.value)


def test_names_and_ids_that_could_change_the_path_are_refused() -> None:
    """Check bad repo names and commit IDs never reach GitHub."""
    github = client(lambda request: httpx2.Response(200, json={}))
    with pytest.raises(GitHubError, match="repo name"):
        github.pull("octo/demo/../x", 7)
    with pytest.raises(GitHubError, match="full commit ID"):
        github.commit("octo/demo", "HEAD")
