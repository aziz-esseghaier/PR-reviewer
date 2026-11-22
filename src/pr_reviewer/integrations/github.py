"""A small GitHub REST client: read a pull request, and post review comments on it.

It sends every review with the event COMMENT, fixed in code, so the reviewer can never
approve a pull request or ask for changes, and it has no call that merges. The token comes
from the GitHub Action (its GITHUB_TOKEN), and is only ever sent to api.github.com.

Imported by other modules; the file isn't run directly.
"""

import re
from typing import Any

import httpx2

API = "https://api.github.com"

# The only review event this client sends: a plain comment. Never APPROVE or REQUEST_CHANGES.
REVIEW_EVENT = "COMMENT"

# Items per page GitHub allows, and the most pages read from one list.
PER_PAGE = 100
MAX_PAGES = 10

# Seconds one request may take.
TIMEOUT_S = 30.0

# A repo name as GitHub allows it: owner/name.
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubError(Exception):
    """Raised when GitHub refuses a request or answers with something unexpected."""


def check_repo(repo: str) -> str:
    """Make sure a repo name is owner/name, so it can't change the request's path.

    Args:
        repo (str): Such as "octo/demo".

    Returns:
        str: The same name.

    Raises:
        GitHubError: If it isn't a plain owner/name.
    """
    if not REPO_PATTERN.match(repo) or ".." in repo:
        raise GitHubError(f"Not a GitHub repo name: {repo!r}")
    return repo


class GitHub:
    """Reads pull requests and posts review comments through GitHub's REST API.

    Args:
        token (str): The token, such as the Action's GITHUB_TOKEN. Never printed.
        http (httpx2.Client | None): The HTTP client. Tests pass one with a fake transport.
        api (str): The API's address.
    """

    def __init__(self, token: str, http: httpx2.Client | None = None, api: str = API) -> None:
        self._api = api.rstrip("/")
        self._http = http or httpx2.Client(timeout=TIMEOUT_S)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pr-reviewer",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx2.Response:
        """Send one request and check its status.

        Args:
            method (str): "GET" or "POST".
            path (str): The path under the API, such as "/repos/o/r/pulls/1".
            **kwargs (Any): Passed to the HTTP client, such as params or json.

        Returns:
            httpx2.Response: The answer.

        Raises:
            GitHubError: If the request fails or GitHub refuses it.
        """
        try:
            response = self._http.request(
                method, f"{self._api}{path}", headers=self._headers, **kwargs
            )
        except httpx2.HTTPError as exc:
            raise GitHubError(f"{method} {path} failed: {exc}") from None
        if response.status_code >= 400:
            message = response.text[:300].replace("\n", " ")
            raise GitHubError(f"{method} {path}: GitHub answered {response.status_code}: {message}")
        return response

    def _get(self, path: str) -> Any:
        """Read one object.

        Args:
            path (str): The path under the API.

        Returns:
            Any: The parsed JSON.
        """
        return self._request("GET", path).json()

    def _list(self, path: str) -> list[Any]:
        """Read a list, page by page.

        Args:
            path (str): The path under the API.

        Returns:
            list[Any]: Every item, up to MAX_PAGES pages.
        """
        items: list[Any] = []
        for page in range(1, MAX_PAGES + 1):
            batch = self._request("GET", path, params={"per_page": PER_PAGE, "page": page}).json()
            if not isinstance(batch, list):
                raise GitHubError(f"GET {path} didn't answer with a list")
            items += batch
            if len(batch) < PER_PAGE:
                break
        return items

    def pull(self, repo: str, number: int) -> dict[str, Any]:
        """Read a pull request.

        Args:
            repo (str): The repo, such as "octo/demo".
            number (int): The pull request's number.

        Returns:
            dict[str, Any]: GitHub's pull request object.
        """
        found: dict[str, Any] = self._get(f"/repos/{check_repo(repo)}/pulls/{int(number)}")
        return found

    def pull_commits(self, repo: str, number: int) -> list[dict[str, Any]]:
        """List a pull request's commits, oldest first.

        Args:
            repo (str): The repo.
            number (int): The pull request's number.

        Returns:
            list[dict[str, Any]]: GitHub's commit objects (at most 250, GitHub's limit).
        """
        return self._list(f"/repos/{check_repo(repo)}/pulls/{int(number)}/commits")

    def pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        """List the files a pull request changes, with each file's patch against its base.

        Args:
            repo (str): The repo.
            number (int): The pull request's number.

        Returns:
            list[dict[str, Any]]: GitHub's file objects.
        """
        return self._list(f"/repos/{check_repo(repo)}/pulls/{int(number)}/files")

    def commit(self, repo: str, sha: str) -> dict[str, Any]:
        """Read one commit, with the files it changed and their patches.

        Args:
            repo (str): The repo.
            sha (str): The commit's full ID.

        Returns:
            dict[str, Any]: GitHub's commit object.

        Raises:
            GitHubError: If the ID isn't a full commit ID.
        """
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise GitHubError(f"Not a full commit ID: {sha!r}")
        found: dict[str, Any] = self._get(f"/repos/{check_repo(repo)}/commits/{sha}")
        return found

    def review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """List the inline review comments already on a pull request.

        Args:
            repo (str): The repo.
            number (int): The pull request's number.

        Returns:
            list[dict[str, Any]]: GitHub's review comment objects.
        """
        return self._list(f"/repos/{check_repo(repo)}/pulls/{int(number)}/comments")

    def post_review(
        self,
        repo: str,
        number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Post a review made only of comments. It never approves or asks for changes.

        Args:
            repo (str): The repo.
            number (int): The pull request's number.
            commit_id (str): The commit the comments are about (the reviewed head).
            body (str): The review's summary text.
            comments (list[dict[str, Any]]): The inline comments: path, line, side and body.

        Returns:
            dict[str, Any]: GitHub's review object.
        """
        payload = {
            "commit_id": commit_id,
            "body": body,
            "event": REVIEW_EVENT,
            "comments": comments,
        }
        response = self._request(
            "POST", f"/repos/{check_repo(repo)}/pulls/{int(number)}/reviews", json=payload
        )
        posted: dict[str, Any] = response.json()
        return posted
