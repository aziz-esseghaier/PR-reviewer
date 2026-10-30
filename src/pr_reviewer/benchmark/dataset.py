"""The SWR-Bench PRs: read the dataset file, check every record, and pick the benchmark's PRs.

Each line of the file is one PR, which SWR-Bench calls an instance: its repo, title and
description, the commits to review, and the issues the PR's real reviewers raised. SWR-Bench
calls these issues "changes". Clean PRs have none. Every record is checked with pydantic, so
a surprise in the data fails loudly instead of skewing results. Fields not used yet, such as
the PR's discussion timeline and the commits after the reviewed ones, are skipped.

The benchmark uses PRs from 2019 onward that are clean or have at least one functional (F.*)
issue. PRs whose issues are all evolvability (E.*) ones are left out.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from pr_reviewer.benchmark.source import DATASET, DatasetError, sha256_of
from pr_reviewer.core.schemas import Category

# The first year of PRs the benchmark uses.
FIRST_YEAR = 2019

# A full git commit SHA.
Sha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]

# How the benchmark sorts a PR: at least one functional issue, only evolvability issues, or none.
PRGroup = Literal["functional", "evolvability", "clean"]

_CATEGORIES = frozenset(get_args(Category))


class FilePatch(BaseModel):
    """One file's changes in a commit, as a unified diff.

    Attributes:
        file (str): Path of the file in the repo.
        patch (str): The diff of that file.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    file: str
    patch: str


class Commit(BaseModel):
    """One commit of the PR.

    Attributes:
        sha (Sha): The commit's SHA.
        message (str): The commit message.
        diff (tuple[FilePatch, ...]): The changes, one entry per file.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    sha: Sha
    message: str
    diff: tuple[FilePatch, ...]


class IssueOrigin(BaseModel):
    """Where a known issue came in.

    Attributes:
        commit_sha (Sha): The PR commit that introduced the issue.
        code_snippet (str): The code with the issue.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    commit_sha: Sha
    code_snippet: str


class IssueDiscussion(BaseModel):
    """What the PR's reviewers said about a known issue.

    Attributes:
        discussion_summary (str): SWR-Bench's summary of the issue. The judge compares
            review points with it.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    discussion_summary: str


class IssueFix(BaseModel):
    """How a known issue was fixed later in the PR.

    Attributes:
        commit_sha (Sha): The commit that fixed it. It comes after the reviewed commits.
        resolution_explanation (str): How the fix resolved the issue.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    commit_sha: Sha
    resolution_explanation: str


class KnownIssue(BaseModel):
    """One issue the PR's real reviewers raised. SWR-Bench calls it a "change".

    Attributes:
        change_type (str): Category code and name, such as "F.2 Logic".
        change_introducing (IssueOrigin): The commit and code that brought the issue in.
        change_discussion (IssueDiscussion): What the reviewers said about it.
        change_resolve_info (IssueFix | None): How it was fixed. None when SWR-Bench has no
            fix for it.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    change_type: str
    change_introducing: IssueOrigin
    change_discussion: IssueDiscussion
    change_resolve_info: IssueFix | None

    @field_validator("change_type")
    @classmethod
    def _check_category(cls, value: str) -> str:
        """Make sure the category code is one of SWR-Bench's.

        Args:
            value (str): The change type as written, such as "F.2 Logic".

        Returns:
            str: The unchanged value.

        Raises:
            ValueError: If the first word isn't a known category code.
        """
        if not value.split() or value.split()[0] not in _CATEGORIES:
            raise ValueError("must start with a category code, such as F.2 or E.3.1")
        return value

    @property
    def category(self) -> Category:
        """The category code, such as "F.2"."""
        return cast(Category, self.change_type.split()[0])

    @property
    def is_functional(self) -> bool:
        """Whether the issue is about behavior (F) rather than evolvability (E)."""
        return self.category.startswith("F")


class Instance(BaseModel):
    """One SWR-Bench PR: what to review, and the known issues the review is scored against.

    Attributes:
        instance_id (str): Unique ID, such as "psf__requests-2317".
        repo (str): GitHub repo, such as "psf/requests".
        pr_title (str): The PR's title.
        pr_statement (str): The PR's description.
        change_introduced (bool): Whether the PR has known issues.
        base_commit (Sha): The commit the PR starts from.
        created_at (dt.datetime): When the PR was opened.
        changes (tuple[KnownIssue, ...]): The known issues. Empty for a clean PR.
        pr_commits (tuple[Commit, ...]): The commits to review, oldest first.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    instance_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+$")
    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    pr_title: str
    pr_statement: str
    change_introduced: bool
    base_commit: Sha
    created_at: dt.datetime
    changes: tuple[KnownIssue, ...]
    pr_commits: tuple[Commit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_issue_flag(self) -> Self:
        """Make sure the PR is marked as having issues exactly when it lists some.

        Returns:
            Self: The unchanged instance.

        Raises:
            ValueError: If change_introduced disagrees with the list of changes.
        """
        if self.change_introduced != bool(self.changes):
            raise ValueError("change_introduced doesn't match the list of changes")
        return self

    @property
    def year(self) -> int:
        """The year the PR was opened, in UTC."""
        return self.created_at.astimezone(dt.UTC).year

    @property
    def has_functional_issue(self) -> bool:
        """Whether at least one known issue is functional (F)."""
        return any(issue.is_functional for issue in self.changes)

    @property
    def group(self) -> PRGroup:
        """How the benchmark sorts the PR.

        "functional" when at least one known issue is functional, "evolvability" when all of
        them are evolvability issues, and "clean" when it has none.
        """
        if not self.changes:
            return "clean"
        return "functional" if self.has_functional_issue else "evolvability"


def load_dataset(path: Path, sha256: str | None = DATASET.sha256) -> list[Instance]:
    """Read the SWR-Bench file and check every record.

    Args:
        path (Path): The dataset file.
        sha256 (str | None): Fingerprint the file must have. None skips the check, for small
            test files.

    Returns:
        list[Instance]: The PRs, in file order.

    Raises:
        DatasetError: If the file is missing, isn't the pinned version, has a record that
            doesn't fit the expected shape, or repeats an instance ID.
    """
    if not path.is_file():
        raise DatasetError(f"{path} not found. Download it with: pr-reviewer bench download")
    if sha256 is not None and sha256_of(path) != sha256:
        raise DatasetError(
            f"{path} isn't the pinned SWR-Bench version (its fingerprint differs). "
            "Download it again with: pr-reviewer bench download"
        )
    instances: list[Instance] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                instance = Instance.model_validate_json(line)
            except ValidationError as exc:
                error = exc.errors()[0]
                where = ".".join(str(part) for part in error["loc"]) or "(record)"
                raise DatasetError(f"{path.name}, line {number}: {where}: {error['msg']}") from None
            if instance.instance_id in seen:
                raise DatasetError(f"{path.name}, line {number}: {instance.instance_id} repeats")
            seen.add(instance.instance_id)
            instances.append(instance)
    return instances


def is_benchmark_pr(instance: Instance) -> bool:
    """Tell whether the benchmark uses a PR: from FIRST_YEAR onward, and clean or functional.

    Args:
        instance (Instance): The PR.

    Returns:
        bool: True when the benchmark uses it.
    """
    return instance.year >= FIRST_YEAR and instance.group != "evolvability"


def select_prs(instances: Sequence[Instance]) -> list[Instance]:
    """Pick the PRs the benchmark uses, keeping their order.

    Args:
        instances (Sequence[Instance]): All PRs from the dataset.

    Returns:
        list[Instance]: The PRs for which is_benchmark_pr() is true.
    """
    return [instance for instance in instances if is_benchmark_pr(instance)]
