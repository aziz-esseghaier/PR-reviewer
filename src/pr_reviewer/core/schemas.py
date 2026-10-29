"""Shared data shapes: review findings with their evidence, and environment recipes.

A Finding is a Claim plus the Evidence collected for it:
- Claim: what the reviewer model says is wrong, and where. It's the only shape the model fills
  in, so text in a PR can't talk the model into attaching proof.
- Evidence: made only by our own code. Strongest first: a test that passes on the base commit
  and fails on the PR with the predicted error (tier 1), a static tool flagging the claimed
  lines (tier 2), and reasoning that a verifier failed to refute (tier 3). A finding with none
  of these is tier 4 and gets dropped.

An EnvRecipe says how to install and test one repo over a period of time. It counts as verified
only when the repo's own tests passed with it, never on the setup agent's word.

render_review() turns findings into the plain-text review that SWR-Bench scores.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt
from collections.abc import Sequence
from enum import IntEnum
from pathlib import PurePosixPath
from textwrap import indent
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# SWR-Bench's change categories, as its judge uses them. E = evolvability, F = functional.
Category = Literal[
    "E.1",
    "E.1.1",
    "E.1.2",
    "E.2",
    "E.3",
    "E.3.1",
    "E.3.2",
    "F.1",
    "F.2",
    "F.3",
    "F.4",
    "F.5",
    "F.6",
]

CATEGORY_NAMES: dict[Category, str] = {
    "E.1": "Documentation",
    "E.1.1": "Documentation: text (comments, names)",
    "E.1.2": "Documentation: language features",
    "E.2": "Visual representation (layout, style)",
    "E.3": "Structure",
    "E.3.1": "Structure: organization",
    "E.3.2": "Structure: solution approach",
    "F.1": "Interface",
    "F.2": "Logic",
    "F.3": "Resource",
    "F.4": "Check",
    "F.5": "Support",
    "F.6": "Larger defects",
}

# An exception class name, plain or dotted: "AttributeError", "json.decoder.JSONDecodeError".
ExceptionName = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
]


def _check_repo_path(value: str) -> str:
    """Make sure a path stays inside the repo, and write it in one standard form.

    Paths come from model output and our tools open them later, so a path such as
    "../../.ssh/id_rsa" or "/etc/passwd" must never get through.

    Args:
        value (str): The path as written, such as "./src/pkg/io.py".

    Returns:
        str: The path with "/" between folders and no "./", such as "src/pkg/io.py".

    Raises:
        ValueError: If the path is empty, absolute, starts with "~", contains "\\" or ":",
            or climbs out of the repo with "..".
    """
    path = PurePosixPath(value)
    if (
        not value.strip()
        or str(path) == "."
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or any(char in value for char in ("\\", ":", "\x00"))
    ):
        raise ValueError("must be a file path inside the repo, such as src/pkg/module.py")
    return str(path)


# A file path relative to the repo root, checked by _check_repo_path.
RepoPath = Annotated[str, AfterValidator(_check_repo_path)]


class Claim(BaseModel):
    """One problem the reviewer model says the PR introduces, and where it is.

    This is everything the model writes for a finding. Keys the model adds on its own are
    dropped: a reply can't smuggle in evidence, and a stray key doesn't cost a retry.

    Attributes:
        file (RepoPath): File the problem is in, relative to the repo root.
        start_line (int): First line of the problem in the PR version of the file, from 1.
        end_line (int): Last line of the problem. Left out, it equals start_line.
        category (Category): SWR-Bench category, such as "F.2" for a logic error.
        summary (str): The problem in one sentence. The review text shows it.
        details (str): Why it's a problem and when it happens. Empty if the summary says it all.
        expected_failure (ExceptionName | None): The error a test would hit if the claim is
            true, such as "AttributeError", or "AssertionError" for a wrong result. None when
            no test could show it, as with naming, style or design problems.
        confidence (float): The model's own estimate, from 0 to 1, that the claim is right.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    file: RepoPath
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    category: Category
    summary: str = Field(min_length=1, max_length=300)
    details: str = Field(default="", max_length=2000)
    expected_failure: ExceptionName | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _fill_in_gaps(cls, data: Any) -> Any:
        """Accept two common shortcuts in model output before checking it.

        Args:
            data (Any): The raw reply, usually a dict parsed from JSON.

        Returns:
            Any: The reply with a missing end_line set to start_line, and an empty
            expected_failure turned into None.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("end_line") is None and "start_line" in data:
            data["end_line"] = data["start_line"]
        failure = data.get("expected_failure")
        if isinstance(failure, str) and not failure.strip():
            data["expected_failure"] = None
        return data

    @model_validator(mode="after")
    def _check_line_order(self) -> Self:
        """Make sure the line range isn't backwards.

        Returns:
            Self: The unchanged claim.

        Raises:
            ValueError: If end_line comes before start_line.
        """
        if self.end_line < self.start_line:
            raise ValueError("end_line comes before start_line")
        return self

    @property
    def is_functional(self) -> bool:
        """Whether the claim is about behavior (an F category) rather than evolvability (E).

        Returns:
            bool: True for F categories.
        """
        return self.category.startswith("F")


class Tier(IntEnum):
    """How strong a finding's proof is. Lower is stronger."""

    EXECUTED = 1  # a test passes on the base commit and fails on the PR
    STATIC_TOOL = 2  # a static analysis tool flags the claimed lines
    VERIFIED = 3  # reasoning that a verifier tried and failed to refute
    NONE = 4  # no evidence: the finding is dropped


def _same_exception(seen: str, expected: str) -> bool:
    """Compare two exception names by their last part, so module prefixes don't matter.

    Args:
        seen (str): The error the test hit, such as "json.decoder.JSONDecodeError".
        expected (str): The error the claim predicted, such as "JSONDecodeError".

    Returns:
        bool: True when the class names are the same.
    """
    return seen.rsplit(".", 1)[-1] == expected.rsplit(".", 1)[-1]


# How a test ended on one commit: it passed, it failed (an assertion or an exception inside
# the test), or it couldn't run at all (such as a collection error).
RunResult = Literal["passed", "failed", "error"]


class ExecutionEvidence(BaseModel):
    """Tier 1: a generated test, run on the base commit and on the PR commit.

    Attributes:
        kind (Literal["execution"]): Tells the evidence types apart when read back from JSON.
        test_file (RepoPath): Where the test was placed in the repo.
        test_code (str): The test's source code.
        base_result (RunResult): How the test ended on the base commit.
        pr_result (RunResult): How the test ended on the PR commit.
        failure (ExceptionName | None): The error behind the PR failure, such as
            "AttributeError". None when it didn't fail.
        log_path (str | None): File holding the full test output.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    tier: ClassVar[Tier] = Tier.EXECUTED

    kind: Literal["execution"] = "execution"
    test_file: RepoPath
    test_code: str = Field(min_length=1)
    base_result: RunResult
    pr_result: RunResult
    failure: ExceptionName | None = None
    log_path: str | None = None

    def supports(self, claim: Claim) -> bool:
        """Tell whether this test run proves the claim.

        It does only when the test passes on the base commit, fails on the PR commit, and fails
        with the error the claim predicted. An ImportError doesn't prove an AttributeError claim.

        Args:
            claim (Claim): The claim the test was written for.

        Returns:
            bool: True when the run proves the claim.
        """
        return (
            self.base_result == "passed"
            and self.pr_result == "failed"
            and self.failure is not None
            and claim.expected_failure is not None
            and _same_exception(self.failure, claim.expected_failure)
        )


class ToolEvidence(BaseModel):
    """Tier 2: a static analysis tool flagged a line the claim points at.

    Attributes:
        kind (Literal["tool"]): Tells the evidence types apart when read back from JSON.
        tool (Literal["ruff", "mypy", "bandit", "semgrep"]): The tool that raised the warning.
        rule (str): The tool's rule or error code, such as "B006" or "attr-defined".
        message (str): The tool's message.
        file (RepoPath): File the warning is in.
        line (int): Line the warning is on, from 1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    tier: ClassVar[Tier] = Tier.STATIC_TOOL

    kind: Literal["tool"] = "tool"
    tool: Literal["ruff", "mypy", "bandit", "semgrep"]
    rule: str = Field(min_length=1)
    message: str
    file: RepoPath
    line: int = Field(ge=1)

    def supports(self, claim: Claim) -> bool:
        """Tell whether the warning lands on the claimed lines.

        Args:
            claim (Claim): The claim to check against.

        Returns:
            bool: True when the warning is in the claim's file and line range.
        """
        return self.file == claim.file and claim.start_line <= self.line <= claim.end_line


class VerifierEvidence(BaseModel):
    """Tier 3: the reasoning behind a claim, checked by a model call told to refute it.

    Attributes:
        kind (Literal["verifier"]): Tells the evidence types apart when read back from JSON.
        reasoning (str): Step-by-step trace from the changed code to the problem.
        verdict (Literal["upheld", "refuted"]): Whether the claim survived the verifier.
        argument (str): The verifier's reason for its verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    tier: ClassVar[Tier] = Tier.VERIFIED

    kind: Literal["verifier"] = "verifier"
    reasoning: str = Field(min_length=1)
    verdict: Literal["upheld", "refuted"]
    argument: str

    def supports(self, claim: Claim) -> bool:
        """Tell whether the claim survived the verifier.

        Args:
            claim (Claim): The claim that was checked.

        Returns:
            bool: True when the verdict is "upheld".
        """
        return self.verdict == "upheld"


# Any one piece of evidence. Pydantic reads "kind" to pick the right class when loading JSON.
Evidence = Annotated[
    ExecutionEvidence | ToolEvidence | VerifierEvidence, Field(discriminator="kind")
]


class Finding(BaseModel):
    """A claim plus the evidence collected for it.

    Failed attempts, such as a test that didn't fail or a refuted claim, stay in the evidence
    list, so reports can show how often proof failed and why.

    Attributes:
        id (str): Short ID, unique within one PR's review, such as "c3".
        claim (Claim): What the reviewer model said.
        evidence (tuple[Evidence, ...]): Evidence collected so far, in the order it was made.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    claim: Claim
    evidence: tuple[Evidence, ...] = ()

    @property
    def tier(self) -> Tier:
        """The strongest tier among the evidence that supports the claim.

        Returns:
            Tier: That tier, or Tier.NONE when no evidence supports the claim.
        """
        supporting = (item.tier for item in self.evidence if item.supports(self.claim))
        return min(supporting, default=Tier.NONE)

    def with_evidence(self, item: Evidence) -> Self:
        """Return a copy of this finding with one more piece of evidence.

        Args:
            item (Evidence): The new evidence.

        Returns:
            Self: The new finding. This one is left unchanged.
        """
        return self.model_copy(update={"evidence": (*self.evidence, item)})


# The whole review when there's nothing to report. SWR-Bench counts it as "no issues".
NO_ISSUES = "No issues found. The changes look correct."


def render_review(findings: Sequence[Finding]) -> str:
    """Write findings as the plain-text review that SWR-Bench scores.

    Each finding becomes one numbered point, so the judge counts one point per finding.
    Categories and evidence are left out, so the judge classifies each point on its own.
    The caller picks which findings to pass: all of them for an ungated run, only the ones
    that pass the gate for a gated run.

    Args:
        findings (Sequence[Finding]): The findings to report, in order.

    Returns:
        str: The review text, or NO_ISSUES when the list is empty.
    """
    if not findings:
        return NO_ISSUES
    points = []
    for number, finding in enumerate(findings, start=1):
        claim = finding.claim
        if claim.start_line == claim.end_line:
            where = f"{claim.file}, line {claim.start_line}"
        else:
            where = f"{claim.file}, lines {claim.start_line}-{claim.end_line}"
        point = f"{number}. {where}: {claim.summary}"
        if claim.details:
            point += "\n" + indent(claim.details, "   ")
        points.append(point)
    return "\n\n".join(points)


# Share of the repo's tests that must pass for a setup check to count.
# Tune it once the setup agent runs on the real repos.
MIN_PASS_RATE = 0.8

# A name that is safe to paste into a shell command: a Debian package or an environment variable.
PackageName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9+.-]*$")]
EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class SetupCheck(BaseModel):
    """One run of the repo's own tests in an environment built from a recipe.

    Attributes:
        commit (str): SHA of the commit the tests ran at.
        commit_date (dt.date): When that commit was made. Tells which period a recipe covers.
        passed (int): Tests that passed.
        failed (int): Tests that failed.
        errors (int): Tests that couldn't run, such as collection errors.
        log_path (str | None): File holding the full test output.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    commit_date: dt.date
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    log_path: str | None = None

    @property
    def ok(self) -> bool:
        """Whether enough tests passed for the environment to count as working.

        Returns:
            bool: True when at least one test passed and at least MIN_PASS_RATE of all did.
        """
        total = self.passed + self.failed + self.errors
        return self.passed > 0 and self.passed >= MIN_PASS_RATE * total


class EnvRecipe(BaseModel):
    """How to build a working environment for one repo over a period of time.

    The setup agent writes it and the sandbox runs it. Its commands come from a model, so they
    only ever run inside a container.

    Attributes:
        repo (str): GitHub repo, such as "psf/requests".
        python_version (str): Python version, such as "3.11".
        base_image (str): Docker image to start from, such as "python:3.11-slim".
        system_packages (tuple[PackageName, ...]): Debian packages to install first.
        install_steps (tuple[str, ...]): Shell commands that install the repo and its test
            dependencies, run in order at the repo root with the network on.
        test_command (str): Shell command that runs the tests with the network off, such as
            "python -m pytest -q". Test paths are added at the end.
        env (dict[EnvName, str]): Extra environment variables for install and test.
        checks (tuple[SetupCheck, ...]): Test runs made with this recipe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    python_version: str = Field(pattern=r"^3\.\d{1,2}$")
    base_image: str = Field(pattern=r"^[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")
    system_packages: tuple[PackageName, ...] = ()
    install_steps: tuple[str, ...] = Field(min_length=1)
    test_command: str = Field(min_length=1)
    env: dict[EnvName, str] = Field(default_factory=dict)
    checks: tuple[SetupCheck, ...] = ()

    @property
    def verified(self) -> bool:
        """Whether the repo's tests passed with this recipe at least once.

        Returns:
            bool: True when any check is ok.
        """
        return any(check.ok for check in self.checks)

    @property
    def period(self) -> tuple[dt.date, dt.date] | None:
        """The dates of the oldest and newest commits where the tests passed.

        Returns:
            tuple[dt.date, dt.date] | None: First and last date, or None if no check is ok.
        """
        dates = [check.commit_date for check in self.checks if check.ok]
        return (min(dates), max(dates)) if dates else None
