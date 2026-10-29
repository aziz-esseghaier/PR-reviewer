"""Tests for the shared data shapes in core/schemas.py.

Checks that model output is cleaned up or rejected as it should be (including paths that would
leave the repo), that a finding's tier comes only from evidence that supports its claim, that
everything survives a trip through JSON, what the review text for SWR-Bench looks like, and
when an environment recipe counts as verified.

How to run (from the project folder):
    pytest tests/core/test_schemas.py
"""

import datetime as dt
from typing import Any, Literal, get_args

import pytest
from pydantic import ValidationError

from pr_reviewer.core.schemas import (
    CATEGORY_NAMES,
    NO_ISSUES,
    Category,
    Claim,
    EnvRecipe,
    Evidence,
    ExecutionEvidence,
    Finding,
    RunResult,
    SetupCheck,
    Tier,
    ToolEvidence,
    VerifierEvidence,
    render_review,
)

# A typical claim as the reviewer model would write it, parsed from JSON.
CLAIM: dict[str, Any] = {
    "file": "src/pkg/io.py",
    "start_line": 10,
    "end_line": 12,
    "category": "F.2",
    "summary": "read_config returns None when the file is empty.",
    "expected_failure": "AttributeError",
    "confidence": 0.7,
}

# A typical recipe as the setup agent would write it.
RECIPE: dict[str, Any] = {
    "repo": "psf/requests",
    "python_version": "3.11",
    "base_image": "python:3.11-slim",
    "system_packages": ["git"],
    "install_steps": ["pip install -e .", "pip install pytest"],
    "test_command": "python -m pytest -q",
}


def make_claim(**changes: Any) -> Claim:
    """Build a claim from the typical model output, with some keys changed.

    Args:
        **changes (Any): Keys to replace or add.

    Returns:
        Claim: The checked claim.
    """
    return Claim.model_validate({**CLAIM, **changes})


def executed(
    base: RunResult = "passed", pr: RunResult = "failed", failure: str | None = "AttributeError"
) -> ExecutionEvidence:
    """Build a test-run result for the default claim.

    Args:
        base (RunResult): How the test ended on the base commit.
        pr (RunResult): How the test ended on the PR commit.
        failure (str | None): The error behind the PR failure.

    Returns:
        ExecutionEvidence: The test-run evidence.
    """
    return ExecutionEvidence(
        test_file="tests/test_io_empty.py",
        test_code="def test_empty_file(tmp_path): ...",
        base_result=base,
        pr_result=pr,
        failure=failure,
    )


def flagged(file: str = "src/pkg/io.py", line: int = 11) -> ToolEvidence:
    """Build a mypy warning at a given place.

    Args:
        file (str): File the warning is in.
        line (int): Line the warning is on.

    Returns:
        ToolEvidence: The tool evidence.
    """
    return ToolEvidence(
        tool="mypy", rule="union-attr", message="Item None has no attribute", file=file, line=line
    )


def verified(verdict: Literal["upheld", "refuted"]) -> VerifierEvidence:
    """Build a verifier result.

    Args:
        verdict (Literal["upheld", "refuted"]): The verifier's decision.

    Returns:
        VerifierEvidence: The verifier evidence.
    """
    return VerifierEvidence(
        reasoning="safe_load returns None for an empty file; line 11 calls .get on it.",
        verdict=verdict,
        argument="No caller checks for None first.",
    )


def check(passed: int = 95, failed: int = 5, date: dt.date = dt.date(2023, 5, 1)) -> SetupCheck:
    """Build a setup check with the given test counts.

    Args:
        passed (int): Tests that passed.
        failed (int): Tests that failed.
        date (dt.date): Date of the commit the tests ran at.

    Returns:
        SetupCheck: The check.
    """
    return SetupCheck(commit="a1b2c3d", commit_date=date, passed=passed, failed=failed, errors=0)


def test_claim_cleans_up_typical_model_output() -> None:
    """Check that stray keys are dropped, end_line is filled in and spaces are trimmed."""
    claim = Claim.model_validate(
        {
            **CLAIM,
            "end_line": None,
            "severity": "high",
            "summary": "  Off by one.  ",
            "expected_failure": "",
        }
    )
    assert claim.end_line == claim.start_line == 10
    assert claim.summary == "Off by one."
    assert claim.expected_failure is None
    assert "severity" not in claim.model_dump()


@pytest.mark.parametrize(
    ("written", "kept"),
    [("F.2 Logic", "F.2"), ("F.4: Check", "F.4"), ("E.1.1 Documentation: text", "E.1.1")],
)
def test_claim_keeps_only_the_category_code(written: str, kept: str) -> None:
    """Check that a category written with its name, as models often do, keeps just the code."""
    assert make_claim(category=written).category == kept


def test_claim_cannot_carry_evidence() -> None:
    """Check that proof written by the model is thrown away, so the claim stays unproven."""
    fake_proof = executed().model_dump()
    claim = make_claim(evidence=[fake_proof], tier=1)
    assert "evidence" not in claim.model_dump()
    assert Finding(id="c1", claim=claim).tier is Tier.NONE


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("category", "F.9"),
        ("confidence", 1.5),
        ("start_line", 0),
        ("end_line", 5),  # before start_line
        ("summary", "   "),
        ("expected_failure", "it crashes"),
    ],
)
def test_claim_rejects_bad_model_output(key: str, value: Any) -> None:
    """Check that values outside the allowed shape fail, so the caller can ask again."""
    with pytest.raises(ValidationError):
        make_claim(**{key: value})


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../secrets.py",
        "src/../../x.py",
        "C:/x.py",
        "src\\pkg\\io.py",
        "~/x",
        "",
        ".",
    ],
)
def test_claim_rejects_paths_outside_the_repo(path: str) -> None:
    """Check that absolute paths, Windows paths and paths climbing out with '..' are refused."""
    with pytest.raises(ValidationError, match="inside the repo"):
        make_claim(file=path)


def test_claim_paths_are_written_one_way() -> None:
    """Check that './' and doubled slashes are removed, so the same file always looks the same."""
    assert make_claim(file="./src//pkg/io.py").file == "src/pkg/io.py"


def test_category_names_cover_every_category() -> None:
    """Check that each SWR-Bench category has a name and that F means functional."""
    assert set(CATEGORY_NAMES) == set(get_args(Category))
    assert make_claim(category="F.4").is_functional
    assert not make_claim(category="E.3.1").is_functional


@pytest.mark.parametrize(
    ("evidence", "tier"),
    [
        ((), Tier.NONE),
        ((executed(),), Tier.EXECUTED),
        ((executed(failure="ImportError"),), Tier.NONE),
        ((executed(base="failed"),), Tier.NONE),
        ((executed(pr="error", failure=None),), Tier.NONE),
        ((flagged(line=11),), Tier.STATIC_TOOL),
        ((flagged(line=40),), Tier.NONE),
        ((flagged(file="src/pkg/other.py"),), Tier.NONE),
        ((verified("upheld"),), Tier.VERIFIED),
        ((verified("refuted"),), Tier.NONE),
        ((verified("refuted"), flagged(), executed()), Tier.EXECUTED),
    ],
    ids=[
        "no evidence",
        "passes on base, fails on PR with the predicted error",
        "fails with a different error",
        "already failing on base",
        "can't run on PR",
        "tool flags a claimed line",
        "tool flags another line",
        "tool flags another file",
        "verifier upholds",
        "verifier refutes",
        "strongest evidence wins",
    ],
)
def test_tier_comes_from_supporting_evidence(evidence: tuple[Evidence, ...], tier: Tier) -> None:
    """Check that only evidence that really backs the claim sets the tier."""
    assert Finding(id="c1", claim=make_claim(), evidence=evidence).tier is tier


def test_error_names_match_without_module_prefix() -> None:
    """Check that 'json.decoder.JSONDecodeError' matches a predicted 'JSONDecodeError'."""
    claim = make_claim(expected_failure="JSONDecodeError")
    evidence = (executed(failure="json.decoder.JSONDecodeError"),)
    assert Finding(id="c1", claim=claim, evidence=evidence).tier is Tier.EXECUTED


def test_claim_without_expected_error_cannot_be_proven_by_a_test() -> None:
    """Check that a failing test doesn't prove a claim that said no test could show it."""
    finding = Finding(id="c1", claim=make_claim(expected_failure=None), evidence=(executed(),))
    assert finding.tier is Tier.NONE


def test_with_evidence_leaves_the_original_unchanged() -> None:
    """Check that adding evidence returns a new finding and keeps the old one as it was."""
    finding = Finding(id="c1", claim=make_claim())
    proven = finding.with_evidence(executed())
    assert finding.evidence == ()
    assert finding.tier is Tier.NONE
    assert proven.tier is Tier.EXECUTED


def test_finding_survives_a_json_round_trip() -> None:
    """Check that each evidence type comes back as the right class after saving to JSON."""
    finding = Finding(
        id="c1", claim=make_claim(), evidence=(verified("refuted"), flagged(), executed())
    )
    restored = Finding.model_validate_json(finding.model_dump_json())
    assert restored == finding
    assert [type(item) for item in restored.evidence] == [
        VerifierEvidence,
        ToolEvidence,
        ExecutionEvidence,
    ]


def test_unknown_evidence_kind_is_rejected() -> None:
    """Check that saved data with a made-up evidence type doesn't load."""
    data = Finding(id="c1", claim=make_claim()).model_dump()
    data["evidence"] = [{"kind": "hunch", "note": "trust me"}]
    with pytest.raises(ValidationError):
        Finding.model_validate(data)


def test_review_without_findings_says_no_issues() -> None:
    """Check that an empty review says so plainly, which SWR-Bench counts as 'no issues'."""
    assert render_review([]) == NO_ISSUES


def test_review_has_one_numbered_point_per_finding() -> None:
    """Check the review layout: one point per finding, details indented, no category codes."""
    one_line = make_claim(start_line=7, end_line=7, summary="Off by one.")
    with_details = make_claim(details="When the file is empty,\nsafe_load returns None.")
    text = render_review([Finding(id="c1", claim=one_line), Finding(id="c2", claim=with_details)])
    assert text == (
        "1. src/pkg/io.py, line 7: Off by one.\n"
        "\n"
        "2. src/pkg/io.py, lines 10-12: read_config returns None when the file is empty.\n"
        "   When the file is empty,\n"
        "   safe_load returns None."
    )
    assert "F.2" not in text


def test_recipe_is_verified_only_by_passing_tests() -> None:
    """Check that a recipe without checks, or whose tests mostly fail, isn't verified."""
    assert not EnvRecipe.model_validate(RECIPE).verified
    weak = EnvRecipe.model_validate({**RECIPE, "checks": [check(passed=50, failed=50)]})
    assert not weak.verified
    empty = EnvRecipe.model_validate({**RECIPE, "checks": [check(passed=0, failed=0)]})
    assert not empty.verified
    good = EnvRecipe.model_validate({**RECIPE, "checks": [check()]})
    assert good.verified


def test_recipe_period_spans_the_passing_checks() -> None:
    """Check that the period runs from the oldest to the newest commit whose tests passed."""
    recipe = EnvRecipe.model_validate(
        {
            **RECIPE,
            "checks": [
                check(date=dt.date(2023, 5, 1)),
                check(date=dt.date(2022, 1, 10)),
                check(passed=10, failed=90, date=dt.date(2021, 1, 1)),  # didn't pass
            ],
        }
    )
    assert recipe.period == (dt.date(2022, 1, 10), dt.date(2023, 5, 1))
    assert EnvRecipe.model_validate(RECIPE).period is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("system_packages", ["curl; rm -rf /"]),
        ("env", {"BAD NAME": "x"}),
        ("base_image", "python:3.11 && curl evil.example"),
        ("repo", "requests"),
        ("python_version", "python3"),
        ("install_steps", []),
    ],
)
def test_recipe_rejects_bad_values(key: str, value: Any) -> None:
    """Check that values pasted into shell commands can't carry extra commands."""
    with pytest.raises(ValidationError):
        EnvRecipe.model_validate({**RECIPE, key: value})


def test_recipe_survives_a_json_round_trip() -> None:
    """Check that a recipe with checks and env vars loads back the same from JSON."""
    recipe = EnvRecipe.model_validate({**RECIPE, "env": {"CI": "1"}, "checks": [check()]})
    assert EnvRecipe.model_validate_json(recipe.model_dump_json()) == recipe
