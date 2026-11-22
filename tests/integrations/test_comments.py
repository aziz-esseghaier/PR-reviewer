"""Tests for writing proven findings as review comments (integrations/comments.py).

Checks the escaping of model text, the fenced test, the diff lines comments can sit on, the
markers that stop repeats, and the review's layout.

How to run (from the project folder):
    pytest tests/integrations/test_comments.py
"""

from typing import Any

from pr_reviewer.core.schemas import Claim
from pr_reviewer.integrations.comments import (
    ZERO_WIDTH,
    ProvenFinding,
    anchor,
    comment_body,
    commentable_lines,
    escape_markdown,
    fenced,
    fingerprint,
    plan_review,
    posted_fingerprints,
    proven_finding,
)
from tests.integrations.fakes import BASE, HEAD, PATCH

TEST = (
    "from tinystats.stats import mean\n\n\ndef test_mean_of_nothing():\n"
    "    assert mean([]) == 0.0\n"
)


def claim(
    summary: str = "mean() divides by zero on empty input.", lines: tuple[int, int] = (8, 8)
) -> Claim:
    """Build a claim on src/tinystats/stats.py.

    Args:
        summary (str): Its summary.
        lines (tuple[int, int]): Its first and last line.

    Returns:
        Claim: The claim.
    """
    return Claim(
        file="src/tinystats/stats.py",
        start_line=lines[0],
        end_line=lines[1],
        category="F.4",
        summary=summary,
        details="The docstring promises 0.0 for no values.",
        expected_failure="ZeroDivisionError",
        confidence=0.9,
    )


def proven(
    summary: str = "mean() divides by zero on empty input.", proof: str = "executed"
) -> ProvenFinding:
    """Build a proven finding.

    Args:
        summary (str): Its claim's summary.
        proof (str): "executed" or "new_code".

    Returns:
        ProvenFinding: The finding.
    """
    return ProvenFinding(
        "f1", claim(summary), proof, "tests/test_stats.py", TEST, "ZeroDivisionError"
    )


def record(**changes: Any) -> dict[str, Any]:
    """Build an evidence record with one execution item.

    Args:
        **changes (Any): Fields of the execution item to replace.

    Returns:
        dict[str, Any]: The record, as load_records() gives it.
    """
    item = {
        "kind": "execution",
        "test_file": "tests/test_stats.py",
        "test_code": TEST,
        "base_result": "passed",
        "pr_result": "failed",
        "failure": "ZeroDivisionError",
        "proof": "executed",
        "deliberate": False,
    }
    return {"evidence": [{"kind": "tool"}, {**item, **changes}]}


def test_model_text_cant_mention_link_or_add_html() -> None:
    """Check mentions, issue links, addresses, Markdown links and HTML are all disarmed."""
    text = "Ping @everyone, see #12 and [x](https://evil.example) <img src=x> or www.evil.example"
    safe = escape_markdown(text, 300)
    assert "@everyone" not in safe and f"@{ZERO_WIDTH}everyone" in safe
    assert "#12" not in safe
    assert "https://" not in safe and "www.evil" not in safe
    assert "<img" not in safe and "&lt;img" in safe
    assert "[x](" not in safe and r"\[x\]" in safe


def test_model_text_stays_on_one_line_within_its_limit() -> None:
    """Check newlines can't start a new Markdown block, and long text is cut."""
    safe = escape_markdown("first\n# Heading\n> quote " + "word " * 200, 100)
    assert "\n" not in safe
    assert len(safe) < 160 and safe.endswith("…")


def test_the_test_is_fenced_longer_than_its_own_backticks() -> None:
    """Check backticks in the test can't close its code block."""
    code = 'x = "```"\n'
    block = fenced(code)
    assert block.startswith("````python\n") and block.endswith("\n````")


def test_commentable_lines_are_the_new_sides_lines_in_the_patch() -> None:
    """Check added and unchanged lines count, removed ones don't."""
    lines = commentable_lines(PATCH)
    assert lines == set(range(1, 9))
    assert commentable_lines("@@ -10,2 +20,2 @@\n-a\n+b\n c\n\\ No newline at end of file") == {
        20,
        21,
    }


def test_a_comment_sits_on_the_claims_line_or_the_nearest_diff_line() -> None:
    """Check the anchor: inside the claim's lines, near them, or none at all."""
    diff = {"src/tinystats/stats.py": {3, 4, 8}}
    assert anchor(claim(lines=(7, 8)), diff) == 8
    assert anchor(claim(lines=(12, 13)), diff) == 8
    assert anchor(claim(lines=(40, 41)), diff) is None
    assert anchor(claim(), {"other.py": {8}}) is None


def test_the_body_shows_the_problem_the_proof_and_the_test() -> None:
    """Check the comment's parts, and the marker for later runs."""
    body = comment_body(proven(), BASE, HEAD)
    assert "**Problem:** mean\\(\\) divides by zero on empty input." in body
    assert f"passes on the base commit `{BASE[:7]}`" in body
    assert f"fails on this pull request's commit `{HEAD[:7]}` with `ZeroDivisionError`" in body
    assert "<details><summary>Generated test (tests/test_stats.py)</summary>" in body
    assert TEST.strip() in body
    assert f"<!-- pr-reviewer:{fingerprint(claim())} -->" in body
    assert "never approves" in body


def test_a_new_code_proof_says_the_base_cant_run_it() -> None:
    """Check the wording for a bug in code the PR adds."""
    body = comment_body(proven(proof="new_code"), BASE, HEAD)
    assert "raised from a line the pull request adds" in body


def test_markers_are_read_back_and_stop_repeats() -> None:
    """Check a finding already posted isn't posted again."""
    body = comment_body(proven(), BASE, HEAD)
    already = posted_fingerprints(["unrelated", body])
    assert already == {fingerprint(claim())}
    plan = plan_review([proven()], {"src/tinystats/stats.py": {8}}, already, BASE, HEAD, 4)
    assert plan.comments == [] and plan.posted == [] and plan.skipped == 1


def test_the_review_has_inline_comments_and_puts_the_rest_in_its_summary() -> None:
    """Check one finding on a diff line and one off the diff."""
    off = ProvenFinding(
        "f2",
        claim("variance() breaks too.", (90, 91)),
        "executed",
        "tests/t.py",
        TEST,
        "ValueError",
    )
    plan = plan_review([proven(), off], {"src/tinystats/stats.py": {8}}, set(), BASE, HEAD, 5)
    assert [(c["path"], c["line"], c["side"]) for c in plan.comments] == [
        ("src/tinystats/stats.py", 8, "RIGHT")
    ]
    assert "found 2 problems proven by an executed test, out of 5" in plan.body
    assert "lines 90-91" in plan.body and "variance\\(\\) breaks too." in plan.body


def test_only_an_executed_proof_that_wasnt_deliberate_counts() -> None:
    """Check the proving test is read from the record, and other evidence is ignored."""
    found = proven_finding(claim(), "f1", record())
    assert found is not None and found.test_code == TEST and found.failure == "ZeroDivisionError"
    assert proven_finding(claim(), "f1", record(proof=None)) is None
    assert proven_finding(claim(), "f1", record(deliberate=True)) is None
    assert proven_finding(claim(), "f1", {"evidence": []}) is None
