"""Tests for reading the static tools' output (static/warnings.py).

Feeds each parser output shaped like its tool's JSON (ruff, mypy, bandit, semgrep) and checks
the warnings: paths made relative to the repo, paths outside it dropped, long messages cut,
and only warnings that touch a changed line kept.

How to run (from the project folder):
    pytest tests/static/test_warnings.py
"""

import json

import pytest

from pr_reviewer.core.schemas import Claim
from pr_reviewer.static.warnings import (
    MESSAGE_CHARS,
    ToolWarning,
    on_changed_lines,
    parse_bandit,
    parse_mypy,
    parse_ruff,
    parse_semgrep,
    repo_relative,
    semgrep_rule,
)

ROOT = "/work/psf__requests"


def warning(line: int = 3, end_line: int | None = None, **changes: object) -> ToolWarning:
    """Make one warning.

    Args:
        line (int): Its first line.
        end_line (int | None): Its last line; None for the first.
        **changes (object): Fields to change.

    Returns:
        ToolWarning: The warning.
    """
    fields: dict[str, object] = {
        "tool": "mypy",
        "rule": "attr-defined",
        "message": '"Cart" has no attribute "totl"',
        "file": "pkg/cart.py",
        "line": line,
        "end_line": end_line,
        **changes,
    }
    return ToolWarning.model_validate(fields)


def test_paths_are_made_relative_and_kept_inside_the_repo() -> None:
    """Check absolute paths in the repo become relative, and others are refused."""
    assert repo_relative(f"{ROOT}/pkg/a.py", ROOT) == "pkg/a.py"
    assert repo_relative("pkg/./a.py", ROOT) == "pkg/a.py"
    assert repo_relative("/etc/passwd", ROOT) is None
    assert repo_relative("../other/a.py", ROOT) is None
    assert repo_relative(f"{ROOT}x/a.py", ROOT) is None


def test_ruff_output_is_read() -> None:
    """Check ruff's diagnostics, with a syntax error that has no code."""
    output = [
        {
            "code": "F821",
            "message": "Undefined name `totl`",
            "filename": f"{ROOT}/pkg/cart.py",
            "location": {"row": 7, "column": 12},
            "end_location": {"row": 7, "column": 16},
        },
        {
            "code": None,
            "message": "SyntaxError: Expected ')'",
            "filename": f"{ROOT}/pkg/broken.py",
            "location": {"row": 2, "column": 1},
            "end_location": {"row": 2, "column": 2},
        },
        {
            "code": "B006",
            "message": "outside",
            "filename": "/tmp/x.py",
            "location": {"row": 1},
        },
    ]
    found = parse_ruff(json.dumps(output), ROOT)
    assert [(w.rule, w.file, w.line, w.severity) for w in found] == [
        ("F821", "pkg/cart.py", 7, "warning"),
        ("syntax-error", "pkg/broken.py", 2, "error"),
    ]
    assert parse_ruff("", ROOT) == []
    with pytest.raises(ValueError):
        parse_ruff("{}", ROOT)


def test_mypy_output_is_read_without_notes() -> None:
    """Check mypy's JSON lines: errors kept with their hint, notes left out."""
    lines = [
        {
            "file": "pkg/cart.py",
            "line": 9,
            "column": 4,
            "message": '"Cart" has no attribute "totl"',
            "hint": 'Did you mean "total"?',
            "code": "attr-defined",
            "severity": "error",
        },
        {
            "file": "pkg/cart.py",
            "line": 9,
            "column": 4,
            "message": "See the docs",
            "hint": None,
            "code": None,
            "severity": "note",
        },
    ]
    found = parse_mypy("\n".join(json.dumps(line) for line in lines) + "\n", ROOT)
    assert len(found) == 1
    assert found[0].rule == "attr-defined"
    assert found[0].message == '"Cart" has no attribute "totl" (Did you mean "total"?)'
    with pytest.raises(ValueError):
        parse_mypy("not json", ROOT)


def test_bandit_output_is_read() -> None:
    """Check bandit's results, with the line range as the warning's lines."""
    output = {
        "errors": [],
        "results": [
            {
                "filename": "pkg/run.py",
                "line_number": 4,
                "line_range": [4, 6],
                "test_id": "B602",
                "issue_text": "subprocess call with shell=True identified, security issue.",
                "issue_severity": "HIGH",
            }
        ],
    }
    (found,) = parse_bandit(json.dumps(output), ROOT)
    assert (found.rule, found.line, found.end_line, found.severity) == ("B602", 4, 6, "high")


def test_semgrep_output_is_read_with_short_rule_names() -> None:
    """Check semgrep's results, with the folder path cut from the rule ID."""
    check = "opt.tools.semgrep-rules.lang.correctness.useless-eqeq.useless-eqeq"
    output = {
        "errors": [],
        "results": [
            {
                "check_id": check,
                "path": "pkg/cart.py",
                "start": {"line": 3, "col": 8},
                "end": {"line": 4, "col": 2},
                "extra": {"message": "This expression is always True", "severity": "INFO"},
            }
        ],
    }
    (found,) = parse_semgrep(json.dumps(output), ROOT)
    assert semgrep_rule(check) == "useless-eqeq"
    assert (found.rule, found.line, found.end_line, found.severity) == (
        "useless-eqeq",
        3,
        4,
        "info",
    )


def test_messages_are_put_on_one_line_and_cut() -> None:
    """Check whitespace is collapsed and long messages are cut."""
    found = warning(message="one\n   two\tthree " + "x" * 1000)
    assert found.message.startswith("one two three x")
    assert len(found.message) == MESSAGE_CHARS and found.message.endswith("...")


def test_only_warnings_on_changed_lines_are_kept() -> None:
    """Check a warning is kept when its lines touch a changed line, once."""
    inside = warning(line=3)
    spanning = warning(line=8, end_line=12, rule="misc")
    outside = warning(line=20, rule="arg-type")
    other_file = warning(line=3, file="pkg/other.py")
    kept = on_changed_lines(
        [outside, spanning, inside, inside, other_file], {"pkg/cart.py": [3, 10]}
    )
    assert kept == [inside, spanning]


def test_a_warning_becomes_tool_evidence_for_a_claim_on_its_line() -> None:
    """Check the evidence a warning gives supports a claim on the same lines."""
    evidence = warning(line=9).evidence()
    claim = Claim(
        file="pkg/cart.py",
        start_line=8,
        end_line=10,
        category="F.1",
        summary="Typo in an attribute name.",
        confidence=0.9,
    )
    assert evidence.tool == "mypy" and evidence.supports(claim)
    assert (
        warning(line=9).text() == 'pkg/cart.py:9 mypy [attr-defined] "Cart" has no attribute "totl"'
    )
