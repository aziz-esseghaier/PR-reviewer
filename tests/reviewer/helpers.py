"""Shared test data for the reviewer tests: small PRs and model replies.

Imported by the test files in this folder; the file isn't run directly.
"""

import json
from collections.abc import Sequence
from typing import Any

from pr_reviewer.benchmark.dataset import Instance
from tests.benchmark.test_dataset import make_row


def make_pr(
    files: Sequence[tuple[str, str]] = (("src/shop/cart.py", "@@ -1 +1 @@\n-a\n+b"),),
    description: str = "Adds average_price.",
    number: int = 1,
    issues: Sequence[str] = (),
) -> Instance:
    """Make a 2022 PR with one reviewed commit that changes the given files.

    Args:
        files (Sequence[tuple[str, str]]): Each changed file's path and diff.
        description (str): The PR's description.
        number (int): PR number, used in the instance ID.
        issues (Sequence[str]): Change types of its known issues; empty for a clean PR.

    Returns:
        Instance: The PR.
    """
    row = make_row(number, year=2022, issues=issues)
    row["pr_statement"] = description
    row["pr_commits"][0]["diff"] = [{"file": path, "patch": patch} for path, patch in files]
    return Instance.model_validate_json(json.dumps(row))


def finding(file: str = "src/shop/cart.py", line: int = 6, **changes: Any) -> dict[str, Any]:
    """Make one finding as the model writes it.

    Args:
        file (str): The file.
        line (int): Its line.
        **changes (Any): Fields to change, such as category="E.1.1".

    Returns:
        dict[str, Any]: The finding.
    """
    return {
        "file": file,
        "start_line": line,
        "end_line": line,
        "category": "F.4",
        "summary": "average_price divides by zero when the cart is empty.",
        "details": "len(items) is 0 for an empty cart.",
        "expected_failure": "ZeroDivisionError",
        "confidence": 0.8,
        **changes,
    }


def reply(*findings: dict[str, Any]) -> str:
    """Write a model reply holding some findings.

    Args:
        *findings (dict[str, Any]): The findings.

    Returns:
        str: The JSON reply.
    """
    return json.dumps({"findings": list(findings)})
