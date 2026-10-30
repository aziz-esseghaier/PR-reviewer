"""
Tests for the review input in benchmark/review_input.py.

Runs SWR-Bench's own create_prinfo_prompt() from third_party/swrbench/generation.py and checks
that ours writes the same text, on small built PRs and, once it is downloaded, on every PR of
the real dataset. Also checks the token estimate and the input size summary.

How to run (from the project folder):
    pytest tests/benchmark/test_review_input.py
"""

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.review_input import (
    LARGE_INPUT_TOKENS,
    estimate_tokens,
    input_sizes,
    render_pr_details,
)
from tests.benchmark.test_dataset import REAL_DATASET, make_commit, make_row, sha

OFFICIAL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "swrbench"
GENERATION_SCRIPT = OFFICIAL_DIR / "generation.py"

OfficialPrompt = Callable[[dict[str, Any]], str]


@pytest.fixture(scope="module")
def official_prompt() -> OfficialPrompt:
    """Load create_prinfo_prompt() from SWR-Bench's generation.py, unchanged.

    The script imports `utils` (whose model helper crashes on import) and `tqdm` (not
    installed) at the top, but create_prinfo_prompt() uses neither, so both are replaced by
    empty stand-ins while it loads.

    Returns:
        OfficialPrompt: SWR-Bench's function, taking a raw dataset record.
    """
    stand_ins = {"utils": SimpleNamespace(run_chat=None), "tqdm": SimpleNamespace(tqdm=None)}
    spec = importlib.util.spec_from_file_location("swrbench_generation", GENERATION_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stand_ins):
        spec.loader.exec_module(module)
    return cast(OfficialPrompt, module.create_prinfo_prompt)


def as_instance(row: dict[str, Any]) -> Instance:
    """Check a raw record the way the loader does.

    Args:
        row (dict[str, Any]): The record.

    Returns:
        Instance: The checked PR.
    """
    return Instance.model_validate_json(json.dumps(row))


def odd_row() -> dict[str, Any]:
    """Make a PR whose text has the details a careless copy would get wrong.

    Returns:
        dict[str, Any]: Three commits, several files each, multi-line messages, a description
        with Markdown and trailing spaces, a patch with Windows line endings and non-ASCII text.
    """
    row = make_row(3, commits=3)
    row["pr_title"] = "Fix `mean()` for empty carts  "
    row["pr_statement"] = "Fixes #12.\r\n\r\n* Adds a check\n* Café ünïcode ✓\n\n"
    row["pr_commits"][1] = make_commit(
        11, "2020-03-01T01:00:00+00:00", "Refactor\n\nLonger body.\n", ("x/y.py", "z.py")
    )
    row["all_commits"][1] = row["pr_commits"][1]
    row["pr_commits"][2]["diff"][0]["patch"] = "@@ -1 +1 @@\r\n-old\r\n+new\r\n"
    return row


@pytest.mark.parametrize(
    "row",
    [make_row(1), make_row(2, issues=()), odd_row()],
    ids=["one commit", "clean PR", "odd text"],
)
def test_text_matches_swr_bench(official_prompt: OfficialPrompt, row: dict[str, Any]) -> None:
    """Check that our text equals SWR-Bench's, character for character."""
    assert render_pr_details(as_instance(row)) == official_prompt(row)


def test_text_shows_every_reviewed_commit_and_file() -> None:
    """Check the layout: title, description, then each commit with its files in order."""
    text = render_pr_details(as_instance(odd_row()))
    assert text.startswith("**Pull Request Details**: \n- **Title**: Fix `mean()`")
    parts = (f"Commit: {sha(10)}", f"Commit: {sha(11)}", "x/y.py", "z.py", f"Commit: {sha(12)}")
    positions = [text.index(part) for part in parts]
    assert positions == sorted(positions)
    assert text.endswith("+new\r\n\n``` \n")


def test_text_matches_swr_bench_on_every_real_pr(official_prompt: OfficialPrompt) -> None:
    """Check our text against SWR-Bench's for all 1,000 PRs of the real dataset."""
    if not REAL_DATASET.is_file():
        pytest.skip("SWR-Bench file not downloaded (run: pr-reviewer bench download)")
    with REAL_DATASET.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    instances = load_dataset(REAL_DATASET)
    assert len(rows) == len(instances) == 1000
    different = [
        pr.instance_id
        for row, pr in zip(rows, instances, strict=True)
        if render_pr_details(pr) != official_prompt(row)
    ]
    assert different == []


@pytest.mark.parametrize(("text", "tokens"), [("", 0), ("abcd", 1), ("abcde", 2)])
def test_tokens_are_estimated_and_rounded_up(text: str, tokens: int) -> None:
    """Check the estimate of 4 characters per token, rounded up."""
    assert estimate_tokens(text) == tokens


def test_input_sizes_count_the_inputs_over_the_limit() -> None:
    """Check the median, 90th percentile, largest size and the count over the limit."""
    huge = make_row(1)
    huge["pr_commits"][0]["diff"][0]["patch"] = "x" * (LARGE_INPUT_TOKENS * 4 + 4)
    prs = [as_instance(make_row(1))] * 10 + [as_instance(huge)]  # 90% of 11 is the 10th
    sizes = input_sizes(prs)
    small = estimate_tokens(render_pr_details(prs[0]))
    assert (sizes.median, sizes.p90) == (small, small)
    assert sizes.largest > LARGE_INPUT_TOKENS
    assert sizes.over_limit == 1


def test_input_sizes_need_prs() -> None:
    """Check that measuring no PRs is an error, not a crash on an empty list."""
    with pytest.raises(ValueError, match="no PRs"):
        input_sizes([])
