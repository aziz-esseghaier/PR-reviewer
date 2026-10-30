"""The text a reviewer sees for one PR, written exactly as SWR-Bench's review generator writes it.

render_pr_details() does what create_prinfo_prompt() does in SWR-Bench's generation.py (in
third_party/swrbench/): the PR's title and description, then each reviewed commit with its
message and the diff of every file it changed. SWR-Bench's scorer was designed for reviews
written from this input, and a test checks that both functions give the same text for every PR.

The text comes from the PR's author, so it's untrusted: a prompt that includes it must treat
it as data to review, never as instructions.

Sizes are estimated at about 4 characters per token. That's close enough to plan a token
budget; providers report the exact count after each call.

Imported by other modules; the file isn't run directly.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from pr_reviewer.benchmark.dataset import Instance

CHARS_PER_TOKEN = 4

# Groq's free tier allows 8,000 tokens per minute, so a bigger request can never be sent to it.
LARGE_INPUT_TOKENS = 8_000


def render_pr_details(instance: Instance) -> str:
    """Write a PR's title, description and reviewed commits the way SWR-Bench does.

    Args:
        instance (Instance): The PR.

    Returns:
        str: The "Pull Request Details" text, character for character as SWR-Bench's
        create_prinfo_prompt() writes it.
    """
    changes = ""
    for commit in instance.pr_commits:
        block = f"Commit: {commit.sha}\nCommit Message: {commit.message}\nCommit Code Changes: \n"
        for patch in commit.diff:
            block += f"{patch.file}\n```\n{patch.patch}\n```\n"
        changes += f"{block}\n"
    return (
        "**Pull Request Details**: \n"
        f"- **Title**: {instance.pr_title} \n"
        f"- **Description**: {instance.pr_statement} \n"
        f"- **Code Changes**: {changes.strip()} \n"
    )


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens a text takes, at CHARS_PER_TOKEN characters per token.

    Args:
        text (str): The text.

    Returns:
        int: The estimate, rounded up.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class InputSizes:
    """How big the review inputs of a set of PRs are, in estimated tokens.

    Attributes:
        median (int): Half of the inputs are this size or smaller.
        p90 (int): 90% of the inputs are this size or smaller.
        largest (int): The biggest input.
        over_limit (int): How many inputs are bigger than LARGE_INPUT_TOKENS.
    """

    median: int
    p90: int
    largest: int
    over_limit: int


def input_sizes(prs: Sequence[Instance]) -> InputSizes:
    """Measure the review inputs of a set of PRs.

    Args:
        prs (Sequence[Instance]): The PRs. Must not be empty.

    Returns:
        InputSizes: The median, 90th percentile and largest size, and the count over the limit.

    Raises:
        ValueError: If there are no PRs.
    """
    if not prs:
        raise ValueError("no PRs to measure")
    tokens = sorted(estimate_tokens(render_pr_details(pr)) for pr in prs)
    return InputSizes(
        median=statistics.median_low(tokens),
        p90=tokens[math.ceil(0.9 * len(tokens)) - 1],
        largest=tokens[-1],
        over_limit=sum(size > LARGE_INPUT_TOKENS for size in tokens),
    )
