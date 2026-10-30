"""Reviews written without a model: the floor a reviewer must beat, and a check of the scoring.

- no_comment_reviews(): approves every PR, in the words SWR-Bench's own review prompt asks a
  reviewer to use when it finds nothing. It shows what doing nothing scores.
- known_issue_reviews(): lists each PR's known issues, in SWR-Bench's own summary of each, and
  approves clean PRs. No review can match the answer key better, so a low score here means
  the judge or the scoring is off, not the reviewer.

Clean PRs get the same approval in both, so scoring the second set after the first sends the
judge only the PRs with issues; the cache answers the rest.

The known-issue reviews read the answer key, so make them for the dev set only.

Imported by scripts/write_baseline_reviews.py; the file isn't run directly.
"""

from collections.abc import Sequence

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.scoring import Prediction

# What SWR-Bench's review prompt asks for when a reviewer finds no issue: "If you find no
# issues, please explicitly state that the pull request looks good and meets the standards."
APPROVAL = "The pull request looks good and meets the standards."


def no_comment_reviews(prs: Sequence[Instance]) -> list[Prediction]:
    """Approve every PR without raising anything.

    Args:
        prs (Sequence[Instance]): The PRs to review.

    Returns:
        list[Prediction]: One approval per PR, in the order of `prs`.
    """
    return [Prediction(instance_id=pr.instance_id, review=APPROVAL) for pr in prs]


def known_issue_review(pr: Instance) -> str:
    """Write a review that raises exactly the PR's known issues.

    Args:
        pr (Instance): The PR, with its known issues.

    Returns:
        str: A numbered list of SWR-Bench's summary of each known issue, or APPROVAL when the
            PR is clean.
    """
    if not pr.changes:
        return APPROVAL
    summaries = (issue.change_discussion.discussion_summary.strip() for issue in pr.changes)
    return "\n".join(f"{number}. {text}" for number, text in enumerate(summaries, start=1))


def known_issue_reviews(prs: Sequence[Instance]) -> list[Prediction]:
    """Review each PR with its own known issues.

    Args:
        prs (Sequence[Instance]): The PRs to review. Only dev PRs, since this reads the
            answer key.

    Returns:
        list[Prediction]: One review per PR, in the order of `prs`.
    """
    return [Prediction(instance_id=pr.instance_id, review=known_issue_review(pr)) for pr in prs]
