"""Writing a review run's results folder: its settings, a summary, the reviews and findings.

results/<name>/ gets:
- review_config.json: the run's settings (model, prompt fingerprint, size limit, PRs). In git.
- review_summary.md: how the run went: PRs reviewed, failed and trimmed, findings by
  category, the context the requests held (for reviewers with context), model calls and
  tokens. In git.
- raw/reviews.jsonl: one review per PR, the file `pr-reviewer score` reads. Not in git.
- raw/findings.jsonl: every finding with its claim. Not in git.
The reviews file is written only once every PR in scope has a review, so a stopped run is
never scored on part of its PRs. The scores later go to the same folder (benchmark/results.py).

Imported by other modules; the file isn't run directly.
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.results import REVIEW_CONFIG
from pr_reviewer.benchmark.scoring import Prediction, save_predictions
from pr_reviewer.context.candidates import KIND_ORDER
from pr_reviewer.core.schemas import CATEGORY_NAMES
from pr_reviewer.reviewer.contextual import MIN_CONTEXT_CHARS
from pr_reviewer.reviewer.store import SavedReview

# The run's summary, next to its settings.
REVIEW_SUMMARY = "review_summary.md"

# The reviews file, in raw/.
REVIEWS_FILE = "reviews.jsonl"


def _write_json(path: Path, data: Any) -> None:
    """Write JSON with LF line endings.

    Args:
        path (Path): The file.
        data (Any): What to write.
    """
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def context_lines(done: Sequence[SavedReview]) -> list[str]:
    """Describe the context the reviews' requests held, as Markdown lines.

    Args:
        done (Sequence[SavedReview]): The finished reviews.

    Returns:
        list[str]: A "Context" section, or nothing when no review had a retrieve step.
    """
    contexts = [review.context for review in done if review.context is not None]
    if not contexts:
        return []
    with_context = [c for c in contexts if c.get("included")]
    no_room = sum(1 for c in contexts if c.get("room", 0) < MIN_CONTEXT_CHARS)
    chars = sorted(int(c.get("chars", 0)) for c in with_context)
    included = Counter(label.split(" ", 1)[0] for c in contexts for label in c.get("included", []))
    left_out = Counter(label.split(" ", 1)[0] for c in contexts for label in c.get("left_out", []))
    lines = [
        "## Context",
        "",
        f"- PRs whose request held context: {len(with_context)} of {len(contexts)}; "
        f"{no_room} had no room left after the PR's text, "
        f"{len(contexts) - len(with_context) - no_room} had room but no piece (none found, "
        "or none small enough)",
    ]
    if chars:
        lines.append(
            f"- Characters of context: median {chars[len(chars) // 2]:,}, "
            f"largest {chars[-1]:,}, total {sum(chars):,}"
        )
    lines += ["", "| Kind | Pieces in | Pieces left out |", "|---|---|---|"]
    for kind in KIND_ORDER:
        lines.append(f"| {kind} | {included[kind]} | {left_out[kind]} |")
    lines.append("")
    return lines


def summary_markdown(
    settings: dict[str, Any],
    prs: Sequence[Instance],
    reviews: dict[str, SavedReview],
    findings: Sequence[dict[str, Any]],
) -> str:
    """Describe how a review run went, as Markdown.

    Args:
        settings (dict[str, Any]): The run's settings.
        prs (Sequence[Instance]): The PRs in the run's scope.
        reviews (dict[str, SavedReview]): The saved reviews, by PR.
        findings (Sequence[dict[str, Any]]): Every finding, with its claim.

    Returns:
        str: The text of review_summary.md.
    """
    saved = [reviews[pr.instance_id] for pr in prs if pr.instance_id in reviews]
    done = [review for review in saved if review.status == "done"]
    failed = [review for review in saved if review.status == "failed"]
    waiting = len(prs) - len(saved)
    trimmed = [review for review in done if review.trimmed]
    in_scope = {pr.instance_id for pr in prs}
    claims = [item["claim"] for item in findings if item["instance_id"] in in_scope]
    categories = Counter(claim["category"] for claim in claims)
    functional = sum(claim["category"].startswith("F") for claim in claims)
    calls = sum(review.tries for review in saved)
    tokens_in = sum(review.input_tokens for review in saved)
    tokens_out = sum(review.output_tokens for review in saved)
    effort = settings.get("reasoning_effort") or "the provider's default"
    lines = [
        f"# Review run: {settings['name']}",
        "",
        f"- Reviewer: {settings['reviewer']}, model {settings['model']} (profile "
        f"{settings['profile']}, temperature {settings['temperature']}, reasoning effort "
        f"{effort}, replies up to {settings['max_tokens']:,} tokens)",
        f"- PRs: the {len(prs)} {settings['split']} PRs from {settings['since']} on",
        f"- Reviewed: {len(done)}; failed (no usable reply in {settings['max_tries']} tries): "
        f"{len(failed)}; not reviewed yet: {waiting}",
        f"- Size limit: {settings['input_budget_chars']:,} characters of PR text per request; "
        f"{len(trimmed)} PRs trimmed",
        "",
        "## Findings",
        "",
        f"{len(claims)} findings, {len(claims) / len(done):.2f} per reviewed PR"
        if done
        else "No PR reviewed yet.",
        "",
    ]
    if done:
        none = sum(review.findings == 0 for review in done)
        lines += [
            f'- PRs with no finding (reviewed as "looks good"): {none}',
            f"- Functional (F): {functional}; evolvability (E): {len(claims) - functional}",
            "",
            "| Category | Name | Findings |",
            "|---|---|---|",
        ]
        lines += [
            f"| {code} | {name} | {categories[code]} |"
            for code, name in CATEGORY_NAMES.items()
            if categories[code]
        ]
        lines.append("")
    if trimmed:
        lines += ["## Trimmed PRs", "", "| PR | File diffs left out |", "|---|---|"]
        lines += [f"| {review.instance_id} | {len(review.left_out)} |" for review in trimmed]
        lines.append("")
    lines += context_lines(done)
    retried = [review for review in saved if review.tries > 1 or review.failures]
    lines += [
        "## Model use",
        "",
        f"- Model calls: {calls} ({len(retried)} PRs needed more than one try)",
        f"- Tokens: {tokens_in:,} in, {tokens_out:,} out (thinking included)",
    ]
    if failed:
        lines += ["", "Failed PRs: " + ", ".join(review.instance_id for review in failed)]
    return "\n".join(lines) + "\n"


def write_review_files(
    folder: Path,
    settings: dict[str, Any],
    prs: Sequence[Instance],
    reviews: dict[str, SavedReview],
    findings: Sequence[dict[str, Any]],
) -> Path | None:
    """Write a review run's folder: settings and summary always, reviews once all are in.

    Args:
        folder (Path): results/<name>/. Created if missing.
        settings (dict[str, Any]): The run's settings, saved as review_config.json.
        prs (Sequence[Instance]): The PRs in the run's scope, in the order to write.
        reviews (dict[str, SavedReview]): The saved reviews, by PR.
        findings (Sequence[dict[str, Any]]): Every finding, with its claim.

    Returns:
        Path | None: The reviews file, or None while a PR in scope has no review.
    """
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / REVIEW_CONFIG, settings)
    summary = summary_markdown(settings, prs, reviews, findings)
    (folder / REVIEW_SUMMARY).write_text(summary, encoding="utf-8", newline="\n")
    done = [
        reviews[pr.instance_id]
        for pr in prs
        if pr.instance_id in reviews and reviews[pr.instance_id].status == "done"
    ]
    if len(done) < len(prs):
        return None
    raw = folder / "raw"
    path = raw / REVIEWS_FILE
    save_predictions(path, [Prediction(instance_id=r.instance_id, review=r.review) for r in done])
    in_scope = {pr.instance_id for pr in prs}
    with (raw / "findings.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for item in findings:
            if item["instance_id"] in in_scope:
                file.write(json.dumps(item) + "\n")
    return path
