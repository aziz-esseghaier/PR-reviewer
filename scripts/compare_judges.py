"""Compare candidate judge models on a few dev PRs whose right scores are known.

For each candidate (a Google AI Studio model ID), runs SWR-Bench's own per-PR scoring
(evaluate_one() in third_party/swrbench/evaluation_struct.py) on test reviews whose correct
outcome we know:
- PRs with issues (5, picked from the smallest judge prompt to the largest):
  * known: the reviewers' own comments that raised the known issues. Every issue should be hit.
  * none: "No issues found..." No points, read as approving the PR, nothing hit.
  * wrong: another PR's reviewer comments, plausible but about other code. Nothing should be hit.
  * known-again: "known" asked again as a new sample. The same issues should be hit.
- Clean PRs (2): none (no points, read as approving) and wrong (points raised, not approving).

Every call goes through the LLM client, so it is cached, rate-limited and logged. Writes
results/judge-choice/: config.json, summary.md, and raw/<model>.jsonl with every judgment.

How to run (from the project folder, with the database up and GEMINI_API_KEY in .env):
    python scripts/compare_judges.py gemini-3.1-flash-lite gemma-4-26b-a4b-it
"""

import argparse
import json
import logging
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import psycopg

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.judge import JudgeChat, scorer_settings
from pr_reviewer.benchmark.official import load_evaluation
from pr_reviewer.benchmark.source import SOURCE_COMMIT, DatasetError, default_dataset_path
from pr_reviewer.benchmark.split import load_split, split_path, split_prs
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import (
    ConfigError,
    ModelLimits,
    ModelProfile,
    ModelsConfig,
    load_env,
    load_models_config,
    project_root,
)
from pr_reviewer.core.limits import LimitError
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.connection import connect

OUT_DIR = Path("results") / "judge-choice"
MAX_TOKENS = 8192  # what SWR-Bench's scorer asks the judge for by default
ISSUE_PRS = 5
CLEAN_PRS = 2
# The "none" review this trial ran with. Kept as it was, so a rerun is answered from the cache;
# reviewers now use SWR-Bench's own wording (schemas.NO_ISSUES).
NONE_REVIEW = "No issues found. The changes look correct."


@dataclass
class Judgment:
    """One scored test review and what the judge made of it.

    Attributes:
        model (str): Candidate model ID.
        pr (str): Instance ID.
        kind (str): "issues" or "clean".
        case (str): "known", "none", "wrong" or "known-again".
        valid (bool): Whether a reply passed SWR-Bench's checks within its 3 tries.
        attempts (int): Judge calls the scorer made for this review.
        approved (str | None): The judge's identified_as_good, "YES" or "NO".
        points (int): Review points the judge found.
        hits (list[str]): Known issues the judge counted as hit, such as "GT-POINT-1".
        total (int): Known issues of the PR.
        input_tokens (int): Prompt tokens of the judge calls.
        output_tokens (int): Reply tokens of the judge calls.
        seconds (float): Time of the calls that reached the provider.
        cached (bool): Whether every reply came from the cache.
        reply (str): The last reply text.
    """

    model: str
    pr: str
    kind: str
    case: str
    valid: bool = False
    attempts: int = 0
    approved: str | None = None
    points: int = 0
    hits: list[str] = field(default_factory=list)
    total: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    reply: str = ""

    @property
    def as_expected(self) -> bool:
        """Whether the judge scored this test review the way it should have."""
        if not self.valid:
            return False
        if self.case in ("known", "known-again"):
            return len(self.hits) == self.total and self.approved == "NO"
        if self.case == "none":
            return self.points == 0 and not self.hits and self.approved == "YES"
        if self.kind == "issues":  # wrong review on a PR with issues
            return not self.hits
        return self.points > 0 and self.approved == "NO"  # wrong review on a clean PR


def as_review(comments: Sequence[str]) -> str:
    """Number review comments the way a reviewer's list of points would look.

    Args:
        comments (Sequence[str]): The comments, in order. Repeats are dropped.

    Returns:
        str: The review text.
    """
    unique = list(dict.fromkeys(comment.strip() for comment in comments if comment.strip()))
    return "\n\n".join(f"{number}. {comment}" for number, comment in enumerate(unique, start=1))


def reviewer_comments(pr: Instance) -> list[str]:
    """List the comments in which the PR's reviewers raised its known issues.

    Args:
        pr (Instance): A PR with known issues.

    Returns:
        list[str]: One comment per issue, or SWR-Bench's summary when a comment is empty.
    """
    return [
        issue.change_discussion.original_reviewer_comment.strip()
        or issue.change_discussion.discussion_summary
        for issue in pr.changes
    ]


def spread(prs: list[Instance], count: int, size: dict[str, int]) -> list[Instance]:
    """Pick PRs spread from the smallest judge prompt to the largest.

    Args:
        prs (list[Instance]): The PRs to pick from.
        count (int): How many to pick, at least 2.
        size (dict[str, int]): Judge prompt length per instance ID.

    Returns:
        list[Instance]: The picked PRs, smallest first.
    """
    ordered = sorted(prs, key=lambda pr: (size[pr.instance_id], pr.instance_id))
    last = len(ordered) - 1
    return [ordered[round(index * last / (count - 1))] for index in range(count)]


def make_test_reviews(
    swr: ModuleType, dev: list[Instance], rows: dict[str, dict[str, Any]]
) -> list[tuple[Instance, str, str]]:
    """Pick the dev PRs and write the test reviews for them.

    Args:
        swr (ModuleType): SWR-Bench's scoring script, used to size the judge prompts.
        dev (list[Instance]): The dev PRs.
        rows (dict[str, dict[str, Any]]): The raw dataset record of each dev PR.

    Returns:
        list[tuple[Instance, str, str]]: (PR, case name, review text) for every test review.
    """
    size: dict[str, int] = {}
    for pr in dev:
        item = {
            "instance": swr.fix_change_type(json.loads(json.dumps(rows[pr.instance_id]))),
            "pred": {"review": ""},
        }
        build = swr.create_change_pr_prompt if pr.changes else swr.create_clean_pr_prompt
        size[pr.instance_id] = len(build(item)[0])
    with_issues = spread([pr for pr in dev if pr.changes], ISSUE_PRS, size)
    clean = spread([pr for pr in dev if not pr.changes], CLEAN_PRS + 1, size)[1:]
    reviews: list[tuple[Instance, str, str]] = []
    for index, pr in enumerate(with_issues):
        other = with_issues[(index + 1) % len(with_issues)]
        known = as_review(reviewer_comments(pr))
        reviews += [
            (pr, "known", known),
            (pr, "none", NONE_REVIEW),
            (pr, "wrong", as_review(reviewer_comments(other))),
            (pr, "known-again", known),
        ]
    for index, pr in enumerate(clean):
        reviews += [
            (pr, "none", NONE_REVIEW),
            (pr, "wrong", as_review(reviewer_comments(with_issues[index]))),
        ]
    return reviews


def judge_one(
    swr: ModuleType,
    chat: JudgeChat,
    model: str,
    pr: Instance,
    case: str,
    review: str,
    row: dict[str, Any],
) -> Judgment:
    """Score one test review with SWR-Bench's evaluate_one() and record the outcome.

    Args:
        swr (ModuleType): SWR-Bench's scoring script, loaded with `chat` as its run_chat.
        chat (JudgeChat): The candidate judge.
        model (str): Candidate model ID.
        pr (Instance): The PR.
        case (str): Name of the test review.
        review (str): The test review.
        row (dict[str, Any]): The PR's raw dataset record.

    Returns:
        Judgment: What the judge made of it.
    """
    judgment = Judgment(model, pr.instance_id, "issues" if pr.changes else "clean", case)
    calls, replies = chat.calls, len(chat.responses)
    item = {"instance": json.loads(json.dumps(row)), "pred": {"review": review}}
    result = swr.evaluate_one(scorer_settings(chat.client), item, logging.getLogger("swr"))
    new = chat.responses[replies:]
    judgment.attempts = chat.calls - calls
    judgment.input_tokens = sum(reply.input_tokens for reply in new)
    judgment.output_tokens = sum(reply.output_tokens for reply in new)
    judgment.seconds = round(sum(reply.latency_s for reply in new if not reply.cached), 1)
    judgment.cached = bool(new) and all(reply.cached for reply in new)
    judgment.reply = new[-1].text if new else ""
    if result is not None:
        judgment.valid = True
        judgment.approved = result["identified_as_good"]
        judgment.points = len(result["pred_points"])
        judgment.hits = [gt["id"] for gt in result.get("gt_points", []) if gt["hit"] == "YES"]
        judgment.total = len(pr.changes)
    return judgment


def summarize(model: str, judgments: list[Judgment], stopped: str | None) -> dict[str, Any]:
    """Count how well one candidate did.

    Args:
        model (str): Candidate model ID.
        judgments (list[Judgment]): Its judgments.
        stopped (str | None): Why its run stopped early, if it did.

    Returns:
        dict[str, Any]: The numbers for the summary table.
    """
    by_case = {(j.pr, j.case): j for j in judgments}
    known = [j for j in judgments if j.case == "known"]
    again = [j for j in judgments if j.case == "known-again"]
    same = sum(
        j.valid and by_case[j.pr, "known"].valid and j.hits == by_case[j.pr, "known"].hits
        for j in again
    )
    sent = [j.seconds / max(j.attempts, 1) for j in judgments if not j.cached and j.seconds]
    return {
        "model": model,
        "stopped": stopped,
        "reviews": len(judgments),
        "as_expected": sum(j.as_expected for j in judgments),
        "valid_first_try": sum(j.valid and j.attempts == 1 for j in judgments),
        "no_valid_answer": sum(not j.valid for j in judgments),
        "known_hits": sum(len(j.hits) for j in known),
        "known_total": sum(j.total for j in known),
        "wrong_hits": sum(len(j.hits) for j in judgments if j.case == "wrong"),
        "none_right": sum(j.as_expected for j in judgments if j.case == "none"),
        "none_total": sum(j.case == "none" for j in judgments),
        "same_twice": same,
        "again_total": len(again),
        "median_seconds": round(statistics.median(sent), 1) if sent else None,
        "input_tokens": sum(j.input_tokens for j in judgments),
        "output_tokens": sum(j.output_tokens for j in judgments),
    }


def write_summary(path: Path, rows: list[dict[str, Any]], judgments: list[Judgment]) -> None:
    """Write the comparison as a Markdown table, plus every judgment that went wrong.

    Args:
        path (Path): The summary file.
        rows (list[dict[str, Any]]): One summary per candidate.
        judgments (list[Judgment]): Every judgment of every candidate.
    """
    lines = [
        "# Judge candidates on dev PRs with known right answers",
        "",
        "| Model | As expected | Valid JSON 1st try | Known issues hit | Hits on wrong "
        'reviews | "No issues" read right | Same answer twice | Median s/call | Tokens in/out |',
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        note = f" (stopped: {row['stopped']})" if row["stopped"] else ""
        lines.append(
            f"| {row['model']}{note} | {row['as_expected']}/{row['reviews']} "
            f"| {row['valid_first_try']}/{row['reviews']} "
            f"| {row['known_hits']}/{row['known_total']} | {row['wrong_hits']} "
            f"| {row['none_right']}/{row['none_total']} "
            f"| {row['same_twice']}/{row['again_total']} | {row['median_seconds']} "
            f"| {row['input_tokens']:,} / {row['output_tokens']:,} |"
        )
    misses = [j for j in judgments if not j.as_expected]
    lines += ["", "## Judgments that weren't as expected", ""]
    if not misses:
        lines.append("None.")
    for j in misses:
        state = (
            "no valid answer"
            if not j.valid
            else (f"approved={j.approved}, points={j.points}, hits={len(j.hits)}/{j.total}")
        )
        lines.append(f"- {j.model}, {j.pr}, {j.case}: {state}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(models: Sequence[str], rpm: int) -> int:
    """Run every candidate on the test reviews and write the results folder.

    Args:
        models (Sequence[str]): Candidate model IDs at Google AI Studio.
        rpm (int): Requests per minute to stay under for each candidate.

    Returns:
        int: Exit code, 0 when it finished.
    """
    instances = load_dataset(default_dataset_path())
    dev = split_prs(instances, load_split(split_path()), "dev")
    wanted = {pr.instance_id for pr in dev}
    with default_dataset_path().open(encoding="utf-8") as file:
        rows = {r["instance_id"]: r for r in map(json.loads, file) if r["instance_id"] in wanted}
    base = load_models_config()
    config = ModelsConfig(
        providers={"google": base.provider("google")},
        models={
            f"judge-trial-{model}": ModelProfile(
                provider="google",
                model=model,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                limits=ModelLimits(requests_per_minute=rpm),
            )
            for model in models
        },
    )
    reviews = make_test_reviews(load_evaluation(), dev, rows)
    out = project_root() / OUT_DIR
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "candidates": list(models),
                "settings": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "rpm": rpm},
                "swr_bench_commit": SOURCE_COMMIT,
                "test_reviews": [[pr.instance_id, case] for pr, case, _ in reviews],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    all_judgments: list[Judgment] = []
    summary_rows = []
    with connect() as conn:
        for model in models:
            client = LLMClient(
                f"judge-trial-{model}",
                config,
                cache=PostgresCache(conn),
                usage=PostgresUsageLog(conn),
            )
            chat = JudgeChat(client)
            swr = load_evaluation(chat)
            judgments: list[Judgment] = []
            stopped = None
            print(f"\n{model}: {len(reviews)} test reviews")
            for pr, case, review in reviews:
                try:
                    judgment = judge_one(swr, chat, model, pr, case, review, rows[pr.instance_id])
                except LimitError as exc:
                    stopped = str(exc)
                    print(f"  stopped: {exc}")
                    break
                judgments.append(judgment)
                mark = "ok  " if judgment.as_expected else "MISS"
                print(
                    f"  {mark} {pr.instance_id:<36} {case:<12} points={judgment.points} "
                    f"hits={len(judgment.hits)}/{judgment.total} approved={judgment.approved} "
                    f"tries={judgment.attempts}{' cached' if judgment.cached else ''}"
                )
            with (out / "raw" / f"{model}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                f.writelines(json.dumps(asdict(j)) + "\n" for j in judgments)
            all_judgments += judgments
            summary_rows.append(summarize(model, judgments, stopped))
    write_summary(out / "summary.md", summary_rows, all_judgments)
    print(f"\nWrote {out / 'summary.md'}")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, run the comparison and turn setup problems into short messages.

    Args:
        argv (Sequence[str] | None): Arguments to parse. None means the real command line.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description="Compare judge models on known test reviews.")
    parser.add_argument("models", nargs="+", help="Google AI Studio model IDs to compare")
    parser.add_argument("--rpm", type=int, default=5, help="requests per minute (default: 5)")
    args = parser.parse_args(argv)
    load_env()
    logging.basicConfig(format="%(message)s")
    logging.getLogger("pr_reviewer").setLevel(logging.INFO)
    try:
        return run(args.models, args.rpm)
    except (ConfigError, DatasetError) as exc:
        print(f"Problem: {exc}", file=sys.stderr)
    except psycopg.OperationalError as exc:
        print(
            f"Can't reach PostgreSQL: {exc}\nStart it with: docker compose up -d", file=sys.stderr
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
