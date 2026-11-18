"""The `pr-reviewer review ...` commands: review dev PRs with a reviewer, ready for scoring.

cli.py adds them to the `pr-reviewer` command. Only dev PRs are reviewed; the test set stays
locked until the final run.

How to run (inside the conda env, from the project folder, with the database running):
    pr-reviewer review naive                  review the dev PRs from 2022 on with the naive
                                              reviewer, as run "naive-gpt-oss-120b"
    pr-reviewer review context                the same PRs with the context reviewer, as run
                                              "context-gpt-oss-120b" (needs `pr-reviewer
                                              context build` first)
    pr-reviewer review candidates             the same PRs with the candidate reviewer (recall
                                              first, static warnings, lookups), as run
                                              "candidates-gpt-oss-120b" (needs `pr-reviewer
                                              context build` and `pr-reviewer static run`
                                              first, and Docker for the lookups)
    pr-reviewer review candidates --with-issues --variant v1 --name candidates-gpt-oss-120b-v1
                                              only the PRs with known issues, with prompt
                                              variant v1 (for tuning recall)
    pr-reviewer review naive --since 2019     the older dev PRs too (under another --name)
    pr-reviewer review naive --ids <id> ...   only these PRs (the run's other PRs stay to do)
    pr-reviewer review naive --limit 5        at most 5 PRs this time
    pr-reviewer review compare                compare the scored context run with the scored
                                              naive run, PR by PR (no model calls)
    pr-reviewer review compare --base context-gpt-oss-120b --new candidates-gpt-oss-120b
                                              the same for any two scored runs; a run over
                                              some of the base run's PRs is compared on those
Running a review again continues the run: finished PRs are kept, a review stopped by the daily
quota resumes where it stopped, and failed PRs are tried again with fresh replies. Once every
PR has a review, it prints the `pr-reviewer score` command for the run.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docker import DockerClient

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.judge import check_reviewer
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import (
    SPLIT_SHA256,
    add_split_arguments,
    chosen_prs,
    load_split,
    split_path,
)
from pr_reviewer.context.candidates import KIND_ORDER, load_candidates
from pr_reviewer.context.lookups import Lookup, LookupLimits, LookupResult, run_lookups
from pr_reviewer.context.probe import VERSION as PROBE_VERSION
from pr_reviewer.context.retrieve import ContextError, ready_indexer, retrieval_setup
from pr_reviewer.context.store import stored_contexts
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import ModelProfile, load_models_config
from pr_reviewer.core.limits import LimitError
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.checkpoints import open_checkpointer
from pr_reviewer.db.connection import connect
from pr_reviewer.reviewer import candidates as candidate_reviewer
from pr_reviewer.reviewer.candidates import Explorer, StaticInput
from pr_reviewer.reviewer.compare import comparison_markdown, load_judgments, same_prs
from pr_reviewer.reviewer.contextual import (
    MIN_CONTEXT_CHARS,
    context_prompt_fingerprint,
    retrieve_node,
)
from pr_reviewer.reviewer.naive import (
    MAX_FINDINGS,
    MAX_TRIES,
    PrReview,
    RetrieveNode,
    input_budget_chars,
    prompt_fingerprint,
    review_pr,
)
from pr_reviewer.reviewer.report import write_review_files
from pr_reviewer.reviewer.store import (
    load_findings,
    load_reviews,
    review_status,
    save_review,
    start_run,
)
from pr_reviewer.sandbox.commands import logs_dir
from pr_reviewer.sandbox.config import SandboxConfig, load_sandbox_config
from pr_reviewer.sandbox.repos import RepoError
from pr_reviewer.sandbox.runner import SandboxError, docker_client
from pr_reviewer.setup_agent.store import usable_prs
from pr_reviewer.static.store import StoredStatic, stored_static
from pr_reviewer.static.tools import static_setup

# The reviewer's model profile in configs/models.yaml: one model for every reviewer experiment.
REVIEWER_PROFILE = "reviewer-gpt-oss-120b"

# The runs' default names, which are also their results folders.
DEFAULT_NAME = "naive-gpt-oss-120b"
CONTEXT_NAME = "context-gpt-oss-120b"
CANDIDATES_NAME = "candidates-gpt-oss-120b"

# The candidate reviewer's prompt variant used when none is given.
DEFAULT_VARIANT = "v1"

# Development starts on the dev PRs from this year on.
FIRST_YEAR = 2022


def _chosen(args: argparse.Namespace, in_scope: list[Instance]) -> list[Instance] | None:
    """Pick the PRs to review this time.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `ids`.
        in_scope (list[Instance]): The run's PRs.

    Returns:
        list[Instance] | None: The PRs asked for, or None (after a message) when one isn't in
        the run's scope.
    """
    if not args.ids:
        return in_scope
    wanted = set(args.ids)
    chosen = [pr for pr in in_scope if pr.instance_id in wanted]
    if len(chosen) != len(wanted):
        missing = sorted(wanted - {pr.instance_id for pr in chosen})
        print(
            f"Not {args.split} PRs from {args.since} on (or unknown): {', '.join(missing)}",
            file=sys.stderr,
        )
        return None
    return chosen


def _naive(args: argparse.Namespace) -> int:
    """Review the chosen dev PRs with the naive reviewer, and write the run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `since`, `ids`, `limit`
            and `data`.

    Returns:
        int: Exit code, 0 when every PR in the run has a review, ready to score.
    """
    return _review(args, "naive")


def _context(args: argparse.Namespace) -> int:
    """Review the chosen dev PRs with the context reviewer, and write the run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `since`, `ids`, `limit`
            and `data`.

    Returns:
        int: Exit code, 0 when every PR in the run has a review, ready to score.
    """
    return _review(args, "context")


def _candidates(args: argparse.Namespace) -> int:
    """Review the chosen dev PRs with the candidate reviewer, and write the run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `since`, `ids`, `limit`,
            `with_issues`, `variant` and `data`.

    Returns:
        int: Exit code, 0 when every PR in the run has a review, ready to score.
    """
    return _review(args, "candidates")


class LookupRunner:
    """Answers a PR's lookups in an indexer sandbox, connecting to Docker only when needed.

    Args:
        config (SandboxConfig): The sandbox settings.
        limits (LookupLimits): How much each answer may show.
    """

    def __init__(self, config: SandboxConfig, limits: LookupLimits) -> None:
        self._config = config
        self._limits = limits
        self._client: DockerClient | None = None
        self._image = ""

    def for_pr(self, pr: Instance) -> Explorer:
        """Give the explorer for one PR.

        Args:
            pr (Instance): The PR.

        Returns:
            Explorer: Answers the PR's lookups. A lookup that can't be answered for a lasting
            reason (the tool failed, the commit is missing) gets that reason as its answer;
            a Docker failure stops the run (SandboxError), so the review resumes later.
        """

        def explore(lookups: list[Lookup]) -> list[LookupResult]:
            if self._client is None:
                self._client = docker_client()
                self._image = ready_indexer(self._client, self._config)
            try:
                return run_lookups(
                    self._client, self._config, self._image, pr, lookups, self._limits, logs_dir()
                )
            except (ContextError, RepoError) as exc:
                reason = f"the lookup tools failed ({str(exc)[:200]})"
                return [
                    LookupResult(lookup=lookup, ok=False, text=f"{lookup.describe()}: {reason}")
                    for lookup in lookups
                ]

        return explore


def _static_input(stored: StoredStatic) -> StaticInput:
    """Give a PR's static warnings on changed lines, for the candidate reviewer's request.

    Args:
        stored (StoredStatic): The PR's stored static run.

    Returns:
        StaticInput: Its status and warnings.
    """
    result = stored.load()
    if stored.status != "ok" or result is None:
        return StaticInput(stored.status)
    return StaticInput("ok", tuple(result.kept))


def _candidate_settings(
    args: argparse.Namespace, profile: ModelProfile, config: SandboxConfig
) -> tuple[dict[str, Any], int, str, str]:
    """Give the candidate reviewer's settings, its budget and the keys of its inputs.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `variant`.
        profile (ModelProfile): The reviewer's model profile.
        config (SandboxConfig): The sandbox settings.

    Returns:
        tuple[dict[str, Any], int, str, str]: The settings to add to the run's, the budget of
        the user message in characters, the retrieval key and the static key.
    """
    variant = args.variant
    budget = candidate_reviewer.candidate_budget_chars(profile, candidate_reviewer.PROMPTS[variant])
    retrieval, indexer, retrieval_key, _ = retrieval_setup(config, profile)
    static, tools, static_key = static_setup(config)
    settings = {
        "input_budget_chars": budget,
        "max_findings": candidate_reviewer.MAX_CANDIDATES,
        "prompt_sha256": candidate_reviewer.prompt_fingerprint(variant),
        "variant": variant,
        "lookups": {
            "max_lookups": candidate_reviewer.MAX_LOOKUPS,
            "min_room": candidate_reviewer.MIN_LOOKUP_ROOM,
            "room": candidate_reviewer.LOOKUP_ROOM,
            **asdict(LookupLimits()),
        },
        "context": {
            **asdict(retrieval),
            "settings_key": retrieval_key,
            "indexer_image": indexer,
            "probe_version": PROBE_VERSION,
            "order": list(KIND_ORDER),
            "min_context_chars": MIN_CONTEXT_CHARS,
        },
        "static": {**asdict(static), "settings_key": static_key, "tools_image": tools},
    }
    return settings, budget, retrieval_key, static_key


def _review(args: argparse.Namespace, reviewer: str) -> int:
    """Review the chosen dev PRs with a reviewer, and write the run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `since`, `ids`, `limit`,
            `with_issues` and `data`, and `variant` for the candidate reviewer.
        reviewer (str): "naive", "context" or "candidates".

    Returns:
        int: Exit code, 0 when every PR in the run has a review, ready to score.
    """
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    in_scope = [pr for pr in chosen_prs(instances, split, args) if pr.year >= args.since]
    if args.with_issues:
        in_scope = [pr for pr in in_scope if pr.changes]
    if args.usable_only:
        with connect() as conn:
            usable = usable_prs(conn)
        in_scope = [pr for pr in in_scope if pr.instance_id in usable]
    chosen = _chosen(args, in_scope)
    if chosen is None:
        return 1
    models = load_models_config()
    model = check_reviewer(REVIEWER_PROFILE, models)
    profile = models.profile(REVIEWER_PROFILE)
    budget = input_budget_chars(profile)
    folder = results_folder(args.name)
    settings: dict[str, Any] = {
        "name": args.name,
        "reviewer": reviewer,
        "split": args.split,
        "since": args.since,
        "profile": REVIEWER_PROFILE,
        "provider": profile.provider,
        "model": model,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "reasoning_effort": profile.reasoning_effort,
        "tokens_per_minute": profile.limits.tokens_per_minute,
        "input_budget_chars": budget,
        "max_findings": MAX_FINDINGS,
        "max_tries": MAX_TRIES,
        "prompt_sha256": prompt_fingerprint(),
        "dataset_sha256": DATASET.sha256,
        "split_sha256": SPLIT_SHA256,
    }
    if args.with_issues:
        # Only set when used, so earlier runs keep their settings (and thread IDs).
        settings["with_issues"] = True
    if args.usable_only:
        settings["usable_only"] = True
    key = static_key = ""
    sandbox_config = load_sandbox_config()
    if reviewer == "context":
        retrieval, tag, key, _ = retrieval_setup(sandbox_config, profile)
        settings["prompt_sha256"] = context_prompt_fingerprint()
        settings["context"] = {
            **asdict(retrieval),
            "settings_key": key,
            "indexer_image": tag,
            "probe_version": PROBE_VERSION,
            "order": list(KIND_ORDER),
            "min_context_chars": MIN_CONTEXT_CHARS,
        }
    elif reviewer == "candidates":
        extra, budget, key, static_key = _candidate_settings(args, profile, sandbox_config)
        settings.update(extra)
    # As saved in the database (tuples become lists), so a rerun compares equal.
    settings = json.loads(json.dumps(settings))
    # Part of every thread ID, so a saved review made with other settings is never resumed.
    settings_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    stopped = ""
    with connect() as conn, open_checkpointer() as checkpointer:
        save_bench_prs(conn, instances, split)
        contexts: dict[str, Path] = {}
        statics: dict[str, StoredStatic] = {}
        if reviewer in ("context", "candidates"):
            contexts = stored_contexts(conn, key)
            missing = [pr.instance_id for pr in chosen if pr.instance_id not in contexts]
            if missing:
                print(
                    f"{len(missing)} PRs have no context yet, such as {missing[0]}. Build it "
                    f"first with: pr-reviewer context build --since {args.since}",
                    file=sys.stderr,
                )
                return 1
        if reviewer == "candidates":
            statics = stored_static(conn, static_key)
            missing = [pr.instance_id for pr in chosen if pr.instance_id not in statics]
            if missing:
                print(
                    f"{len(missing)} PRs have no static run yet, such as {missing[0]}. Run the "
                    f"tools first with: pr-reviewer static run --since {args.since}",
                    file=sys.stderr,
                )
                return 1
        start_run(
            conn,
            args.name,
            reviewer=reviewer,
            profile=REVIEWER_PROFILE,
            model=model,
            split=args.split,
            settings=settings,
        )
        status = review_status(conn, args.name)
        todo = [pr for pr in chosen if status.get(pr.instance_id, ("", 0))[0] != "done"]
        if args.limit:
            todo = todo[: args.limit]
        llm = LLMClient(
            REVIEWER_PROFILE, models, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        which = "with known issues " if args.with_issues else ""
        print(
            f"Reviewing {len(todo)} of the {len(in_scope)} {args.split} PRs {which}from "
            f"{args.since} on with {REVIEWER_PROFILE} (run {args.name}, {reviewer} reviewer, "
            f"up to {budget:,} characters of request text)"
        )
        lookups = LookupRunner(sandbox_config, LookupLimits())
        for number, pr in enumerate(todo, start=1):
            print(f"[{number}/{len(todo)}] {pr.instance_id}", flush=True)
            failures = status.get(pr.instance_id, ("", 0))[1]
            thread = f"{args.name}:{settings_key}:{pr.instance_id}:{failures}"
            show = lambda line: print(f"  {line}", flush=True)  # noqa: E731
            try:
                result: PrReview
                if reviewer == "candidates":
                    result = candidate_reviewer.review_candidates(
                        llm,
                        checkpointer,
                        pr,
                        thread,
                        sample=failures,
                        budget_chars=budget,
                        variant=args.variant,
                        static=_static_input(statics[pr.instance_id]),
                        pieces=load_candidates(contexts[pr.instance_id]).pieces,
                        explorer=lookups.for_pr(pr),
                        on_step=show,
                    )
                else:
                    retrieve: RetrieveNode | None = None
                    if reviewer == "context":
                        pieces = load_candidates(contexts[pr.instance_id]).pieces
                        retrieve = retrieve_node(pr, budget, pieces)
                    result = review_pr(
                        llm,
                        checkpointer,
                        pr,
                        thread,
                        sample=failures,
                        budget_chars=budget,
                        on_step=show,
                        retrieve=retrieve,
                        run_name=f"{reviewer}-review",
                    )
            except (LimitError, SandboxError) as exc:
                stopped = str(exc)
                break
            save_review(conn, args.name, result, failures + (0 if result.ok else 1))
            outcome = f"{len(result.findings)} findings" if result.ok else "failed"
            print(f"  => {outcome}; {result.tries} tries", flush=True)
        reviews = load_reviews(conn, args.name)
        findings = load_findings(conn, args.name)
    settings_file = {
        **settings,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "prs": len(in_scope),
    }
    predictions = write_review_files(folder, settings_file, in_scope, reviews, findings)
    if stopped:
        print(
            f"Stopped: {stopped}\nRun the same command again later: finished PRs are kept, "
            "and a stopped review continues where it stopped.",
            file=sys.stderr,
        )
        return 1
    if predictions is None:
        failed = sum(review.status == "failed" for review in reviews.values())
        waiting = len(in_scope) - sum(r.status == "done" for r in reviews.values())
        print(
            f"{waiting} PRs have no review yet ({failed} failed). Run the same command again: "
            f"failed PRs get fresh replies. Summary: {folder / 'review_summary.md'}"
        )
        return 1
    print(f"All {len(in_scope)} PRs reviewed. Saved {folder} (review_summary.md)")
    subset = " --with-issues" if args.with_issues else ""
    if args.split == "test":
        subset += " --split test --unlock-test-set"
    if args.usable_only:
        subset += " --usable-only"
    print(
        "Score it with:\n"
        f"  pr-reviewer score {predictions.relative_to(folder.parent.parent).as_posix()} "
        f"--name {args.name} --reviewer {REVIEWER_PROFILE} --since {args.since}{subset}"
    )
    return 0


def _subsets(conn: Any, run: str) -> list[tuple[str, set[str]]]:
    """Choose the subsets of a run's PRs worth comparing on, from its saved reviews.

    Args:
        conn (Any): Database connection.
        run (str): The run's name.

    Returns:
        list[tuple[str, set[str]]]: (label, PRs): with and without context, and for the
        candidate reviewer, with and without lookups.
    """
    rows = conn.execute(
        """
        SELECT instance_id, COALESCE(context_pieces, 0) > 0,
               jsonb_array_length(COALESCE(details -> 'candidates' -> 'lookups', '[]'::jsonb)) > 0,
               details ? 'candidates'
        FROM pr_reviews WHERE run = %s
        """,
        (run,),
    ).fetchall()
    ids = {row[0] for row in rows}
    context = {row[0] for row in rows if row[1]}
    subsets = [("with context", context), ("without context", ids - context)]
    if any(row[3] for row in rows):
        looked = {row[0] for row in rows if row[2]}
        subsets += [("with lookups", looked), ("without lookups", ids - looked)]
    return subsets


def _compare(args: argparse.Namespace) -> int:
    """Compare two scored runs PR by PR, and write comparison.md in the new run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `base` and `new`.

    Returns:
        int: Exit code, 0 on success.
    """
    new = load_judgments(results_folder(args.new))
    base = same_prs(load_judgments(results_folder(args.base)), new)
    with connect() as conn:
        subsets = _subsets(conn, args.new)
    text = comparison_markdown(args.base, args.new, base, new, subsets)
    path = results_folder(args.new) / "comparison.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"Saved {path}")
    return 0


def _add_run_options(parser: argparse.ArgumentParser, default_name: str) -> None:
    """Add the options every review command takes.

    Args:
        parser (argparse.ArgumentParser): The command's parser.
        default_name (str): The run's default name.
    """
    parser.add_argument(
        "--name",
        default=default_name,
        help=f"the run's name and results folder (default: {default_name})",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=FIRST_YEAR,
        help=f"only dev PRs from this year on (default: {FIRST_YEAR})",
    )
    parser.add_argument("--ids", nargs="+", default=None, help="only these PRs of the run")
    parser.add_argument("--limit", type=int, default=None, help="at most this many PRs this time")
    parser.add_argument(
        "--usable-only",
        action="store_true",
        help="only the PRs whose environment setup was verified (setup failures are reported "
        "apart)",
    )
    add_split_arguments(parser)
    parser.add_argument(
        "--with-issues",
        action="store_true",
        help="the run holds only the PRs with known issues (recall depends on them alone)",
    )


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer review ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    review = commands.add_parser("review", help="review dev PRs, ready for scoring")
    sub = review.add_subparsers(metavar="REVIEW_COMMAND", required=True)
    naive = sub.add_parser(
        "naive",
        parents=[data_option],
        help="one model call per PR, from the diff alone (the baseline)",
    )
    _add_run_options(naive, DEFAULT_NAME)
    naive.set_defaults(handler=_naive)
    context = sub.add_parser(
        "context",
        parents=[data_option],
        help="the naive reviewer plus code around the change (needs `context build`)",
    )
    _add_run_options(context, CONTEXT_NAME)
    context.set_defaults(handler=_context)
    candidates = sub.add_parser(
        "candidates",
        parents=[data_option],
        help="list every plausible problem, with static warnings and lookups (needs `context "
        "build` and `static run`)",
    )
    _add_run_options(candidates, CANDIDATES_NAME)
    candidates.add_argument(
        "--variant",
        choices=sorted(candidate_reviewer.PROMPTS),
        default=DEFAULT_VARIANT,
        help=f"the prompt variant (default: {DEFAULT_VARIANT})",
    )
    candidates.set_defaults(handler=_candidates)
    compare = sub.add_parser(
        "compare", help="compare two scored runs on the same PRs, PR by PR (no model calls)"
    )
    compare.add_argument(
        "--base", default=DEFAULT_NAME, help=f"the run to compare with (default: {DEFAULT_NAME})"
    )
    compare.add_argument(
        "--new",
        default=CONTEXT_NAME,
        help=f"the run to measure (default: {CONTEXT_NAME}); it may cover some of the base "
        "run's PRs, which are then compared on those",
    )
    compare.set_defaults(handler=_compare)
