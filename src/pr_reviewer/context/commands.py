"""The `pr-reviewer context ...` commands: retrieve the dev PRs' context, and check it.

cli.py adds them to the `pr-reviewer` command. Only dev PRs are used; the test set stays
locked until the final run.

How to run (inside the conda env, from the project folder, with Docker Desktop and the
database running):
    pr-reviewer context build                 find the candidate context of the dev PRs from
                                              2022 on, one indexer sandbox per PR (Docker;
                                              about a minute or two per PR, no model calls)
    pr-reviewer context build --ids <id> ...  only these PRs
    pr-reviewer context build --limit 5       at most 5 PRs this time
    pr-reviewer context build --again         retrieve again PRs already stored
    pr-reviewer context check                 the retrieval check, for the run
                                              context-gpt-oss-120b: did the model see the code
                                              each known issue's fix commit changed?
Running build again only retrieves PRs not stored yet for the current settings. The review
itself is `pr-reviewer review context`, which needs every PR's context built first.
"""

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, load_split, split_path, split_prs
from pr_reviewer.context.candidates import KIND_ORDER, Candidates, FixFunctions, load_candidates
from pr_reviewer.context.check import IssueCheck, check_pr, summary_markdown
from pr_reviewer.context.probe import VERSION as PROBE_VERSION
from pr_reviewer.context.retrieve import (
    ContextError,
    candidates_path,
    fix_functions,
    ready_indexer,
    retrieval_setup,
    retrieve_candidates,
)
from pr_reviewer.context.store import save_context, stored_contexts
from pr_reviewer.core.config import load_models_config
from pr_reviewer.db.connection import connect
from pr_reviewer.reviewer.commands import CONTEXT_NAME, FIRST_YEAR, REVIEWER_PROFILE
from pr_reviewer.reviewer.contextual import build_context_messages
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.repos import RepoError
from pr_reviewer.sandbox.runner import SandboxError, docker_client

# What fix commits changed, cached per PR next to the candidates.
_FIXES = TypeAdapter(dict[str, FixFunctions])


def dev_scope(data: Path | None, since: int) -> tuple[list[Instance], list[Instance], Any]:
    """Load the dataset and the dev PRs from a year on.

    Args:
        data (Path | None): The dataset file, or None for the default one.
        since (int): The first year.

    Returns:
        tuple[list[Instance], list[Instance], Any]: Every PR, the dev PRs in scope, and the
        split.
    """
    instances = load_dataset(data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    in_scope = [pr for pr in split_prs(instances, split, "dev") if pr.year >= since]
    return instances, in_scope, split


def _chosen(ids: list[str] | None, in_scope: list[Instance], since: int) -> list[Instance] | None:
    """Pick the PRs asked for.

    Args:
        ids (list[str] | None): The IDs asked for, or None for all.
        in_scope (list[Instance]): The PRs in scope.
        since (int): The first year, for messages.

    Returns:
        list[Instance] | None: The PRs, or None (after a message) when one isn't in scope.
    """
    if not ids:
        return in_scope
    chosen = [pr for pr in in_scope if pr.instance_id in set(ids)]
    if len(chosen) != len(set(ids)):
        missing = sorted(set(ids) - {pr.instance_id for pr in chosen})
        print(f"Not dev PRs from {since} on (or unknown): {', '.join(missing)}", file=sys.stderr)
        return None
    return chosen


def describe(candidates: Candidates) -> str:
    """Say in one line what the probe found for a PR.

    Args:
        candidates (Candidates): The candidates.

    Returns:
        str: Pieces by kind, how calls were resolved, and the time taken.
    """
    counts = candidates.counts()
    stats = candidates.stats
    pieces = ", ".join(f"{counts[kind]} {kind}" for kind in KIND_ORDER)
    return (
        f"{len(candidates.changed)} changed functions; pieces: {pieces}; calls: "
        f"{stats.get('resolved_ast', 0)} ast, {stats.get('resolved_jedi', 0)} Jedi, "
        f"{stats.get('unresolved', 0)} unresolved, {stats.get('external', 0)} outside the "
        f"repo; {candidates.seconds:.0f} s"
    )


def build_summary(
    settings: dict[str, Any], prs: list[Instance], found: dict[str, Candidates]
) -> str:
    """Describe the retrieved context of the dev PRs, as Markdown.

    Args:
        settings (dict[str, Any]): The retrieval settings.
        prs (list[Instance]): The PRs in scope.
        found (dict[str, Candidates]): The candidates stored, by PR.

    Returns:
        str: The text of summary.md.
    """
    rows = [found[pr.instance_id] for pr in prs if pr.instance_id in found]
    total: dict[str, int] = {}
    for candidates in rows:
        for key, value in candidates.stats.items():
            total[key] = total.get(key, 0) + value
    calls = total.get("calls", 0) or 1
    seconds = [c.seconds for c in rows]
    lines = [
        f"# Context retrieval: dev PRs from {settings['since']} on",
        "",
        f"- PRs: {len(prs)}; retrieved: {len(rows)}",
        f"- Indexer image: {settings['indexer_image']}; probe version {settings['probe_version']}",
        f"- Caps: {settings['callers']} callers and {settings['tests']} tests per changed "
        f"function; called definitions, callers and tests up to {settings['short_lines']} "
        f"lines shown whole; pieces over {settings['max_piece_chars']:,} characters skipped",
        f"- Time per PR (probe only): median {statistics.median(seconds):.0f} s, "
        f"longest {max(seconds):.0f} s"
        if seconds
        else "- No PR retrieved yet",
        "",
        "## Calls in the changed functions",
        "",
        f"- {total.get('calls', 0):,} calls: {total.get('resolved_ast', 0):,} resolved by ast "
        f"({total.get('resolved_ast', 0) / calls:.0%}), {total.get('resolved_jedi', 0):,} by "
        f"Jedi ({total.get('resolved_jedi', 0) / calls:.0%}), "
        f"{total.get('unresolved', 0):,} unresolved ({total.get('unresolved', 0) / calls:.0%}), "
        f"{total.get('external', 0):,} outside the repo ({total.get('external', 0) / calls:.0%})",
        f"- Jedi: {total.get('jedi_calls', 0):,} questions, {total.get('jedi_timeouts', 0)} "
        f"timed out, {total.get('jedi_errors', 0)} failed",
        f"- Callers found {total.get('callers_found', 0):,}, "
        f"kept {total.get('callers_kept', 0):,}; "
        f"tests found {total.get('tests_found', 0):,}, "
        f"kept {total.get('tests_kept', 0):,}; "
        f"uses refused as ambiguous {total.get('ambiguous_skipped', 0):,}",
        "",
        "## Per PR",
        "",
        "| PR | Changed functions | " + " | ".join(KIND_ORDER) + " | ast | Jedi | "
        "unresolved | Seconds |",
        "|---|---|" + "---|" * len(KIND_ORDER) + "---|---|---|---|",
    ]
    for candidates in rows:
        counts = candidates.counts()
        stats = candidates.stats
        lines.append(
            f"| {candidates.instance_id} | {len(candidates.changed)} | "
            + " | ".join(str(counts[kind]) for kind in KIND_ORDER)
            + f" | {stats.get('resolved_ast', 0)} | {stats.get('resolved_jedi', 0)} | "
            f"{stats.get('unresolved', 0)} | {candidates.seconds:.0f} |"
        )
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    """Write a text file with LF line endings.

    Args:
        path (Path): The file. Its folder is created if needed.
        text (str): The text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _build(args: argparse.Namespace) -> int:
    """Retrieve the candidate context of the chosen dev PRs, and write the summary.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `since`, `ids`, `limit`, `again`
            and `data`.

    Returns:
        int: Exit code, 0 when every PR in scope has its context.
    """
    instances, in_scope, split = dev_scope(args.data, args.since)
    chosen = _chosen(args.ids, in_scope, args.since)
    if chosen is None:
        return 1
    config = load_sandbox_config()
    profile = load_models_config().profile(REVIEWER_PROFILE)
    settings, _, key, _ = retrieval_setup(config, profile)
    client = docker_client()
    ready_image(client, config)
    tag = ready_indexer(client, config)
    failed: list[str] = []
    with connect() as conn:
        save_bench_prs(conn, instances, split)
        stored = stored_contexts(conn, key)
        todo = [pr for pr in chosen if args.again or pr.instance_id not in stored]
        if args.limit:
            todo = todo[: args.limit]
        print(
            f"Retrieving context for {len(todo)} of the {len(in_scope)} dev PRs from "
            f"{args.since} on ({tag}, settings {key[:12]})",
            flush=True,
        )
        for number, pr in enumerate(todo, start=1):
            print(f"[{number}/{len(todo)}] {pr.instance_id}", flush=True)
            try:
                candidates = retrieve_candidates(client, config, tag, pr, settings, logs_dir())
            except (ContextError, RepoError, SandboxError) as exc:
                failed.append(pr.instance_id)
                print(f"  => failed: {exc}", flush=True)
                continue
            save_context(conn, key, candidates_path(key, pr.instance_id), candidates)
            print(f"  => {describe(candidates)}", flush=True)
        stored = stored_contexts(conn, key)
    scope_ids = {pr.instance_id for pr in in_scope}
    found = {i: load_candidates(path) for i, path in stored.items() if i in scope_ids}
    folder = results_folder(f"context-dev-{args.since}")
    run_settings = {
        **asdict(settings),
        "since": args.since,
        "indexer_image": tag,
        "probe_version": PROBE_VERSION,
        "settings_key": key,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    _write(folder / "config.json", json.dumps(run_settings, indent=2) + "\n")
    _write(folder / "summary.md", build_summary(run_settings, in_scope, found))
    waiting = len(in_scope) - len(found)
    if failed:
        print(f"{len(failed)} PRs failed: {', '.join(failed)}", file=sys.stderr)
    print(f"{len(found)} of {len(in_scope)} PRs have their context. Saved {folder} (summary.md)")
    return 0 if waiting == 0 else 1


def _fixes_path(key: str, instance_id: str) -> Path:
    """Give the file caching what a PR's fix commits changed.

    Args:
        key (str): The retrieval settings key.
        instance_id (str): The PR.

    Returns:
        Path: The file, next to the PR's candidates.
    """
    return candidates_path(key, instance_id).parent / "fixes" / f"{instance_id}.json"


def _check(args: argparse.Namespace) -> int:
    """Run the retrieval check for the dev PRs with known issues, and write its results.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `since` and `data`.

    Returns:
        int: Exit code, 0 when every PR could be checked.
    """
    _, in_scope, _ = dev_scope(args.data, args.since)
    with_issues = [pr for pr in in_scope if pr.changes]
    config = load_sandbox_config()
    profile = load_models_config().profile(REVIEWER_PROFILE)
    _, _, key, budget = retrieval_setup(config, profile)
    client = None
    checks: list[IssueCheck] = []
    with connect() as conn:
        stored = stored_contexts(conn, key)
    missing = [pr.instance_id for pr in with_issues if pr.instance_id not in stored]
    if missing:
        print(
            f"{len(missing)} PRs have no context yet, such as {missing[0]}. Run first: "
            f"pr-reviewer context build --since {args.since}",
            file=sys.stderr,
        )
        return 1
    print(f"Checking {len(with_issues)} dev PRs with known issues from {args.since} on")
    for pr in with_issues:
        candidates = load_candidates(stored[pr.instance_id])
        commits = sorted(
            {
                issue.change_resolve_info.commit_sha
                for issue in pr.changes
                if issue.change_resolve_info is not None
            }
        )
        cache = _fixes_path(key, pr.instance_id)
        fixes: dict[str, FixFunctions] = {}
        if cache.is_file():
            fixes = _FIXES.validate_json(cache.read_text(encoding="utf-8"))
        if set(commits) - set(fixes):
            if client is None:
                client = docker_client()
                ready_image(client, config)
            tag = ready_indexer(client, config)
            fixes = fix_functions(client, config, tag, pr, commits, logs_dir())
            _write(cache, _FIXES.dump_json(fixes).decode("utf-8"))
        _, fitted, packed = build_context_messages(pr, budget, candidates.pieces)
        found = check_pr(pr, candidates, fitted, packed, fixes)
        checks.extend(found)
        print(f"  {pr.instance_id}: " + ", ".join(c.status for c in found), flush=True)
    folder = results_folder(args.name)
    _write(folder / "retrieval_check.md", summary_markdown(checks, args.name, len(with_issues)))
    raw = [asdict(check) for check in checks]
    _write(folder / "raw" / "retrieval_check.json", json.dumps(raw, indent=2) + "\n")
    print(f"Saved {folder / 'retrieval_check.md'}")
    return 0


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer context ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    context = commands.add_parser("context", help="the code around each PR's change (RAG)")
    sub = context.add_subparsers(metavar="CONTEXT_COMMAND", required=True)
    build = sub.add_parser(
        "build", parents=[data_option], help="find each dev PR's candidate context (Docker)"
    )
    build.add_argument(
        "--since",
        type=int,
        default=FIRST_YEAR,
        help=f"only dev PRs from this year on (default: {FIRST_YEAR})",
    )
    build.add_argument("--ids", nargs="+", default=None, help="only these PRs")
    build.add_argument("--limit", type=int, default=None, help="at most this many PRs this time")
    build.add_argument("--again", action="store_true", help="retrieve again PRs already stored")
    build.set_defaults(handler=_build)
    check = sub.add_parser(
        "check",
        parents=[data_option],
        help="did the model see the code each known issue's fix changed? (no model calls)",
    )
    check.add_argument(
        "--name",
        default=CONTEXT_NAME,
        help=f"the review run whose folder gets the results (default: {CONTEXT_NAME})",
    )
    check.add_argument(
        "--since",
        type=int,
        default=FIRST_YEAR,
        help=f"only dev PRs from this year on (default: {FIRST_YEAR})",
    )
    check.set_defaults(handler=_check)
