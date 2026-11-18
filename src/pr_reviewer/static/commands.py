"""The `pr-reviewer static ...` command: run the static tools on the dev PRs' changed files.

cli.py adds it to the `pr-reviewer` command. Only dev PRs are used; the test set stays locked
until the final run.

How to run (inside the conda env, from the project folder, with Docker Desktop and the
database running):
    pr-reviewer static run                    run ruff, mypy, bandit and semgrep on the dev PRs
                                              from 2022 on, each in its PR's built environment
                                              (Docker only, no model calls; a PR without
                                              prepared images is prepared first, which can take
                                              minutes)
    pr-reviewer static run --ids <id> ...     only these PRs
    pr-reviewer static run --limit 5          at most 5 PRs this time
    pr-reviewer static run --again            run again on PRs already stored
Running it again only runs PRs not stored yet for the current tools and rules. It writes
results/static-dev-<year>/ (config.json, summary.md). `--split test --unlock-test-set` runs it
on the test set for the final run (results/static-test-<year>/). The candidate reviewer
(`pr-reviewer review candidates`) needs every PR stored first.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.split import split_options
from pr_reviewer.context.commands import dev_scope
from pr_reviewer.db.connection import connect
from pr_reviewer.difftest.environments import PrepareError, pr_environment, ready_prepared
from pr_reviewer.reviewer.commands import FIRST_YEAR
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.repos import RepoError
from pr_reviewer.sandbox.runner import SandboxError, docker_client
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.static.store import StoredStatic, save_failure, save_static, stored_static
from pr_reviewer.static.tools import (
    VERSION,
    ReadyTools,
    StaticError,
    StaticResult,
    StaticSettings,
    ready_tools,
    result_path,
    run_static,
    static_setup,
)
from pr_reviewer.static.warnings import TOOLS


def describe(result: StaticResult) -> str:
    """Say in one line what the tools found on a PR.

    Args:
        result (StaticResult): The result.

    Returns:
        str: Changed files, warnings kept per tool, and any tool that didn't finish.
    """
    if not result.changed:
        return "no Python line changed; no tool run"
    kept = Counter(w.tool for w in result.kept)
    parts = [f"{tool} {kept[tool]}" for tool in TOOLS]
    problems = [f"{run.tool} {run.status}" for run in result.runs if run.status != "ok"]
    mypy = next((run.mode for run in result.runs if run.tool == "mypy"), "")
    text = (
        f"{len(result.changed)} files; on changed lines: {', '.join(parts)} (of "
        f"{len(result.warnings)} on the files); mypy imports {mypy or 'n/a'}; "
        f"{result.seconds:.0f} s"
    )
    return text + (f"; not finished: {', '.join(problems)}" if problems else "")


def summary_markdown(
    settings: dict[str, Any],
    prs: list[Instance],
    stored: dict[str, StoredStatic],
    results: dict[str, StaticResult],
) -> str:
    """Describe the static runs of the dev PRs, as Markdown.

    Args:
        settings (dict[str, Any]): The run's settings, as in config.json.
        prs (list[Instance]): The PRs in scope.
        stored (dict[str, StoredStatic]): The stored runs, by PR.
        results (dict[str, StaticResult]): The full results of the runs that ran, by PR.

    Returns:
        str: The text of summary.md.
    """
    ran = [results[pr.instance_id] for pr in prs if pr.instance_id in results]
    failed = [
        stored[pr.instance_id]
        for pr in prs
        if pr.instance_id in stored and stored[pr.instance_id].status == "failed"
    ]
    no_python = [r for r in ran if not r.changed]
    status: Counter[tuple[str, str]] = Counter()
    modes: Counter[str] = Counter()
    for result in ran:
        for run in result.runs:
            status[(run.tool, run.status)] += 1
            if run.tool == "mypy":
                modes[run.mode] += 1
    on_files = Counter(w.tool for r in ran for w in r.warnings)
    kept = Counter(w.tool for r in ran for w in r.kept)
    prs_with = Counter(tool for r in ran for tool in {w.tool for w in r.kept})
    rules = Counter((w.tool, w.rule) for r in ran for w in r.kept)
    lines = [
        f"# Static tools: {settings.get('split', 'dev')} PRs from {settings['since']} on",
        "",
        f"- PRs: {len(prs)}; tools ran on {len(ran) - len(no_python)}; no Python line changed "
        f"{len(no_python)}; failed {len(failed)}; not run yet "
        f"{len(prs) - len(ran) - len(failed)}",
        f"- Tools image: {settings['tools_image']}",
        "- Versions: " + "; ".join(settings["versions"].splitlines()),
        f"- Rules: ruff {settings['ruff_select']}; mypy defaults "
        f"({' '.join(settings['mypy_flags'])}); bandit severity {settings['bandit_severity']} "
        f"and up; semgrep {settings['semgrep_rules']} rules for Python. Repo configs ignored.",
        f"- Each tool runs at most {settings['tool_minutes']:g} minutes per PR, network off, in "
        "the PR's prepared image at its last commit",
        "",
        "## Per tool",
        "",
        "| Tool | Finished | Timed out | Out of memory | Failed | Warnings on changed files "
        "| On changed lines | PRs with one |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tool in TOOLS:
        lines.append(
            f"| {tool} | {status[(tool, 'ok')]} | {status[(tool, 'timeout')]} | "
            f"{status[(tool, 'out_of_memory')]} | {status[(tool, 'failed')]} | {on_files[tool]} | "
            f"{kept[tool]} | {prs_with[tool]} |"
        )
    lines += [
        "",
        f"mypy followed imports silently on {modes['silent']} PRs and checked the changed files "
        f"alone on {modes['skip']} (following imports ran out of time or memory).",
        "",
        "## Rules on changed lines",
        "",
        "| Tool | Rule | Warnings |",
        "|---|---|---|",
    ]
    lines += [f"| {tool} | {rule} | {count} |" for (tool, rule), count in rules.most_common(30)]
    lines += [
        "",
        "## Per PR",
        "",
        "| PR | Files | " + " | ".join(TOOLS) + " | On the files | mypy imports | Seconds |",
        "|---|---|" + "---|" * len(TOOLS) + "---|---|---|",
    ]
    for result in ran:
        counts = Counter(w.tool for w in result.kept)
        mypy = next((run.mode for run in result.runs if run.tool == "mypy"), "")
        lines.append(
            f"| {result.instance_id} | {len(result.changed)} | "
            + " | ".join(str(counts[tool]) for tool in TOOLS)
            + f" | {len(result.warnings)} | {mypy or 'n/a'} | {result.seconds:.0f} |"
        )
    if failed:
        lines += ["", "## Failed", ""]
        lines += [f"- {item.instance_id}: {item.detail.splitlines()[0][:300]}" for item in failed]
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    """Write a text file with LF line endings.

    Args:
        path (Path): The file. Its folder is created if needed.
        text (str): The text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


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


def _run_one(
    ctx: TryContext, tools: ReadyTools, pr: Instance, settings: StaticSettings, conn: Any, key: str
) -> str:
    """Run the tools on one PR and store the result, or record why they couldn't run.

    Args:
        ctx (TryContext): Docker, the sandbox settings, the sandbox image and the log folder.
        tools (ReadyTools): The tools.
        pr (Instance): The PR.
        settings (StaticSettings): The rules.
        conn (Any): Database connection.
        key (str): The static settings key.

    Returns:
        str: A line saying what happened.

    Raises:
        SandboxError: If Docker fails (nothing is recorded, so the PR is tried again later).
    """
    try:
        env = pr_environment(conn, pr)
        images = ready_prepared(ctx, pr, env, on_step=lambda line: print(f"  {line}", flush=True))
        result = run_static(
            ctx.client,
            ctx.config,
            tools,
            pr,
            images["pr"],
            env.recipe.python_version,
            settings,
            ctx.log_dir,
        )
    except (PrepareError, RepoError, StaticError) as exc:
        save_failure(conn, key, pr.instance_id, str(exc))
        return f"failed: {exc}"
    save_static(conn, key, result_path(key, pr.instance_id), result)
    return describe(result)


def _run(args: argparse.Namespace) -> int:
    """Run the static tools on the chosen dev PRs, and write the summary.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `since`, `ids`, `limit`, `again`
            and `data`.

    Returns:
        int: Exit code, 0 when every PR in scope has a stored run.
    """
    instances, in_scope, split = dev_scope(args.data, args.since, args.split, args.unlock_test_set)
    chosen = _chosen(args.ids, in_scope, args.since)
    if chosen is None:
        return 1
    config = load_sandbox_config()
    client = docker_client()
    ctx = TryContext(client, config, ready_image(client, config), logs_dir())
    tools = ready_tools(client, config)
    settings, _, key = static_setup(config)
    with connect() as conn:
        save_bench_prs(conn, instances, split)
        stored = stored_static(conn, key)
        todo = [pr for pr in chosen if args.again or pr.instance_id not in stored]
        if args.limit:
            todo = todo[: args.limit]
        print(
            f"Running the static tools on {len(todo)} of the {len(in_scope)} {args.split} PRs from "
            f"{args.since} on ({tools.image}, settings {key[:12]})",
            flush=True,
        )
        stopped = ""
        for number, pr in enumerate(todo, start=1):
            print(f"[{number}/{len(todo)}] {pr.instance_id}", flush=True)
            try:
                line = _run_one(ctx, tools, pr, settings, conn, key)
            except SandboxError as exc:
                stopped = str(exc)
                print(f"  => Docker failed, stopping: {exc}", flush=True)
                break
            print(f"  => {line}", flush=True)
        stored = stored_static(conn, key)
    scope = {pr.instance_id for pr in in_scope}
    results = {
        i: result
        for i, item in stored.items()
        if i in scope and (result := item.load()) is not None
    }
    run_settings = {
        **asdict(settings),
        "split": args.split,
        "since": args.since,
        "tools_image": tools.image,
        "tools_volume": tools.volume,
        "versions": tools.versions,
        "static_version": VERSION,
        "settings_key": key,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    folder = results_folder(f"static-{args.split}-{args.since}")
    _write(folder / "config.json", json.dumps(run_settings, indent=2) + "\n")
    _write(folder / "summary.md", summary_markdown(run_settings, in_scope, stored, results))
    waiting = len(scope - set(stored))
    print(f"{len(scope) - waiting} of {len(scope)} PRs have a stored run. Saved {folder}")
    if stopped:
        return 1
    return 0 if waiting == 0 else 1


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer static ...` and its subcommand to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    static = commands.add_parser("static", help="static tools on each PR's changed lines")
    sub = static.add_subparsers(metavar="STATIC_COMMAND", required=True)
    run = sub.add_parser(
        "run",
        parents=[data_option, split_options()],
        help="run ruff, mypy, bandit and semgrep (Docker)",
    )
    run.add_argument(
        "--since",
        type=int,
        default=FIRST_YEAR,
        help=f"only dev PRs from this year on (default: {FIRST_YEAR})",
    )
    run.add_argument("--ids", nargs="+", default=None, help="only these PRs")
    run.add_argument("--limit", type=int, default=None, help="at most this many PRs this time")
    run.add_argument("--again", action="store_true", help="run again on PRs already stored")
    run.set_defaults(handler=_run)
