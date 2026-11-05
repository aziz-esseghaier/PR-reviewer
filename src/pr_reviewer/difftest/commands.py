"""The `pr-reviewer difftest ...` commands: prepare PR environments, run tests, validate.

cli.py adds them to the `pr-reviewer` command. They work on dev PRs whose setup passed
(`pr-reviewer setup run`); test-set PRs are refused.

How to run (inside the conda env, from the project folder, with Docker Desktop and the
database running):
    pr-reviewer difftest prepare [--ids <id> ...]
        install each set-up dev PR at its base and PR state, and save both as images
    pr-reviewer difftest run <id> <test> [<test> ...] [--from base|pr] [--again]
        run tests (pytest node IDs, files or Django labels) at both sides and compare; the
        test files are taken as they are at the PR state (or --from base)
    pr-reviewer difftest run <id> --file my_test.py --path pkg/tests/test_new.py [<test> ...]
        the same with a test file of your own, written at that path at both sides
    pr-reviewer difftest validate [--ids <id> ...] [--again]
        check the runner on known tests; writes results/difftest-validation/
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, load_split, split_path, split_prs
from pr_reviewer.db.connection import connect
from pr_reviewer.difftest.environments import (
    PrepareError,
    find_prepared,
    pr_environment,
    prepare,
    ready_prepared,
)
from pr_reviewer.difftest.outcomes import LABELS, DiffOutcome, Outcome, SideResult
from pr_reviewer.difftest.runner import (
    RUNNER_VERSION,
    RUNS_PER_SIDE,
    FileSpec,
    SideSpec,
    runner_kind,
)
from pr_reviewer.difftest.store import diff_with_store
from pr_reviewer.difftest.validation import run_validation, write_validation
from pr_reviewer.sandbox.checkout import base_of
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.repos import repo_dir
from pr_reviewer.sandbox.runner import OUTPUT_LIMIT, Sandbox, docker_client
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.setup_agent.selection import is_test_file
from pr_reviewer.setup_agent.store import latest_runs

# Where the validation results go.
VALIDATION = "difftest-validation"


def _context() -> TryContext:
    """Connect to Docker and get the sandbox settings and image ready.

    Returns:
        TryContext: Docker, settings, the sandbox image and the log folder.
    """
    client = docker_client()
    config = load_sandbox_config()
    return TryContext(client, config, ready_image(client, config), logs_dir())


def _dev(args: argparse.Namespace) -> tuple[list[Instance], list[Instance]]:
    """Load all PRs and the dev PRs, and make sure bench_prs holds them.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `data`.

    Returns:
        tuple[list[Instance], list[Instance]]: All PRs and the dev PRs.
    """
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    with connect() as conn:
        save_bench_prs(conn, instances, split)
    return instances, split_prs(instances, split, "dev")


def _chosen(dev: Sequence[Instance], ids: Sequence[str] | None) -> list[Instance]:
    """Pick dev PRs by ID, refusing any that aren't dev PRs.

    Args:
        dev (Sequence[Instance]): The dev PRs.
        ids (Sequence[str] | None): The IDs asked for; None means every dev PR.

    Returns:
        list[Instance]: The PRs, in the order asked.

    Raises:
        PrepareError: If an ID isn't a dev PR (test-set PRs are locked).
    """
    if ids is None:
        return list(dev)
    by_id = {pr.instance_id: pr for pr in dev}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise PrepareError(
            f"Not dev PRs (or unknown; the test set is locked): {', '.join(missing)}"
        )
    return [by_id[i] for i in ids]


def _set_up(prs: Sequence[Instance]) -> list[Instance]:
    """Keep the PRs whose latest setup passed.

    Args:
        prs (Sequence[Instance]): The PRs.

    Returns:
        list[Instance]: The ones with a working environment.
    """
    with connect() as conn:
        runs = latest_runs(conn)
    return [pr for pr in prs if runs.get(pr.instance_id, {}).get("verified")]


def _prepare(args: argparse.Namespace) -> int:
    """Prepare base and PR-state images for set-up dev PRs.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `ids` and `data`.

    Returns:
        int: Exit code, 0 when every PR was prepared.
    """
    _, dev = _dev(args)
    prs = _set_up(_chosen(dev, args.ids))
    ctx = _context()
    failed = 0
    with connect() as conn:
        for number, pr in enumerate(prs, start=1):
            env = pr_environment(conn, pr)
            if find_prepared(ctx, pr, env.recipe):
                print(f"[{number}/{len(prs)}] {pr.instance_id}: already prepared")
                continue
            print(f"[{number}/{len(prs)}] {pr.instance_id}: preparing", flush=True)
            started = datetime.now(UTC)
            try:
                prepare(ctx, pr, env, on_step=lambda line: print(f"  {line}", flush=True))
            except PrepareError as exc:
                failed += 1
                print(f"  failed: {exc}", flush=True)
                continue
            minutes = (datetime.now(UTC) - started).total_seconds() / 60
            print(f"  done in {minutes:.1f} min", flush=True)
    print(f"Prepared {len(prs) - failed} of {len(prs)} PRs")
    return 1 if failed else 0


def _test_files(
    ctx: TryContext, pr: Instance, targets: Sequence[str], django: bool, at_base: bool
) -> tuple[FileSpec, ...]:
    """Read the files of some tests as they are at one side of a PR, inside a sandbox.

    Args:
        ctx (TryContext): Docker and the sandbox settings.
        pr (Instance): The PR.
        targets (Sequence[str]): pytest node IDs or files, or Django labels.
        django (bool): Whether the targets are Django labels.
        at_base (bool): Read them at the base instead of the PR state.

    Returns:
        tuple[FileSpec, ...]: The files found, each once.

    Raises:
        PrepareError: If a target's file isn't in the repo at that side.
    """
    files: dict[str, FileSpec] = {}
    git_dir = repo_dir(pr.repo)
    with Sandbox(ctx.client, ctx.config, ctx.base_image, purpose="files") as box:
        sha = base_of(box, git_dir, pr) if at_base else pr.head_commit
        for target in targets:
            if django:
                parts = target.split(".")
                candidates = [
                    "tests/" + "/".join(parts[:end]) + ".py" for end in range(len(parts), 0, -1)
                ]
            else:
                candidates = [target.split("::")[0]]
            for path in candidates:
                if path in files:
                    break
                shown = box.run(
                    ["git", "-C", git_dir, "show", f"{sha}:{path}"], timeout=300, label="show"
                )
                if shown.ok and len(shown.output) < OUTPUT_LIMIT:
                    files[path] = FileSpec(path, shown.output)
                    break
            else:
                raise PrepareError(f"No test file for {target} at {sha[:10]}")
    return tuple(files.values())


def _side_text(side: SideResult) -> str:
    """Describe a test's runs at one side in a few words.

    Args:
        side (SideResult): The test at one side.

    Returns:
        str: Such as "pass (passed, passed, passed)" or "fail (failed x3): AttributeError at
        pkg/a.py:12".
    """
    statuses = ", ".join(side.statuses)
    text = f"{side.verdict} ({statuses})"
    if side.reason:
        text += f", {side.reason}"
    if side.failure:
        where = f" at {side.failure.where}" if side.failure.where else ""
        text += f": {side.failure.exc_type or 'failure'}{where}"
    return text


def _label(outcome: Outcome) -> str:
    """Write an outcome for the terminal, with an arrow every console can show.

    Args:
        outcome (Outcome): Such as "pass_fail".

    Returns:
        str: Such as "pass->fail".
    """
    return LABELS[outcome].replace("→", "->")


def _print_outcome(test: DiffOutcome) -> None:
    """Print one test's outcome and its runs at both sides.

    Args:
        test (DiffOutcome): The test's outcome.
    """
    print(f"{_label(test.outcome):<10} {test.test}")
    print(f"    base: {_side_text(test.old)}")
    print(f"    PR:   {_side_text(test.new)}")


def _run(args: argparse.Namespace) -> int:
    """Run tests at a dev PR's base and PR state, and print each test's outcome.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `instance_id`, `tests`, `file`,
            `path`, `at`, `again` and `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    _, dev = _dev(args)
    (pr,) = _chosen(dev, [args.instance_id])
    ctx = _context()
    with connect() as conn:
        env = pr_environment(conn, pr)
        django = runner_kind(env.recipe) == "django"
        if args.file:
            if not args.path:
                print("--file needs --path: where the file goes in the repo", file=sys.stderr)
                return 1
            content = Path(args.file).read_text(encoding="utf-8")
            files: tuple[FileSpec, ...] = (FileSpec(args.path, content),)
            targets = tuple(args.tests) or (args.path,)
        else:
            if not args.tests:
                print("Name the tests to run, or give --file and --path", file=sys.stderr)
                return 1
            targets = tuple(args.tests)
            files = _test_files(ctx, pr, targets, django, args.at == "base")
        images = ready_prepared(ctx, pr, env, on_step=lambda line: print(f"  {line}", flush=True))
        result = diff_with_store(
            conn,
            ctx,
            pr,
            env.recipe,
            SideSpec("base", images["base"]),
            SideSpec("pr", images["pr"]),
            files,
            targets,
            again=args.again,
        )
    for test in result.tests:
        _print_outcome(test)
    for side, problems in result.problems.items():
        for problem in problems:
            print(f"{side} run problem: {problem.splitlines()[0]}")
    took = "stored result" if result.reused else f"{result.minutes:.1f} min"
    counts = ", ".join(f"{_label(k)} {v}" for k, v in result.counts().items() if v)
    print(f"{len(result.tests)} tests: {counts} ({took}, {RUNS_PER_SIDE} runs per side)")
    return 0


def _validation_prs(dev: Sequence[Instance]) -> list[tuple[Instance, list[str]]]:
    """Choose the PRs to validate on, with the related tests of those that get a related case.

    Every set-up dev PR whose commits change a test file is used. Each repo without one gets
    one PR checked with its related tests: the one whose setup was quickest, among those with
    tests related to their changes.

    Args:
        dev (Sequence[Instance]): The dev PRs.

    Returns:
        list[tuple[Instance, list[str]]]: The PRs, by repo, each with its related test files
        (empty unless it gets a related case).
    """
    with connect() as conn:
        runs = latest_runs(conn)
    usable = [pr for pr in dev if runs.get(pr.instance_id, {}).get("verified")]
    editing = {
        pr.instance_id
        for pr in usable
        if any(is_test_file(patch.file) for commit in pr.pr_commits for patch in commit.diff)
    }
    chosen: list[tuple[Instance, list[str]]] = [
        (pr, []) for pr in usable if pr.instance_id in editing
    ]
    covered = {pr.repo for pr, _ in chosen}
    for repo in sorted({pr.repo for pr in usable} - covered):
        candidates = [
            pr
            for pr in usable
            if pr.repo == repo and not (runs[pr.instance_id]["details"] or {}).get("tests_fallback")
        ] or [pr for pr in usable if pr.repo == repo]
        quickest = min(candidates, key=lambda pr: float(runs[pr.instance_id]["minutes"]))
        tests = list((runs[quickest.instance_id]["details"] or {}).get("tests") or [])
        chosen.append((quickest, tests))
    return sorted(chosen, key=lambda item: (item[0].repo, item[0].instance_id))


def _validate(args: argparse.Namespace) -> int:
    """Check the runner on known tests, and write results/difftest-validation/.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `ids`, `again` and `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    _, dev = _dev(args)
    planned = _validation_prs(dev)
    if args.ids:
        wanted = {pr.instance_id for pr in _chosen(dev, args.ids)}
        planned = [(pr, tests) for pr, tests in planned if pr.instance_id in wanted]
    folder = results_folder(VALIDATION)
    ctx = _context()
    print(f"Validating on {len(planned)} dev PRs, {RUNS_PER_SIDE} runs per side", flush=True)
    results = []
    notes: dict[str, list[str]] = {}
    with connect() as conn:
        for number, (pr, related) in enumerate(planned, start=1):
            print(f"[{number}/{len(planned)}] {pr.instance_id}", flush=True)
            env = pr_environment(conn, pr)
            try:
                images = ready_prepared(
                    ctx, pr, env, on_step=lambda line: print(f"  {line}", flush=True)
                )
            except PrepareError as exc:
                print(f"  skipped: {exc}", flush=True)
                notes[pr.instance_id] = [f"not prepared: {exc}"]
                continue
            found, found_notes = run_validation(
                conn,
                ctx,
                [(pr, env.recipe, images, related)],
                on_line=lambda line: print(f"  {line}", flush=True),
                again=args.again,
            )
            results.extend(found)
            notes.update(found_notes)
    config = {
        "name": VALIDATION,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "split": "dev",
        "prs": [pr.instance_id for pr, _ in planned],
        "runs_per_side": RUNS_PER_SIDE,
        "difftest_minutes": ctx.config.timeouts.difftest_minutes,
        "runner_version": RUNNER_VERSION,
        "sandbox_limits": ctx.config.limits.model_dump(),
    }
    write_validation(folder, results, notes, config)
    expected = sum(result.expected for result in results)
    tests = sum(len(result.base_pr.tests) for result in results)
    fixes = [result for result in results if result.kind == "fix_tests"]
    caught = sum(bool(result.caught) for result in fixes)
    print(
        f"{len(results)} cases, {tests} tests, {expected} as expected; caught {caught} of "
        f"{len(fixes)} known bugs. Saved {folder} (summary.md)"
    )
    return 0


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer difftest ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    difftest = commands.add_parser(
        "difftest", help="run tests before and after a dev PR, and compare"
    )
    sub = difftest.add_subparsers(metavar="DIFFTEST_COMMAND", required=True)
    prepare_cmd = sub.add_parser(
        "prepare", parents=[data_option], help="save each set-up dev PR's base and PR-state images"
    )
    prepare_cmd.add_argument("--ids", nargs="+", default=None, help="only these dev PRs")
    prepare_cmd.set_defaults(handler=_prepare)
    run = sub.add_parser(
        "run", parents=[data_option], help="run tests at a dev PR's base and PR state"
    )
    run.add_argument("instance_id", help="the dev PR, such as pydata__xarray-6260")
    run.add_argument(
        "tests",
        nargs="*",
        help="pytest node IDs or files (pkg/tests/test_a.py::test_b), or Django labels",
    )
    run.add_argument(
        "--from",
        dest="at",
        choices=["base", "pr"],
        default="pr",
        help="which version of the test files to run at both sides (default: pr)",
    )
    run.add_argument("--file", default=None, help="a test file of your own to run instead")
    run.add_argument("--path", default=None, help="where --file goes, relative to the repo root")
    run.add_argument("--again", action="store_true", help="run even if a result is stored")
    run.set_defaults(handler=_run)
    validate = sub.add_parser(
        "validate", parents=[data_option], help="check the runner on known tests"
    )
    validate.add_argument("--ids", nargs="+", default=None, help="only these dev PRs")
    validate.add_argument("--again", action="store_true", help="run even if results are stored")
    validate.set_defaults(handler=_validate)
