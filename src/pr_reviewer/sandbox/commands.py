"""The `pr-reviewer sandbox ...` commands.

They build the sandbox image, download the dev PRs' repos into the repo volume, check every
dev PR's commits and checkouts, run any command at a PR's base or PR state, and clean up.
cli.py adds them to the `pr-reviewer` command.

How to run (inside the conda env, from the project folder, with Docker Desktop running):
    pr-reviewer sandbox build       build the sandbox and proxy images (first time ~11 minutes)
    pr-reviewer sandbox repos       download the dev PRs' repos and commits (~30 minutes once)
    pr-reviewer sandbox check       check every dev PR's commits and checkouts, offline
    pr-reviewer sandbox run <instance_id> [--at base|pr] [--network] -- <command> ...
                                    run one command in a fresh sandbox at that commit
    pr-reviewer sandbox status      show the images, the proxy, the repo copies and leftovers
    pr-reviewer sandbox cleanup     remove leftover sandbox containers
"""

import argparse
import shlex
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import docker
from docker import DockerClient

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, DatasetError, default_dataset_path
from pr_reviewer.benchmark.split import (
    SPLIT_SHA256,
    ensure_unlocked,
    load_split,
    split_path,
    split_prs,
)
from pr_reviewer.core.config import project_root
from pr_reviewer.db.connection import connect
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.commit_check import (
    CommitCheck,
    check_config,
    check_prs,
    save_checks,
    write_check_results,
)
from pr_reviewer.sandbox.config import LABEL, SandboxConfig, load_sandbox_config
from pr_reviewer.sandbox.image import build_image, image_exists, proxy_image, sandbox_image
from pr_reviewer.sandbox.proxy import proxy_status
from pr_reviewer.sandbox.repos import copy_sizes, update_repo
from pr_reviewer.sandbox.runner import Sandbox, docker_client, remove_leftovers

# Folder for the sandboxes' command logs, inside the project's data/ folder (not in git).
LOGS = Path("data") / "sandbox-logs"


def logs_dir() -> Path:
    """Give the folder where sandboxes keep one log file per command.

    Returns:
        Path: data/sandbox-logs/ in the project folder.
    """
    return project_root() / LOGS


def dev_prs(data: Path | None) -> list[Instance]:
    """Load the dev set's PRs.

    Args:
        data (Path | None): The SWR-Bench dataset file. None means the default one.

    Returns:
        list[Instance]: The dev PRs, in the split's order.
    """
    instances = load_dataset(data or default_dataset_path(), sha256=DATASET.sha256)
    return split_prs(instances, load_split(split_path(), sha256=SPLIT_SHA256), "dev")


def ready_image(client: DockerClient, config: SandboxConfig) -> str:
    """Make sure both images are built, building any that's missing or whose Dockerfile changed.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.

    Returns:
        str: The sandbox image's name and tag.
    """
    for spec in (proxy_image(config), sandbox_image(config)):
        if not image_exists(client, spec.tag):
            print(f"Building {spec.tag} (the first sandbox build takes about 11 minutes)")
            build_image(client, spec, on_progress=_print_step)
    return sandbox_image(config).tag


def _print_step(line: str) -> None:
    """Print the build lines that name a step, so a long build shows where it is.

    Args:
        line (str): One line of build output.
    """
    if line.startswith("Step "):
        print(f"  {line}", flush=True)


def _build(args: argparse.Namespace) -> int:
    """Build the sandbox and proxy images, unless they're already built.

    Args:
        args (argparse.Namespace): Parsed arguments (none used).

    Returns:
        int: Exit code, 0 on success.
    """
    client = docker_client()
    config = load_sandbox_config()
    for spec in (sandbox_image(config), proxy_image(config)):
        if image_exists(client, spec.tag):
            print(f"Already built: {spec.tag}")
            continue
        print(f"Building {spec.tag} from {spec.dockerfile.parent.name}/Dockerfile")
        build_image(client, spec, on_progress=_print_step)
        print(f"Built {spec.tag}")
    return 0


def _repos(args: argparse.Namespace) -> int:
    """Download the dev PRs' repos and commits into the repo volume.

    Each repo gets its own downloading sandbox, with the network on and the volume writable.
    Running it again only fetches what's new.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `data`.

    Returns:
        int: Exit code, 0 when every repo was updated (missing commits are reported, and the
        commit check marks their PRs as unusable).
    """
    client = docker_client()
    config = load_sandbox_config()
    tag = ready_image(client, config)
    by_repo: dict[str, list[Instance]] = defaultdict(list)
    for pr in dev_prs(args.data):
        by_repo[pr.repo].append(pr)
    timeout = config.timeouts.download_minutes * 60
    print(f"Updating the shared copies of {len(by_repo)} repos in the volume {config.repos_volume}")
    missing = 0
    for repo, prs in sorted(by_repo.items()):
        print(f"  {repo} ({len(prs)} PRs): ...", flush=True)
        with Sandbox(
            client,
            config,
            tag,
            purpose="repos",
            network=True,
            repos_writable=True,
            log_dir=logs_dir(),
        ) as sandbox:
            update = update_repo(sandbox, repo, prs, timeout=timeout)
        done = "downloaded" if update.cloned else "already there"
        print(
            f"  {repo}: {done}, {update.refs_fetched} PR refs fetched, "
            f"{len(update.fetched_by_id)} commits fetched by ID, "
            f"{len(update.missing)} reviewed commits missing",
            flush=True,
        )
        if update.refs_missing:
            print(f"    PRs GitHub has no ref for: {', '.join(map(str, update.refs_missing))}")
        if update.missing:
            print(f"    missing: {', '.join(sha[:10] for sha in update.missing)}")
        if update.base_fields_missing:
            print(f"    base_commit fields missing: {len(update.base_fields_missing)}")
        missing += len(update.missing)
    print(f"Done. Reviewed commits missing in all: {missing}")
    return 0


def _check(args: argparse.Namespace) -> int:
    """Check every dev PR's commits and checkouts, and save the results.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `data`.

    Returns:
        int: Exit code, 0 when the check ran (PRs found unusable are reported, not errors).
    """
    client = docker_client()
    config = load_sandbox_config()
    tag = ready_image(client, config)
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    prs = split_prs(instances, split, "dev")
    total = len(prs)
    done = 0
    print(f"Checking {total} dev PRs in sandboxes with the network off")

    def show(check: CommitCheck) -> None:
        """Print one line per checked PR."""
        nonlocal done
        done += 1
        if not check.usable:
            outcome = f"NOT usable: {check.problem}"
        else:
            notes = [f"base_commit field {check.base_field_relation}"]
            if check.merge_commits:
                notes.append(f"{check.merge_commits} merge(s)")
            if check.extra_files:
                notes.append(f"{len(check.extra_files)} extra file(s)")
            outcome = "usable, " + ", ".join(notes)
        print(f"  [{done}/{total}] {check.instance_id}: {outcome}", flush=True)

    checks = check_prs(client, config, tag, prs, log_dir=logs_dir(), on_result=show)
    folder = results_folder("commit-check")
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    write_check_results(folder, checks, check_config(date, tag, config, total))
    with connect() as conn:
        save_bench_prs(conn, instances, split)
        save_checks(conn, checks)
    usable = sum(check.usable for check in checks)
    print(f"Usable: {usable} of {total}. Saved {folder} (summary.md) and the pr_checkouts table")
    return 0


def _run(args: argparse.Namespace) -> int:
    """Run one command in a fresh sandbox, at a dev PR's base or PR state.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `instance_id`, `at`, `network`,
            `timeout`, `command` and `data`.

    Returns:
        int: Exit code, 0 when the command succeeded.

    Raises:
        LockedTestSetError: If the PR is in the test set.
        DatasetError: If no PR has that ID, or no command was given.
    """
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise DatasetError("Give the command to run after --, such as: -- git log -1")
    ensure_unlocked(load_split(split_path(), sha256=SPLIT_SHA256), args.instance_id)
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    found = [pr for pr in instances if pr.instance_id == args.instance_id]
    if not found:
        raise DatasetError(f"No PR with the ID {args.instance_id!r}")
    pr = found[0]
    client = docker_client()
    config = load_sandbox_config()
    tag = ready_image(client, config)
    minutes = args.timeout or (
        config.timeouts.install_minutes if args.network else config.timeouts.test_minutes
    )
    with Sandbox(
        client, config, tag, purpose="run", network=args.network, log_dir=logs_dir()
    ) as sandbox:
        path, sha = checkout(sandbox, pr, args.at)
        network = "on" if args.network else "off"
        print(f"{pr.instance_id} at its {args.at} side ({sha[:12]}), network {network}, in {path}")
        print(f"$ {shlex.join(command)}", flush=True)
        result = sandbox.run(command, timeout=minutes * 60, label="command", workdir=path)
    print(result.output, end="" if result.output.endswith("\n") else "\n")
    print(f"[{result.describe()}]")
    return 0 if result.ok else 1


def _status(args: argparse.Namespace) -> int:
    """Show the images, the proxy, the shared repo copies and leftover containers.

    Args:
        args (argparse.Namespace): Parsed arguments (none used).

    Returns:
        int: Exit code, 0 on success.
    """
    client = docker_client()
    config = load_sandbox_config()
    built = True
    for label, spec in (
        ("Sandbox image", sandbox_image(config)),
        ("Proxy image", proxy_image(config)),
    ):
        exists = image_exists(client, spec.tag)
        built = built and exists
        state = "built" if exists else "not built: pr-reviewer sandbox build"
        print(f"{label + ':':<14} {spec.tag} ({state})")
    hosts = len(config.proxy.allowed_hosts)
    print(f"Proxy:         {proxy_status(client, config)} (lets sandboxes reach {hosts} hosts)")
    try:
        client.volumes.get(config.repos_volume)
        has_volume = True
    except docker.errors.NotFound:
        has_volume = False
    print(f"Repo copies:   volume {config.repos_volume} ({'exists' if has_volume else 'not yet'})")
    if built and has_volume:
        tag = sandbox_image(config).tag
        with Sandbox(client, config, tag, purpose="status") as sandbox:
            sizes = copy_sizes(sandbox)
        for repo, size in sorted(sizes.items()):
            print(f"  {repo:<28} {size:>6,} MB")
        print(f"  {'total':<28} {sum(sizes.values()):>6,} MB")
    leftovers = client.containers.list(all=True, filters={"label": LABEL})
    print(f"Leftover sandbox containers: {len(leftovers)}")
    return 0


def _cleanup(args: argparse.Namespace) -> int:
    """Remove leftover sandbox containers.

    Args:
        args (argparse.Namespace): Parsed arguments (none used).

    Returns:
        int: Exit code, 0 on success.
    """
    removed = remove_leftovers(docker_client())
    print(f"Removed {removed} leftover sandbox containers")
    return 0


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer sandbox ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    sandbox = commands.add_parser("sandbox", help="Docker sandboxes that hold PR code")
    sub = sandbox.add_subparsers(metavar="SANDBOX_COMMAND", required=True)
    sub.add_parser("build", help="build the sandbox image").set_defaults(handler=_build)
    sub.add_parser(
        "repos", parents=[data_option], help="download the dev PRs' repos and commits"
    ).set_defaults(handler=_repos)
    sub.add_parser(
        "check",
        parents=[data_option],
        help="check every dev PR's commits and checkouts, with the network off",
    ).set_defaults(handler=_check)
    run = sub.add_parser(
        "run",
        parents=[data_option],
        help="run one command in a fresh sandbox at a dev PR's base or PR state",
    )
    run.add_argument("instance_id", help="the PR's ID, such as psf__requests-2317")
    run.add_argument(
        "--at",
        choices=["base", "pr"],
        default="pr",
        help="base: the commit just before the PR; pr: its last reviewed commit (default)",
    )
    run.add_argument("--network", action="store_true", help="turn the network on (default: off)")
    run.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="minutes the command may run (default: from configs/sandbox.yaml)",
    )
    # Everything after "--" is the command, so options like --at can come before it.
    run.add_argument("command", nargs="*", help="the command, after --, such as: -- git log -1")
    run.set_defaults(handler=_run)
    sub.add_parser(
        "status", help="show the image, the repo copies and leftover containers"
    ).set_defaults(handler=_status)
    sub.add_parser("cleanup", help="remove leftover sandbox containers").set_defaults(
        handler=_cleanup
    )
