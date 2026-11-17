"""The `pr-reviewer evidence ...` commands: check a review run's candidates for evidence.

cli.py adds them to the `pr-reviewer` command. The candidates come from a dev review run; the
test set stays locked until the final run.

How to run (inside the conda env, from the project folder, with the database and Docker
Desktop running):
    pr-reviewer evidence run --with-issues    check the candidates of the PRs with known issues
                                              from candidates-gpt-oss-120b, as run
                                              evidence-gpt-oss-120b
    pr-reviewer evidence run --clean          then the candidates of the clean PRs
    pr-reviewer evidence run                  every candidate not checked yet
    pr-reviewer evidence run --ids <id> ...   only these PRs' candidates
    pr-reviewer evidence run --limit 3        at most 3 PRs this time
    pr-reviewer evidence run --parallel 1     one candidate at a time (default 2)
    pr-reviewer evidence report               write results/<name>/ from what's saved (no
                                              model calls, no Docker)
The run sleeps through the provider's quota waits (up to MAX_WAIT_S each). Running again
continues the run: finished candidates are kept, a PR stopped by a longer wait or a Docker
failure resumes where it stopped, and failed candidates get fresh replies. Model replies and
test runs are cached, so redoing work costs no quota and no Docker time.
"""

import argparse
import hashlib
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.dataset import Instance, load_dataset
from pr_reviewer.benchmark.judge import check_reviewer
from pr_reviewer.benchmark.results import results_folder, save_bench_prs
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, load_split, split_path, split_prs
from pr_reviewer.context.candidates import Piece
from pr_reviewer.context.candidates import load_candidates as load_context
from pr_reviewer.context.lookups import LookupLimits
from pr_reviewer.context.store import stored_contexts
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import ConfigError, ModelProfile, load_models_config, project_root
from pr_reviewer.core.limits import LimitError, RetryPolicy
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.checkpoints import open_checkpointer
from pr_reviewer.db.connection import connect
from pr_reviewer.difftest.environments import PrepareError, pr_environment, ready_prepared
from pr_reviewer.difftest.runner import RUNS_PER_SIDE, FileSpec, SideSpec, probe_source, runner_kind
from pr_reviewer.difftest.store import diff_with_store
from pr_reviewer.evidence.graph import (
    DOCKER_MINUTES,
    MAX_WRITER_STEPS,
    RUN_SECONDS,
    EvidenceRecord,
    GeneratedRun,
    PrInputs,
    SerialClient,
    Services,
    build_pr_graph,
    check_pr,
)
from pr_reviewer.evidence.inputs import Candidate, load_candidates, run_settings
from pr_reviewer.evidence.matching import MATCHING_VERSION
from pr_reviewer.evidence.prompts import (
    INTENT_PROMPT,
    LOOKUP_TOOLS,
    MAX_LOOKUPS,
    TEST_PROMPT,
    VERIFY_PROMPT,
    budget_chars,
    prompts_fingerprint,
)
from pr_reviewer.evidence.report import judge_labels, write_evidence_files
from pr_reviewer.evidence.store import (
    evidence_status,
    load_records,
    save_record,
    start_evidence_run,
)
from pr_reviewer.evidence.survey import SurveyError, fallback_materials, survey_pr
from pr_reviewer.reviewer.commands import LookupRunner
from pr_reviewer.reviewer.naive import MAX_TRIES
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.repos import RepoError
from pr_reviewer.sandbox.runner import SandboxError, docker_client
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.static.store import stored_static
from pr_reviewer.static.warnings import ToolWarning

# The run's default name (also its results folder) and the review run it checks. "tools":
# lookups are tool calls (an earlier run, evidence-gpt-oss-120b, had them in the JSON reply).
DEFAULT_NAME = "evidence-tools-gpt-oss-120b"
SOURCE_RUN = "candidates-gpt-oss-120b"

# The evidence agent's model profile in configs/models.yaml: the reviewer's model, with more
# room for its replies.
EVIDENCE_PROFILE = "evidence-gpt-oss-120b"

# Candidates checked at the same time, by default.
DEFAULT_PARALLEL = 2

# Longest wait for the provider before the run stops, in seconds. Groq's free daily quota
# refills gradually and asks for waits of up to about 18 minutes, which the run sleeps through
# instead of stopping. A stop would lose a request's retry in progress (such as asking again
# after a refused reply), so the restarted run would send the refused request again.
MAX_WAIT_S = 1800.0

# Where generated tests are kept, under the project folder.
EVIDENCE_DIR = Path("data") / "evidence"


class _Environment:
    """A PR's recipe and prepared images, found once and shared by its candidates.

    Args:
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
    """

    def __init__(self, ctx: TryContext, pr: Instance) -> None:
        self._ctx = ctx
        self._pr = pr
        self._lock = threading.Lock()
        self._ready: tuple[EnvRecipe, dict[str, str]] | None = None

    def get(self) -> tuple[EnvRecipe, dict[str, str]]:
        """Give the recipe and the base and PR images, preparing the images if needed.

        Returns:
            tuple[EnvRecipe, dict[str, str]]: The recipe, and the image of each side.

        Raises:
            PrepareError: If the PR has no working environment.
            SandboxError: If Docker fails.
        """
        with self._lock:
            if self._ready is None:
                with connect() as conn:
                    env = pr_environment(conn, self._pr)
                images = ready_prepared(
                    self._ctx, self._pr, env, on_step=lambda line: print(f"  {line}", flush=True)
                )
                self._ready = (env.recipe, {"base": images["base"], "pr": images["pr"]})
            return self._ready


def evidence_folder(run: str, pr: str, finding_id: str) -> Path:
    """Give the folder of a candidate's generated tests.

    Args:
        run (str): The evidence run.
        pr (str): The PR.
        finding_id (str): The candidate.

    Returns:
        Path: data/evidence/<run>/<PR>/<finding>/ under the project folder.
    """
    return project_root() / EVIDENCE_DIR / run / pr / finding_id


def _test_runner(run: str, ctx: TryContext, pr: Instance, environment: _Environment) -> Any:
    """Make the function that runs a candidate's generated test at both sides.

    Args:
        run (str): The evidence run, for where tests are kept.
        ctx (TryContext): The sandbox settings, image and log folder.
        pr (Instance): The PR.
        environment (_Environment): Its recipe and images.

    Returns:
        Any: A RunTest: (finding ID, path in the repo, code, attempt, lines to watch) ->
        GeneratedRun. Each call keeps the test's code in a file first, and uses its own
        Docker and database connections, since candidates run in parallel threads.
    """

    def run_test(
        finding_id: str, path: str, code: str, attempt: int, watch: Mapping[str, Sequence[int]]
    ) -> GeneratedRun:
        folder = evidence_folder(run, pr.instance_id, finding_id)
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / f"attempt-{attempt}.py"
        saved.write_text(code, encoding="utf-8", newline="\n")
        recipe, images = environment.get()
        own = TryContext(docker_client(), ctx.config, ctx.base_image, ctx.log_dir)
        with connect() as conn:
            result = diff_with_store(
                conn,
                own,
                pr,
                recipe,
                SideSpec("base", images["base"]),
                SideSpec("pr", images["pr"]),
                (FileSpec(path, code),),
                (path,),
                runs=RUNS_PER_SIDE,
                timeout=RUN_SECONDS,
                watch=watch,
            )
        problems = "; ".join(
            f"{side}: {found[0][:150]}" for side, found in result.problems.items() if found
        )
        return GeneratedRun(
            outcomes=result.tests,
            minutes=0.0 if result.reused else result.minutes,
            problems=problems,
            file=saved.relative_to(project_root()).as_posix(),
        )

    return run_test


def _hints(pieces: Sequence[Piece]) -> dict[str, list[str]]:
    """Give the test files the retrieved context found using each changed function.

    Args:
        pieces (Sequence[Piece]): The PR's context pieces.

    Returns:
        dict[str, list[str]]: By the changed function's qualified name, its test files,
        best first, each once.
    """
    hints: dict[str, list[str]] = {}
    for piece in pieces:
        if piece.kind in ("test", "changed_test") and piece.target:
            files = hints.setdefault(piece.target, [])
            if piece.file not in files:
                files.append(piece.file)
    return hints


def _settings(args: argparse.Namespace, source: dict[str, Any], models: Any) -> dict[str, Any]:
    """Give everything that decides an evidence run's results.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name` and `source`.
        source (dict[str, Any]): The source review run's settings.
        models (Any): The loaded model config.

    Returns:
        dict[str, Any]: The settings, as saved in the database.

    Raises:
        ConfigError: If the source run isn't a dev run with context and static keys, or its
            candidates came from another model (one model per run).
    """
    if source.get("split") != "dev":
        raise ConfigError(f"{args.source} isn't a dev run; the test set stays locked.")
    if "context" not in source or "static" not in source:
        raise ConfigError(f"{args.source} has no retrieved context or static warnings.")
    model = check_reviewer(EVIDENCE_PROFILE, models)
    if source.get("model") != model:
        raise ConfigError(
            f"{args.source}'s candidates came from {source.get('model')}, but the evidence "
            f"agent uses {model}; one model per run."
        )
    profile = models.profile(EVIDENCE_PROFILE)
    settings = {
        "name": args.name,
        "source_run": args.source,
        "split": "dev",
        "profile": EVIDENCE_PROFILE,
        "provider": profile.provider,
        "model": model,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "reasoning_effort": profile.reasoning_effort,
        "tokens_per_minute": profile.limits.tokens_per_minute,
        "prompts_sha256": prompts_fingerprint(),
        "request_chars": {
            "writer": budget_chars(profile, TEST_PROMPT, LOOKUP_TOOLS),
            "verifier": budget_chars(profile, VERIFY_PROMPT, LOOKUP_TOOLS),
            "intent": budget_chars(profile, INTENT_PROMPT),
        },
        "budget": {
            "writer_replies": MAX_WRITER_STEPS,
            "lookup_rounds": 1,
            "max_lookups": MAX_LOOKUPS,
            "docker_minutes": DOCKER_MINUTES,
            "run_seconds": RUN_SECONDS,
            "runs_per_side": RUNS_PER_SIDE,
            "verifier_lookup_rounds": 1,
            "max_tries": MAX_TRIES,
        },
        "lookup_limits": asdict(LookupLimits()),
        "context_key": source["context"]["settings_key"],
        "static_key": source["static"]["settings_key"],
        "probe_sha256": hashlib.sha256(probe_source().encode()).hexdigest(),
        "matching_version": MATCHING_VERSION,
        "dataset_sha256": DATASET.sha256,
        "split_sha256": SPLIT_SHA256,
    }
    # As saved in the database (tuples become lists), so a rerun compares equal.
    loaded: dict[str, Any] = json.loads(json.dumps(settings))
    return loaded


def _scope(
    args: argparse.Namespace, candidates: dict[str, list[Candidate]], dev: dict[str, Instance]
) -> list[Instance] | None:
    """Pick the PRs whose candidates to check this time.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `with_issues`, `clean` and `ids`.
        candidates (dict[str, list[Candidate]]): The source run's candidates by PR.
        dev (dict[str, Instance]): The dev PRs by ID.

    Returns:
        list[Instance] | None: The PRs, or None (after a message) when one asked for isn't
        a dev PR with candidates.
    """
    unknown = sorted(pr for pr in candidates if pr not in dev)
    if unknown:
        print(f"Not dev PRs (the test set is locked): {', '.join(unknown)}", file=sys.stderr)
        return None
    prs = [dev[pr] for pr in sorted(candidates)]
    if args.with_issues:
        prs = [pr for pr in prs if pr.changes]
    if args.clean:
        prs = [pr for pr in prs if not pr.changes]
    if args.ids:
        wanted = set(args.ids)
        missing = sorted(wanted - {pr.instance_id for pr in prs})
        if missing:
            print(f"No candidates in scope for: {', '.join(missing)}", file=sys.stderr)
            return None
        prs = [pr for pr in prs if pr.instance_id in wanted]
    return prs


def _run(args: argparse.Namespace) -> int:
    """Check the chosen PRs' candidates, and write the run's folder.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`, `source`, `with_issues`,
            `clean`, `ids`, `limit`, `parallel` and `data`.

    Returns:
        int: Exit code, 0 when every candidate in this scope is checked.
    """
    if args.parallel < 1:
        print("--parallel must be 1 or more", file=sys.stderr)
        return 1
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    dev = {pr.instance_id: pr for pr in split_prs(instances, split, "dev")}
    models = load_models_config()
    profile = models.profile(EVIDENCE_PROFILE)
    sandbox_config = load_sandbox_config()
    stopped = ""
    with connect() as conn, open_checkpointer() as checkpointer:
        save_bench_prs(conn, instances, split)
        source = run_settings(conn, args.source)
        settings = _settings(args, source, models)
        candidates = load_candidates(conn, args.source)
        prs = _scope(args, candidates, dev)
        if prs is None:
            return 1
        contexts = stored_contexts(conn, settings["context_key"])
        statics = stored_static(conn, settings["static_key"])
        start_evidence_run(
            conn,
            args.name,
            source_run=args.source,
            profile=EVIDENCE_PROFILE,
            model=settings["model"],
            settings=settings,
        )
        status = evidence_status(conn, args.name)
        todo = []
        for pr in prs:
            left = [
                (c.finding_id, status.get((pr.instance_id, c.finding_id), ("", 0))[1])
                for c in candidates[pr.instance_id]
                if status.get((pr.instance_id, c.finding_id), ("", 0))[0] != "done"
            ]
            if left:
                todo.append((pr, left))
        if args.limit:
            todo = todo[: args.limit]
        settings_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
        llm = SerialClient(
            LLMClient(
                EVIDENCE_PROFILE,
                models,
                cache=PostgresCache(conn),
                usage=PostgresUsageLog(conn),
                retry_policy=RetryPolicy(max_wait_s=MAX_WAIT_S),
            )
        )
        client = docker_client()
        ctx = TryContext(client, sandbox_config, ready_image(client, sandbox_config), logs_dir())
        lookups = LookupRunner(sandbox_config, LookupLimits())
        waiting = sum(len(left) for _, left in todo)
        print(
            f"Checking {waiting} candidates of {len(todo)} PRs from {args.source} with "
            f"{EVIDENCE_PROFILE} (run {args.name}, {args.parallel} at a time)",
            flush=True,
        )
        for number, (pr, left) in enumerate(todo, start=1):
            print(f"[{number}/{len(todo)}] {pr.instance_id}: {len(left)} candidates", flush=True)
            try:
                _check_one(
                    args,
                    checkpointer,
                    ctx,
                    profile,
                    llm,
                    lookups,
                    pr,
                    left,
                    candidates[pr.instance_id],
                    contexts,
                    statics,
                    settings_key,
                )
            except (LimitError, SandboxError) as exc:
                stopped = str(exc)
                break
        records = load_records(conn, args.name)
        labels = judge_labels(conn, args.source, candidates, dev)
    _write(args, settings, records, candidates, labels)
    if stopped:
        print(
            f"Stopped: {stopped}\nRun the same command again later: finished candidates are "
            "kept, and a stopped PR continues where it stopped.",
            file=sys.stderr,
        )
        return 1
    failed = sum(r["status"] == "failed" for r in records)
    if failed:
        print(f"{failed} candidates failed; run the same command again for fresh replies.")
        return 1
    return 0


def _check_one(
    args: argparse.Namespace,
    checkpointer: Any,
    ctx: TryContext,
    profile: ModelProfile,
    llm: SerialClient,
    lookups: LookupRunner,
    pr: Instance,
    left: list[tuple[str, int]],
    found: list[Candidate],
    contexts: dict[str, Path],
    statics: dict[str, Any],
    settings_key: str,
) -> None:
    """Check one PR's remaining candidates with the PR graph.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name` and `parallel`.
        checkpointer (Any): Where the graphs' progress is saved.
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        profile (ModelProfile): The model's profile, for the request sizes.
        llm (SerialClient): The model.
        lookups (LookupRunner): Answers lookups.
        pr (Instance): The PR.
        left (list[tuple[str, int]]): Its candidates to check, with their failures so far.
        found (list[Candidate]): All its candidates.
        contexts (dict[str, Path]): The stored context files, by PR.
        statics (dict[str, Any]): The stored static runs, by PR.
        settings_key (str): The run's settings fingerprint, part of the thread ID.

    Raises:
        LimitError, SandboxError: If the quota runs out or Docker fails.
    """
    pieces: Sequence[Piece] = ()
    if pr.instance_id in contexts:
        pieces = load_context(contexts[pr.instance_id]).pieces
    warnings: tuple[ToolWarning, ...] = ()
    stored = statics.get(pr.instance_id)
    result = stored.load() if stored is not None and stored.status == "ok" else None
    if result is not None:
        warnings = tuple(result.kept)
    environment = _Environment(ctx, pr)
    pytest = False
    try:
        recipe, _ = environment.get()
        pytest = runner_kind(recipe) == "pytest"
    except PrepareError as exc:
        print(f"  no environment ({exc}); verifier only", flush=True)
    by_id = {c.finding_id: c for c in found}
    inputs = PrInputs(
        pr=pr,
        candidates=by_id,
        warnings=warnings,
        pieces=pieces,
        pytest=pytest,
        writer_budget=budget_chars(profile, TEST_PROMPT, LOOKUP_TOOLS),
        verify_budget=budget_chars(profile, VERIFY_PROMPT, LOOKUP_TOOLS),
        intent_budget=budget_chars(profile, INTENT_PROMPT),
    )
    hints = _hints(pieces)

    def survey() -> Any:
        _, images = environment.get()
        return survey_pr(ctx, pr, images["pr"], found, hints)

    def save(record: EvidenceRecord) -> None:
        with connect() as own:
            save_record(own, args.name, record)

    services = Services(
        llm=llm,
        survey=survey,
        run_test=_test_runner(args.name, ctx, pr, environment),
        explore=lookups.for_pr(pr),
        save=save,
        show=lambda line: print(f"  {line}", flush=True),
    )
    graph = build_pr_graph(
        inputs,
        services,
        checkpointer,
        survey_problems=(SurveyError, RepoError, PrepareError),
        fallback=partial(fallback_materials, pr),
    )
    round_ = max(failures for _, failures in left)
    thread = f"{args.name}:{settings_key}:{pr.instance_id}:{round_}"
    check_pr(
        graph,
        pr.instance_id,
        left,
        thread,
        parallel=args.parallel,
        on_line=lambda line: print(f"  {line}", flush=True),
        tags=[EVIDENCE_PROFILE],
    )


def _write(
    args: argparse.Namespace,
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    candidates: dict[str, list[Candidate]],
    labels: dict[tuple[str, str], str],
) -> None:
    """Write the run's results folder and say where.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name`.
        settings (dict[str, Any]): The run's settings.
        records (list[dict[str, Any]]): Every candidate's record.
        candidates (dict[str, list[Candidate]]): The source run's candidates by PR.
        labels (dict[tuple[str, str], str]): The judge's label of each candidate.
    """
    folder = results_folder(args.name)
    shown = {**settings, "date": datetime.now(UTC).strftime("%Y-%m-%d")}
    write_evidence_files(folder, shown, records, candidates, labels)
    done = sum(r["status"] == "done" for r in records)
    total = sum(len(found) for found in candidates.values())
    print(f"{done} of {total} candidates checked. Saved {folder} (evidence_summary.md)")


def _report(args: argparse.Namespace) -> int:
    """Write the run's results folder from what's saved, with no model calls or Docker.

    Args:
        args (argparse.Namespace): Parsed arguments. Uses `name` and `data`.

    Returns:
        int: Exit code, 0 on success.
    """
    instances = load_dataset(args.data or default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    dev = {pr.instance_id: pr for pr in split_prs(instances, split, "dev")}
    with connect() as conn:
        row = conn.execute(
            "SELECT source_run, config FROM evidence_runs WHERE name = %s", (args.name,)
        ).fetchone()
        if row is None:
            print(f"There's no evidence run named {args.name!r}.", file=sys.stderr)
            return 1
        source_run, settings = row
        candidates = load_candidates(conn, source_run)
        records = load_records(conn, args.name)
        labels = judge_labels(conn, source_run, candidates, dev)
    _write(args, settings, records, candidates, labels)
    return 0


def add_commands(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    data_option: argparse.ArgumentParser,
) -> None:
    """Add `pr-reviewer evidence ...` and its subcommands to the command line.

    Args:
        commands (argparse._SubParsersAction[argparse.ArgumentParser]): The top-level
            subcommands of `pr-reviewer`.
        data_option (argparse.ArgumentParser): Parent parser with the `--data` option.
    """
    evidence = commands.add_parser(
        "evidence", help="check candidate findings with tests, static tools or a verifier"
    )
    sub = evidence.add_subparsers(metavar="EVIDENCE_COMMAND", required=True)
    run = sub.add_parser(
        "run",
        parents=[data_option],
        help="check a review run's candidates (model calls and Docker)",
    )
    run.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"the run's name and results folder (default: {DEFAULT_NAME})",
    )
    run.add_argument(
        "--source",
        default=SOURCE_RUN,
        help=f"the review run whose candidates to check (default: {SOURCE_RUN})",
    )
    which = run.add_mutually_exclusive_group()
    which.add_argument("--with-issues", action="store_true", help="only the PRs with known issues")
    which.add_argument("--clean", action="store_true", help="only the clean PRs")
    run.add_argument("--ids", nargs="+", default=None, help="only these PRs")
    run.add_argument("--limit", type=int, default=None, help="at most this many PRs this time")
    run.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"candidates checked at the same time (default: {DEFAULT_PARALLEL})",
    )
    run.set_defaults(handler=_run)
    report = sub.add_parser(
        "report", parents=[data_option], help="write the run's results folder from what's saved"
    )
    report.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"the run's name and results folder (default: {DEFAULT_NAME})",
    )
    report.set_defaults(handler=_report)
