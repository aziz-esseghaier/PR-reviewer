"""The differential test runner: run the same tests before and after a PR, and compare.

run_diff() takes a PR, some test files (path and text) and the tests to run in them, and:
1. starts a sandbox from the old side's prepared image (the base), network off;
2. writes the test files into the working copy, so both sides run the very same test code
   and only the code under test differs;
3. runs the tests RUNS_PER_SIDE times, each run within the difftest time limit, and reads
   each test's result (our pytest plugin, or Django's verbose output);
4. does the same at the new side (the PR state);
5. gives each test one outcome (outcomes.py): pass_fail, pass_pass, fail_pass, fail_fail,
   flaky or not_run, with each run's status and the failures' exception types and lines.
A side can also be one of the PR's later commits (a fix): the sandbox then checks it out and
reinstalls the recipe's steps before the tests, network on only for the install.

Every run gets a request key: a fingerprint of everything that decides its result (the
images, the test code, the targets, the settings). store.py keeps results by it, so the same
test on the same environment is never run twice.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import json
import posixpath
import time
from collections import Counter
from dataclasses import dataclass
from importlib import resources

from pydantic import BaseModel, ConfigDict

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.difftest.outcomes import (
    OUTCOMES,
    DiffOutcome,
    Outcome,
    Runner,
    compare,
    covers,
)
from pr_reviewer.difftest.reports import (
    Failure,
    RunReport,
    read_django_output,
    read_pytest_report,
)
from pr_reviewer.sandbox.checkout import checkout_commit, working_copy
from pr_reviewer.sandbox.runner import DOCKER_ERRORS, Sandbox, SandboxError
from pr_reviewer.setup_agent.attempt import TryContext, recipe_fingerprint, runner_of
from pr_reviewer.setup_agent.install import environment, install

# How many times each side's tests run. A test whose runs disagree is flaky.
RUNS_PER_SIDE = 3

# Where the pytest plugin goes inside a sandbox (outside the repo), and the file it writes.
PROBE_DIR = "/tmp/pr_reviewer_probe"
PROBE_MODULE = "pr_reviewer_probe"
REPORT_FILE = f"{PROBE_DIR}/report.jsonl"

# Raised when what the runner does changes, so results stored before aren't reused.
# 2: pytest stopping early is a run problem; Django's output is unbuffered and read more
# loosely; files skipped at import count as skipped; when pytest stops because asked-for tests
# are in files that didn't load, the other tests run again without them.
RUNNER_VERSION = 2

# pytest exit codes for a run that stopped before giving every result: interrupted (such as
# a test calling pytest.exit), an internal error, or a usage error (such as a test ID it
# can't find, which stops the whole run).
_PYTEST_STOPS = {2: "interrupted", 3: "internal_error", 4: "usage_error"}

# Characters of a run's output kept with its result.
DETAIL_CHARS = 1500


@dataclass(frozen=True)
class FileSpec:
    """A test file to write into the working copy before the tests run.

    Attributes:
        path (str): Its path relative to the repo root, such as "pkg/tests/test_a.py".
        content (str): Its text.
    """

    path: str
    content: str

    def __post_init__(self) -> None:
        """Refuse paths that would leave the repo.

        Raises:
            ValueError: If the path is absolute, empty or goes up with "..".
        """
        parts = self.path.split("/")
        if not self.path or self.path.startswith("/") or ".." in parts or "\\" in self.path:
            raise ValueError(f"Test file paths must stay inside the repo: {self.path!r}")


@dataclass(frozen=True)
class SideSpec:
    """One side of a comparison: where its sandbox starts.

    Attributes:
        name (str): What the side is, such as "base", "pr" or "fix".
        image (str): The image its sandbox starts from (a prepared PR image).
        commit (str | None): One of the PR's commits to check out and reinstall at before the
            tests, or None to test the image as it is.
    """

    name: str
    image: str
    commit: str | None = None


class DiffResult(BaseModel):
    """The outcome of one differential run.

    Attributes:
        instance_id (str): The PR.
        old_side (str): The side tested first, such as "base".
        new_side (str): The side compared with it, such as "pr".
        targets (tuple[str, ...]): What was asked to run.
        tests (tuple[DiffOutcome, ...]): Each test's outcome.
        problems (dict[str, list[str]]): Per side, the runs that couldn't run and why, such as
            "timeout: ...".
        minutes (float): How long the run took.
        key (str): The request key the result is stored under.
        reused (bool): Whether it came from the store instead of running.
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str
    old_side: str
    new_side: str
    targets: tuple[str, ...]
    tests: tuple[DiffOutcome, ...]
    problems: dict[str, list[str]]
    minutes: float
    key: str
    reused: bool = False

    def counts(self) -> dict[Outcome, int]:
        """Count the tests per outcome.

        Returns:
            dict[Outcome, int]: Every outcome, with 0 for those no test had.
        """
        found = Counter(test.outcome for test in self.tests)
        return {outcome: found[outcome] for outcome in OUTCOMES}


def runner_kind(recipe: EnvRecipe) -> Runner:
    """Tell which test runner a recipe's tests use.

    Args:
        recipe (EnvRecipe): The recipe.

    Returns:
        Runner: "django" for Django's runtests.py, otherwise "pytest".
    """
    return "django" if runner_of(recipe) == "django" else "pytest"


def probe_source() -> str:
    """Give the pytest plugin's source, to copy into sandboxes.

    Returns:
        str: The text of pytest_probe.py.
    """
    return resources.files("pr_reviewer.difftest").joinpath("pytest_probe.py").read_text("utf-8")


def image_id(ctx: TryContext, image: str) -> str:
    """Give an image's ID, which changes whenever the image is built or prepared again.

    Args:
        ctx (TryContext): Docker.
        image (str): The image's name and tag.

    Returns:
        str: Its ID, such as "sha256:...".

    Raises:
        SandboxError: If Docker doesn't have it.
    """
    try:
        found = ctx.client.images.get(image).id
    except DOCKER_ERRORS as exc:
        raise SandboxError(f"Can't find the image {image}: {exc}") from None
    return str(found)


def request_key(
    ctx: TryContext,
    pr: Instance,
    recipe: EnvRecipe,
    sides: tuple[SideSpec, SideSpec],
    files: tuple[FileSpec, ...],
    targets: tuple[str, ...],
    runs: int,
    timeout: float,
) -> str:
    """Fingerprint everything that decides a differential run's result.

    Args:
        ctx (TryContext): Docker, to read the images' IDs.
        pr (Instance): The PR.
        recipe (EnvRecipe): Its recipe.
        sides (tuple[SideSpec, SideSpec]): The old and new sides.
        files (tuple[FileSpec, ...]): The test files written before the tests.
        targets (tuple[str, ...]): What runs.
        runs (int): Runs per side.
        timeout (float): Seconds per run.

    Returns:
        str: 64 hex characters.
    """
    what = {
        "runner": RUNNER_VERSION,
        "probe": hashlib.sha256(probe_source().encode()).hexdigest(),
        "pr": pr.instance_id,
        "recipe": recipe_fingerprint(recipe),
        "sides": [
            {"name": side.name, "image": image_id(ctx, side.image), "commit": side.commit}
            for side in sides
        ],
        "files": [
            {"path": f.path, "sha256": hashlib.sha256(f.content.encode()).hexdigest()}
            for f in files
        ],
        "targets": list(targets),
        "runs": runs,
        "timeout": timeout,
    }
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()


def _run_once(
    box: Sandbox,
    pr: Instance,
    recipe: EnvRecipe,
    path: str,
    targets: tuple[str, ...],
    runner: Runner,
    timeout: float,
) -> RunReport:
    """Run the targets once, network off, and read each test's result.

    Args:
        box (Sandbox): The sandbox, with the test files written.
        pr (Instance): The PR.
        recipe (EnvRecipe): Its recipe, for the test command and environment.
        path (str): The working copy.
        targets (tuple[str, ...]): pytest node IDs or files, or Django labels.
        runner (Runner): "pytest" or "django".
        timeout (float): Seconds the run may take.

    Returns:
        RunReport: Each test's result, or why the run couldn't run.
    """
    env = environment(recipe, pr.created_at)
    if runner == "django":
        # Unbuffered, Django's start-up lines (stdout) come out in order with the test lines
        # (stderr), instead of in the middle of one.
        env["PYTHONUNBUFFERED"] = "1"
        script = f'{recipe.test_command} --parallel 1 -v 2 "$@"'
    else:
        before = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{PROBE_DIR}:{before}" if before else PROBE_DIR
        env["PR_REVIEWER_REPORT"] = REPORT_FILE
        # Without --continue-on-collection-errors, one test file that fails to import (such
        # as a new test at the base) stops pytest before any other test runs.
        script = (
            f"rm -f {REPORT_FILE}; {recipe.test_command} -p {PROBE_MODULE} -p no:cacheprovider "
            '--continue-on-collection-errors -q "$@"'
        )
    if box.online:
        raise SandboxError("Tests must run with the network off")
    seconds = 0.0
    errors: dict[str, Failure] = {}
    skips: dict[str, str] = {}
    left = targets
    while True:
        # The targets go in as arguments to the shell, never pasted into the command text.
        result = box.run(
            ["sh", "-c", script, "difftest", *left],
            timeout=timeout,
            label="difftest",
            workdir=path,
            env=env,
        )
        seconds += result.seconds
        detail = result.output[-DETAIL_CHARS:]
        if result.timed_out or result.out_of_memory:
            problem = "timeout" if result.timed_out else "out_of_memory"
            return RunReport(
                results={}, collect_errors={}, problem=problem, detail=detail, seconds=seconds
            )
        if runner == "django":
            read = read_django_output(result.output, path)
            ran, stopped = "Ran " in result.output, None
        else:
            text = box.run(["cat", REPORT_FILE], timeout=60, label="report")
            report = text.output if text.ok else ""
            read = read_pytest_report(report, path)
            ran = '"kind": "session"' in report
            stopped = _PYTEST_STOPS.get(result.exit_code)
        errors.update(read.collect_errors)
        skips.update(read.collect_skips)
        if stopped != "usage_error":
            break
        # pytest stops the whole run when a test asked for by ID is in a file that didn't
        # load: one that fails to import (such as a PR's new tests at the base), or that
        # skipped itself (such as `pytest.importorskip()` at its top). Those tests count as
        # errors or skips, and the others run again without them.
        unloaded = [*read.collect_errors, *read.collect_skips]
        rest = tuple(
            target
            for target in left
            if not any(covers(owner, target, runner) for owner in unloaded)
        )
        if rest == left:
            break
        if not rest:
            stopped = None
            break
        left = rest
    if stopped is None and not (ran or read.results or errors):
        stopped = "runner_error"
    return RunReport(
        results=read.results,
        collect_errors=errors,
        collect_skips=skips,
        problem=stopped,
        detail=detail,
        seconds=seconds,
    )


def run_side(
    ctx: TryContext,
    pr: Instance,
    recipe: EnvRecipe,
    side: SideSpec,
    files: tuple[FileSpec, ...],
    targets: tuple[str, ...],
    *,
    runs: int = RUNS_PER_SIDE,
    timeout: float | None = None,
) -> list[RunReport]:
    """Run the tests several times at one side, in one fresh sandbox.

    Args:
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        recipe (EnvRecipe): Its recipe.
        side (SideSpec): Where the sandbox starts, and any commit to switch to first.
        files (tuple[FileSpec, ...]): Test files to write before the runs.
        targets (tuple[str, ...]): What to run.
        runs (int): How many runs.
        timeout (float | None): Seconds per run. None means the config's difftest limit.

    Returns:
        list[RunReport]: One report per run. When switching to the side's commit fails, every
        run reports that problem instead.

    Raises:
        SandboxError: If Docker or the sandbox fails.
    """
    seconds = timeout if timeout is not None else ctx.config.timeouts.difftest_minutes * 60
    runner = runner_kind(recipe)
    with Sandbox(
        ctx.client, ctx.config, side.image, purpose="difftest", log_dir=ctx.log_dir
    ) as box:
        path = working_copy(box, pr.repo)
        if side.commit is not None:
            path, _ = checkout_commit(box, pr, side.commit)
            done = install(
                box,
                recipe,
                path,
                pr.created_at,
                timeout=ctx.config.timeouts.install_minutes * 60,
                steps_only=True,
            )
            if not done.ok:
                failed = RunReport(
                    results={},
                    collect_errors={},
                    problem=f"install_failed ({done.problem})",
                    detail=done.detail[-DETAIL_CHARS:],
                    seconds=0.0,
                )
                return [failed] * runs
        # Folders are made first, by the sandbox user, so new test folders belong to it.
        folders = sorted({posixpath.dirname(f.path) for f in files} - {""})
        made = box.run(
            ["mkdir", "-p", PROBE_DIR, *folders], timeout=60, label="mkdir", workdir=path
        )
        if not made.ok:
            raise SandboxError(f"Can't make the test folders: {made.output.strip()[-300:]}")
        box.write_files(PROBE_DIR, {f"{PROBE_MODULE}.py": probe_source()})
        if files:
            box.write_files(path, {f.path: f.content for f in files})
        return [_run_once(box, pr, recipe, path, targets, runner, seconds) for _ in range(runs)]


def run_diff(
    ctx: TryContext,
    pr: Instance,
    recipe: EnvRecipe,
    old: SideSpec,
    new: SideSpec,
    files: tuple[FileSpec, ...],
    targets: tuple[str, ...],
    *,
    runs: int = RUNS_PER_SIDE,
    timeout: float | None = None,
    key: str = "",
) -> DiffResult:
    """Run tests at two sides of a PR and give each test's outcome.

    Args:
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        recipe (EnvRecipe): The recipe its prepared images were made with.
        old (SideSpec): The side before, usually the base.
        new (SideSpec): The side after, usually the PR state.
        files (tuple[FileSpec, ...]): Test files written at both sides before the runs.
        targets (tuple[str, ...]): What to run: test files, classes or tests (pytest node IDs,
            or Django labels).
        runs (int): Runs per side.
        timeout (float | None): Seconds per run. None means the config's difftest limit.
        key (str): The request key to record with the result.

    Returns:
        DiffResult: Each test's outcome.

    Raises:
        SandboxError: If Docker or a sandbox fails.
        ValueError: If no targets are given.
    """
    if not targets:
        raise ValueError("Give at least one test to run")
    if old.name == new.name:
        raise ValueError(f"The two sides need different names, not both {old.name!r}")
    started = time.monotonic()
    runner = runner_kind(recipe)
    reports = {
        side.name: run_side(ctx, pr, recipe, side, files, targets, runs=runs, timeout=timeout)
        for side in (old, new)
    }
    problems = {
        name: [f"{run.problem}: {run.detail[-300:]}" for run in runs_at if run.problem]
        for name, runs_at in reports.items()
    }
    return DiffResult(
        instance_id=pr.instance_id,
        old_side=old.name,
        new_side=new.name,
        targets=targets,
        tests=tuple(compare(targets, reports[old.name], reports[new.name], runner)),
        problems={name: found for name, found in problems.items() if found},
        minutes=round((time.monotonic() - started) / 60, 2),
        key=key,
    )
