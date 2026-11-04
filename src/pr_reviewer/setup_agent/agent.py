"""The setup agent: a LangGraph graph that builds a working environment for one PR.

    inspect -> cached -> standard -> propose <-> read_files
                                        |  ^
                                        v  |
                                     agent_try          (any node) -> give_up -> record

- inspect: read the repo's setup files at the PR's base (facts.py) and pick the related tests
  (selection.py). A PR that changes no Python code gets the repo's smallest few test files.
- cached: try the recipes kept for the repo, its own year's first (store.py). No model call.
- standard: try the standard recipe (standard.py). No model call.
- propose: ask the model for a recipe, showing it the repo's facts and how every earlier
  attempt failed. It may ask to read up to two files first (read_files).
- agent_try: try the model's recipe. A failure goes back to propose, with its log.
- give_up: when the budget is spent (MAX_MODEL_CALLS calls or MAX_MINUTES of attempts), when
  the agent's last SAME_FAILURE_LIMIT attempts all failed with the same error, or when the
  repo's own tests ran out of memory at the base, which no recipe can fix.
- record: save the outcome to setup_runs, and a working recipe to env_recipes.

Every attempt runs in a fresh sandbox (attempt.py), and only the repo's own tests decide
whether it worked. The model only writes JSON; our code runs its commands, in the sandbox.
Progress is saved after each node in PostgreSQL, so a setup stopped by the model's daily limit
continues where it stopped, one thread per PR.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt
import operator
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, TypedDict

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg.rows import TupleRow
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.llm import ChatMessage, LLMClient
from pr_reviewer.core.schemas import EnvName, EnvRecipe
from pr_reviewer.demo_graph import parse_json_reply
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.image import image_exists
from pr_reviewer.sandbox.repos import RepoError, git
from pr_reviewer.sandbox.runner import Sandbox, SandboxError
from pr_reviewer.setup_agent.attempt import (
    Attempt,
    TryContext,
    recipe_fingerprint,
    try_recipe,
)
from pr_reviewer.setup_agent.facts import RepoFacts, probe, summary
from pr_reviewer.setup_agent.install import RESERVED
from pr_reviewer.setup_agent.selection import (
    fallback_tests,
    main_test_folder,
    related_tests,
    tree_sizes,
)
from pr_reviewer.setup_agent.standard import standard_recipe
from pr_reviewer.setup_agent.store import SetupRun, cached_recipes, save_recipe, save_run
from pr_reviewer.setup_agent.versions import OLDEST_PYTHON, RELEASES

# The budget of one setup: model calls, and minutes spent in attempts.
MAX_MODEL_CALLS = 12
MAX_MINUTES = 60.0

# A setup stops early when the agent's last this-many attempts all failed with the same error:
# the model is stuck, and its remaining calls are better kept for other PRs.
SAME_FAILURE_LIMIT = 3

# Most lines of a file shown to the model when it asks to read one.
FILE_LINES = 80

SYSTEM_PROMPT = """You set up test environments for old commits of Python repos.

Each environment is built in a fresh Linux sandbox: Debian 12 with gcc, g++, gfortran, make,
pkg-config, ccache, and headers for freetype, png, jpeg, zlib, ffi, ssl, openblas, lapack,
hdf5 and libmemcached. Commands run as a normal user: no apt-get, no sudo. The Python version
you choose (3.7 or newer) is installed for you, and a virtual environment at /work/venv is
already active for every command.

Your install steps run in order at the repo root, with internet access only to: {hosts}.
Downloads from any other site fail. Each step runs with /bin/sh (not bash); curl isn't
installed. Install packages with `uv pip install ...`: it picks the package versions released
before the PR's date, which keeps old code working (only setuptools may be newer, so that
editable installs work). Prefer the repo's own requirement files and extras.
A meson-python repo (build backend mesonpy) rebuilds on import: install its build requirements
and ninja first, then `uv pip install --no-build-isolation -C compile-args=-j1 -e .` (one
compile job fits the memory; MAX_JOBS does nothing), and test with `pytest`, not
`python -m pytest`. When the build's setuptools pin can't be met (too old for editable
installs, or newer than the PR's date), install setuptools and wheel first, then
`uv pip install --no-build-isolation -e .`.
Then the related tests run with the network off, at the PR's base and again at the PR state
(your install steps run again there). The environment works when at least 90% of them pass.
The test files are added to the end of test_command. Don't run tests in install_steps.

Reply with only a JSON object:
{{"python_version": "3.X", "install_steps": ["..."], "test_command": "python -m pytest",
"env": {{}}, "read_files": [], "reason": "one sentence"}}
To read up to 2 of the repo's files first, give their paths in "read_files" and leave
"install_steps" empty."""


class Proposal(BaseModel):
    """The model's reply: a recipe to try, or files to read first.

    Attributes:
        python_version (str | None): Such as "3.8".
        install_steps (list[str]): Shell commands, run in order at the repo root.
        test_command (str): The command the test files are added to.
        env (dict[EnvName, str]): Extra environment variables.
        read_files (list[str]): Files to read before proposing, relative to the repo root.
        reason (str): The model's one-sentence reason.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    python_version: str | None = Field(default=None, pattern=r"^3\.\d{1,2}$")
    install_steps: list[str] = Field(default_factory=list, max_length=12)
    test_command: str = Field(default="python -m pytest", min_length=1, max_length=300)
    env: dict[EnvName, str] = Field(default_factory=dict, max_length=10)
    read_files: list[str] = Field(default_factory=list, max_length=2)
    reason: str = Field(default="", max_length=500)

    @field_validator("install_steps")
    @classmethod
    def _check_steps(cls, steps: list[str]) -> list[str]:
        """Keep install steps non-empty and of a sane length.

        Args:
            steps (list[str]): The steps as written.

        Returns:
            list[str]: The steps.

        Raises:
            ValueError: If a step is empty or longer than 600 characters.
        """
        for step in steps:
            if not step.strip() or len(step) > 600:
                raise ValueError("each step must be a non-empty command of at most 600 characters")
        return steps

    @field_validator("read_files")
    @classmethod
    def _check_paths(cls, paths: list[str]) -> list[str]:
        """Allow only plain paths inside the repo.

        Args:
            paths (list[str]): The paths as written.

        Returns:
            list[str]: The paths.

        Raises:
            ValueError: If a path is absolute, climbs out with "..", or has odd characters.
        """
        for path in paths:
            parts = PurePosixPath(path).parts
            if path.startswith("/") or ".." in parts or not path or len(path) > 200:
                raise ValueError(f"read_files must be paths inside the repo, not {path!r}")
        return paths

    @field_validator("env")
    @classmethod
    def _check_env(cls, env: dict[str, str]) -> dict[str, str]:
        """Refuse variables our code sets itself.

        Args:
            env (dict[str, str]): The variables as written.

        Returns:
            dict[str, str]: The variables.

        Raises:
            ValueError: If one of them is reserved, such as PATH.
        """
        taken = sorted(set(env) & RESERVED)
        if taken:
            raise ValueError(f"env can't set {', '.join(taken)}")
        return env

    @model_validator(mode="after")
    def _recipe_or_files(self) -> "Proposal":
        """Make sure the reply is either a recipe or a request to read files.

        Returns:
            Proposal: The unchanged proposal.

        Raises:
            ValueError: If it's neither, or names a Python version uv can't install.
        """
        if self.install_steps:
            version = self.python_version
            if version is None or version not in RELEASES:
                raise ValueError(f"python_version must be one of {', '.join(RELEASES)}")
            if tuple(map(int, version.split("."))) < tuple(map(int, OLDEST_PYTHON.split("."))):
                raise ValueError(f"python_version must be {OLDEST_PYTHON} or newer")
        elif not self.read_files:
            raise ValueError("give install_steps and python_version, or read_files")
        return self


class SetupState(TypedDict):
    """Everything the setup graph knows about one PR's setup. Saved after each node.

    Attributes:
        instance_id (str): The PR.
        thread_id (str): The thread its progress is saved under.
        agent_model (str): The model profile of the run. A thread keeps its model.
        facts (dict[str, Any] | None): RepoFacts data, once inspected.
        tests (list[str]): The test files that check an environment.
        tests_fallback (bool): The PR has no related tests, so these are a few of the repo's.
        attempts (list[dict[str, Any]]): Every attempt, as Attempt data. Nodes add to it.
        model_calls (int): Calls to the model so far.
        minutes (float): Minutes spent in sandboxes so far.
        proposal (dict[str, Any] | None): The model's recipe waiting to be tried.
        file_requests (list[str]): Files the model asked to read.
        files (dict[str, str]): The files it read last, by path.
        feedback (str): Why the model's last reply couldn't be used, for its next call.
        outcome (dict[str, Any] | None): How the setup ended, once it has.
        log (list[str]): One line per node run. Nodes add to it.
    """

    instance_id: str
    thread_id: str
    agent_model: str
    facts: dict[str, Any] | None
    tests: list[str]
    tests_fallback: bool
    attempts: Annotated[list[dict[str, Any]], operator.add]
    model_calls: int
    minutes: float
    proposal: dict[str, Any] | None
    file_requests: list[str]
    files: dict[str, str]
    feedback: str
    outcome: dict[str, Any] | None
    log: Annotated[list[str], operator.add]


SetupGraph = CompiledStateGraph[SetupState, None, SetupState, SetupState]


@dataclass(frozen=True)
class SetupDeps:
    """What the setup graph's nodes use.

    Attributes:
        ctx (TryContext): Docker, sandbox settings and the log folder.
        llm (LLMClient): The setup agent's model. One model for every setup.
        conn (psycopg.Connection[TupleRow]): Database connection, for the recipe cache.
        hosts (tuple[str, ...]): The hosts the proxy allows, for the model's prompt.
    """

    ctx: TryContext
    llm: LLMClient
    conn: psycopg.Connection[TupleRow]
    hosts: tuple[str, ...]


def within_budget(state: SetupState) -> bool:
    """Tell whether the setup may still ask the model.

    Args:
        state (SetupState): The current state.

    Returns:
        bool: True when calls and minutes are both under the budget.
    """
    return state["model_calls"] < MAX_MODEL_CALLS and state["minutes"] < MAX_MINUTES


def failure_key(attempt: dict[str, Any]) -> str:
    """Sum up why an attempt failed in one line, to spot the same error coming back.

    The line is where the attempt ended, its failure category, and the last line of its output
    that mentions an error, with temporary folder names and run times blanked out, since those
    change on every try.

    Args:
        attempt (dict[str, Any]): Attempt data.

    Returns:
        str: The summary, such as "install|build_failed|error: Failed to download ...".
    """
    lines = [line.strip() for line in attempt["detail"].splitlines() if line.strip()]
    errors = [
        line
        for line in lines
        if re.search(r"error", line, re.IGNORECASE) and not line.startswith("hint:")
    ]
    last = errors[-1] if errors else (lines[-1] if lines else "")
    last = re.sub(r"tmp[\w-]+", "tmp", last)
    last = re.sub(r"\b\d+(\.\d+)?s\b", "Ns", last)
    return f"{attempt['stage']}|{attempt['problem']}|{last[:300]}"


def stuck(state: SetupState) -> bool:
    """Tell whether the agent's last few attempts all failed with the same error.

    Args:
        state (SetupState): The current state.

    Returns:
        bool: True when the last SAME_FAILURE_LIMIT attempts all came from the agent and
        failed with the same failure_key.
    """
    last = state["attempts"][-SAME_FAILURE_LIMIT:]
    if len(last) < SAME_FAILURE_LIMIT:
        return False
    if any(a["source"] != "agent" or a["problem"] is None for a in last):
        return False
    return len({failure_key(a) for a in last}) == 1


def tests_out_of_memory(state: SetupState) -> bool:
    """Tell whether the last attempt's tests ran out of memory at the base.

    The install worked, and the repo's own tests need more memory than a sandbox has, so a new
    recipe can't help.

    Args:
        state (SetupState): The current state.

    Returns:
        bool: True when the last attempt ended in the base tests, out of memory.
    """
    if not state["attempts"]:
        return False
    last = state["attempts"][-1]
    return bool(last["stage"] == "base_tests" and last["problem"] == "tests_out_of_memory")


def keep_going(state: SetupState) -> bool:
    """Tell whether the setup should ask the model again.

    Args:
        state (SetupState): The current state.

    Returns:
        bool: True when the budget isn't spent, the agent isn't stuck on one error, and the
        repo's tests didn't run out of memory.
    """
    return within_budget(state) and not stuck(state) and not tests_out_of_memory(state)


def _attempt_line(number: int, attempt: dict[str, Any], full: bool) -> str:
    """Describe an earlier attempt for the model.

    Args:
        number (int): Its number, from 1.
        attempt (dict[str, Any]): Attempt data.
        full (bool): Include the recipe's steps and the end of the failing output.

    Returns:
        str: The description.
    """
    recipe = attempt["recipe"]
    head = (
        f"Attempt {number} ({attempt['source']}): Python {recipe['python_version']}, "
        f"ended at {attempt['stage']}: {attempt['problem']}"
    )
    if not full:
        return head
    steps = "\n".join(f"  $ {step}" for step in recipe["install_steps"])
    return (
        f"{head}\n Steps:\n{steps}\n Test command: {recipe['test_command']}\n"
        f" Output end:\n{attempt['detail'][-1000:]}"
    )


def build_messages(state: SetupState, pr: Instance, hosts: tuple[str, ...]) -> list[ChatMessage]:
    """Write the conversation for the next model call, from the state.

    Built fresh each time, from short pieces, to stay within the model's token limit.

    Args:
        state (SetupState): The current state.
        pr (Instance): The PR.
        hosts (tuple[str, ...]): The hosts the proxy allows.

    Returns:
        list[ChatMessage]: The system and user messages.
    """
    facts = RepoFacts.model_validate(state["facts"])
    attempts = state["attempts"]
    lines = [
        f"Repo: {pr.repo}. PR opened {pr.created_at:%Y-%m-%d} (packages are pinned to that date).",
        f"Related tests: {', '.join(state['tests'])}",
        "",
        "What the repo's files say:",
        summary(facts),
        "",
        "Attempts so far (latest last):" if attempts else "No attempts yet.",
    ]
    for number, attempt in enumerate(attempts, start=1):
        lines.append(_attempt_line(number, attempt, full=number > len(attempts) - 2))
    for path, text in state["files"].items():
        lines += ["", f"--- {path} (first {FILE_LINES} lines) ---", text]
    if state["feedback"]:
        lines += ["", f"Your last reply couldn't be used: {state['feedback']}"]
    lines += ["", "Propose the next recipe."]
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT.format(hosts=", ".join(hosts))),
        ChatMessage(role="user", content="\n".join(lines)),
    ]


def build_setup_graph(
    deps: SetupDeps, pr: Instance, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> SetupGraph:
    """Wire the setup nodes and edges for one PR.

    Args:
        deps (SetupDeps): Docker, the model and the database.
        pr (Instance): The PR to build an environment for.
        checkpointer (BaseCheckpointSaver[Any] | None): Where progress is saved after each
            node. None saves nothing.

    Returns:
        SetupGraph: The compiled graph.
    """
    ctx = deps.ctx

    def spent(started: float) -> float:
        return round((time.monotonic() - started) / 60, 2)

    def inspect(state: SetupState) -> dict[str, Any]:
        """Read the repo's facts at the base and pick the related tests."""
        started = time.monotonic()
        try:
            with Sandbox(ctx.client, ctx.config, ctx.base_image, purpose="inspect") as box:
                path, _ = checkout(box, pr, "base")
                facts = probe(box, path)
                files = git(box, ["-C", path, "ls-files", "*.py"], timeout=600).splitlines()
                changed = [patch.file for commit in pr.pr_commits for patch in commit.diff]
                tests = related_tests(changed, files)
                fallback = not tests
                folder = main_test_folder(files)
                if fallback and folder is not None:
                    # The PR changes no Python code near any test: check with the repo's
                    # smallest test files. "HEAD:<folder>" can't be read as an option.
                    tree = "HEAD" if folder == "." else f"HEAD:{folder}"
                    listing = git(box, ["-C", path, "ls-tree", "-l", tree], timeout=600)
                    tests = fallback_tests(files, tree_sizes(listing, folder))
        except (RepoError, SandboxError) as exc:
            outcome = {"verified": False, "failure_category": "inspect_failed", "note": str(exc)}
            return {"outcome": outcome, "minutes": spent(started), "log": [f"inspect: {exc}"]}
        kind = "of the repo's test files (no related ones)" if fallback else "related test files"
        update: dict[str, Any] = {
            "facts": facts.model_dump(),
            "tests": tests,
            "tests_fallback": fallback,
            "minutes": state["minutes"] + spent(started),
            "log": [f"inspect: {len(tests)} {kind}: {', '.join(tests) or 'none'}"],
        }
        if not tests:
            update["outcome"] = {"verified": False, "failure_category": "no_tests_found"}
        return update

    def tried(minutes: float, attempt: Attempt, recipe_id: int | None) -> dict[str, Any]:
        """Turn an attempt into a state update, ending the setup if it passed."""
        update: dict[str, Any] = {
            "attempts": [asdict(attempt)],
            "minutes": round(minutes + attempt.minutes, 2),
            "log": [f"{attempt.source}: {attempt.describe()} in {attempt.minutes:.1f} min"],
        }
        if attempt.ok:
            update["outcome"] = {
                "verified": True,
                "source": attempt.source,
                "recipe_id": recipe_id,
            }
        return update

    def cached(state: SetupState) -> dict[str, Any]:
        """Try the recipes kept for this repo, nearest year first."""
        merged: dict[str, Any] = {"attempts": [], "log": [], "minutes": state["minutes"]}
        for stored in cached_recipes(deps.conn, pr.repo, pr.year):
            image = (
                stored.image if stored.image and image_exists(ctx.client, stored.image) else None
            )
            attempt = try_recipe(
                ctx,
                pr,
                stored.recipe,
                "cache",
                state["tests"],
                start_image=image,
                save_image=image is None,
            )
            recipe_id = None
            if attempt.ok:
                # Kept again under this PR's year, so the next PR of the year finds it first.
                recipe_id = save_recipe(
                    deps.conn, stored.recipe, pr.year, stored.source, attempt.image or image
                )
            update = tried(merged["minutes"], attempt, recipe_id)
            merged["attempts"] += update["attempts"]
            merged["log"] += update["log"]
            merged["minutes"] = update["minutes"]
            if "outcome" in update:
                merged["outcome"] = update["outcome"]
                break
        if not merged["log"]:
            merged["log"] = ["cached: no kept recipe for this repo yet"]
        return merged

    def standard(state: SetupState) -> dict[str, Any]:
        """Try the standard recipe, unless a cached recipe was the same."""
        recipe = standard_recipe(pr, RepoFacts.model_validate(state["facts"]))
        seen = {
            recipe_fingerprint(EnvRecipe.model_validate(a["recipe"])) for a in state["attempts"]
        }
        if recipe_fingerprint(recipe) in seen:
            return {"log": ["standard: same as a recipe already tried"]}
        attempt = try_recipe(ctx, pr, recipe, "standard", state["tests"], save_image=True)
        recipe_id = (
            save_recipe(deps.conn, recipe, pr.year, "standard", attempt.image)
            if attempt.ok
            else None
        )
        return tried(state["minutes"], attempt, recipe_id)

    def propose(state: SetupState) -> dict[str, Any]:
        """Ask the model for the next recipe, or for files to read."""
        response = deps.llm.complete(build_messages(state, pr, deps.hosts))
        calls = state["model_calls"] + 1
        base: dict[str, Any] = {"model_calls": calls, "proposal": None, "file_requests": []}
        try:
            proposal = parse_json_reply(response.text, Proposal)
        except ValueError as exc:
            return {
                **base,
                "feedback": str(exc)[:500],
                "log": [f"propose: call {calls}, unusable: {exc}"],
            }
        if not proposal.install_steps:
            return {
                **base,
                "feedback": "",
                "file_requests": proposal.read_files,
                "log": [f"propose: call {calls}, wants to read {', '.join(proposal.read_files)}"],
            }
        recipe = EnvRecipe(
            repo=pr.repo,
            python_version=proposal.python_version or "",
            install_steps=tuple(proposal.install_steps),
            test_command=proposal.test_command,
            env=proposal.env,
        )
        seen = [
            recipe_fingerprint(EnvRecipe.model_validate(a["recipe"])) for a in state["attempts"]
        ]
        if recipe_fingerprint(recipe) in seen:
            number = seen.index(recipe_fingerprint(recipe)) + 1
            feedback = f"that recipe is the same as attempt {number}, which failed; change it"
            return {
                **base,
                "feedback": feedback,
                "log": [f"propose: call {calls}, repeated attempt {number}"],
            }
        return {
            **base,
            "feedback": "",
            "proposal": recipe.model_dump(mode="json"),
            "log": [f"propose: call {calls}, Python {recipe.python_version}: {proposal.reason}"],
        }

    def read_files(state: SetupState) -> dict[str, Any]:
        """Read the files the model asked for, at the base."""
        started = time.monotonic()
        files: dict[str, str] = {}
        try:
            with Sandbox(ctx.client, ctx.config, ctx.base_image, purpose="read") as box:
                path, _ = checkout(box, pr, "base")
                for name in state["file_requests"]:
                    result = box.run(
                        ["sh", "-c", f'head -n {FILE_LINES} -- "$1" | cut -c1-200', "read", name],
                        timeout=60,
                        label="read",
                        workdir=path,
                    )
                    files[name] = result.output if result.ok else "(no such file)"
        except (RepoError, SandboxError) as exc:
            files = {name: f"(couldn't read: {exc})" for name in state["file_requests"]}
        return {
            "files": files,
            "file_requests": [],
            "minutes": state["minutes"] + spent(started),
            "log": [f"read_files: {', '.join(files)}"],
        }

    def agent_try(state: SetupState) -> dict[str, Any]:
        """Try the model's recipe."""
        recipe = EnvRecipe.model_validate(state["proposal"])
        attempt = try_recipe(ctx, pr, recipe, "agent", state["tests"], save_image=True)
        recipe_id = (
            save_recipe(deps.conn, recipe, pr.year, "agent", attempt.image) if attempt.ok else None
        )
        return {**tried(state["minutes"], attempt, recipe_id), "proposal": None, "files": {}}

    def give_up(state: SetupState) -> dict[str, Any]:
        """End a setup that can't go on, naming the last failure and why it stopped."""
        last = state["attempts"][-1]["problem"] if state["attempts"] else "no_usable_recipe"
        used = f"{state['model_calls']} model calls, {state['minutes']:.0f} minutes"
        if tests_out_of_memory(state):
            why = f"the repo's tests ran out of memory, which a new recipe can't fix ({used})"
        elif stuck(state):
            why = f"stuck: the last {SAME_FAILURE_LIMIT} attempts failed the same way ({used})"
        else:
            why = f"budget spent: {used}"
        outcome = {"verified": False, "failure_category": last, "note": why}
        return {"outcome": outcome, "log": [f"give_up: {last} ({why})"]}

    def record(state: SetupState) -> dict[str, Any]:
        """Save the setup's outcome to setup_runs."""
        outcome = state["outcome"] or {}
        good = next((a for a in reversed(state["attempts"]) if a["problem"] is None), None)
        first = next((a for a in state["attempts"] if a["base"]), None)

        def rate(side: dict[str, Any] | None) -> float | None:
            if not side:
                return None
            ran = side["passed"] + side["failed"] + side["errors"]
            return round(side["passed"] / ran, 4) if ran else None

        base_side = (good or first or {}).get("base")
        save_run(
            deps.conn,
            SetupRun(
                instance_id=pr.instance_id,
                thread_id=state["thread_id"],
                repo=pr.repo,
                year=pr.year,
                verified=bool(outcome.get("verified")),
                source=outcome.get("source"),
                failure_category=outcome.get("failure_category"),
                recipe_id=outcome.get("recipe_id"),
                attempts=len(state["attempts"]),
                model_calls=state["model_calls"],
                minutes=state["minutes"],
                base_pass_rate=rate(base_side),
                pr_pass_rate=rate((good or {}).get("pr_side")),
                tests_run=(
                    base_side["passed"] + base_side["failed"] + base_side["errors"]
                    if base_side
                    else None
                ),
                agent_model=state["agent_model"],
                details={
                    "tests": state["tests"],
                    "tests_fallback": state["tests_fallback"],
                    "attempts": state["attempts"],
                    "note": outcome.get("note"),
                },
            ),
        )
        verdict = "verified" if outcome.get("verified") else outcome.get("failure_category")
        return {"log": [f"record: {verdict}"]}

    def after_inspect(state: SetupState) -> Literal["cached", "record"]:
        return "record" if state["outcome"] else "cached"

    def after_cached(state: SetupState) -> Literal["standard", "record"]:
        return "record" if state["outcome"] else "standard"

    def next_or_stop(state: SetupState) -> Literal["propose", "give_up", "record"]:
        if state["outcome"]:
            return "record"
        return "propose" if keep_going(state) else "give_up"

    def after_propose(
        state: SetupState,
    ) -> Literal["agent_try", "read_files", "propose", "give_up"]:
        if state["proposal"]:
            return "agent_try"
        if state["file_requests"]:
            return "read_files"
        return "propose" if keep_going(state) else "give_up"

    builder = StateGraph(SetupState)
    for name, node in (
        ("inspect", inspect),
        ("cached", cached),
        ("standard", standard),
        ("propose", propose),
        ("read_files", read_files),
        ("agent_try", agent_try),
        ("give_up", give_up),
        ("record", record),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "inspect")
    builder.add_conditional_edges("inspect", after_inspect, ["cached", "record"])
    builder.add_conditional_edges("cached", after_cached, ["standard", "record"])
    builder.add_conditional_edges("standard", next_or_stop, ["propose", "give_up", "record"])
    builder.add_conditional_edges(
        "propose", after_propose, ["agent_try", "read_files", "propose", "give_up"]
    )
    builder.add_conditional_edges("read_files", next_or_stop, ["propose", "give_up", "record"])
    builder.add_conditional_edges("agent_try", next_or_stop, ["propose", "give_up", "record"])
    builder.add_edge("give_up", "record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class SetupResult:
    """How one PR's setup ended.

    Attributes:
        instance_id (str): The PR.
        outcome (dict[str, Any]): Verified or not, what worked, or why it failed.
        attempts (int): Recipes tried.
        model_calls (int): Model calls made.
        minutes (float): Minutes spent in sandboxes.
        already_finished (bool): The thread had finished before, so nothing ran.
    """

    instance_id: str
    outcome: dict[str, Any]
    attempts: int
    model_calls: int
    minutes: float
    already_finished: bool


def thread_for(pr: Instance, fresh: bool = False) -> str:
    """Name the thread a PR's setup is saved under.

    Args:
        pr (Instance): The PR.
        fresh (bool): Give a new thread, to set the PR up again from the start.

    Returns:
        str: Such as "setup-psf__requests-2317", plus the time for a fresh one.
    """
    thread = f"setup-{pr.instance_id}"
    return f"{thread}-{dt.datetime.now():%Y%m%d-%H%M%S}" if fresh else thread


def run_setup(
    deps: SetupDeps,
    pr: Instance,
    checkpointer: BaseCheckpointSaver[Any],
    thread: str,
    on_step: Callable[[str], None] | None = None,
) -> SetupResult:
    """Run the setup graph for a PR: start it, resume it, or give its saved result.

    Args:
        deps (SetupDeps): Docker, the model and the database.
        pr (Instance): The PR.
        checkpointer (BaseCheckpointSaver[Any]): Where progress is saved.
        thread (str): The thread to use (see thread_for).
        on_step (Callable[[str], None] | None): Called with each log line as nodes finish.

    Returns:
        SetupResult: How it ended.

    Raises:
        ConfigError: If the thread was started with another model.
        openai.APIError, LimitError, LLMError: If a model call fails, such as when the daily
            limit is reached. Finished nodes stay saved; running again resumes after them.
    """
    show = on_step or (lambda line: None)
    graph = build_setup_graph(deps, pr, checkpointer)
    config: RunnableConfig = {
        "configurable": {"thread_id": thread},
        "run_name": "setup",
        "tags": [deps.llm.name, pr.repo],
        "metadata": {"instance_id": pr.instance_id, "agent_model": deps.llm.name},
    }
    saved = graph.get_state(config)
    start: SetupState | None
    if saved.values and saved.values["agent_model"] != deps.llm.name:
        raise ConfigError(
            f"{thread} was run with {saved.values['agent_model']}; a setup keeps one model"
        )
    if saved.values and not saved.next:
        values = saved.values
        return SetupResult(
            pr.instance_id,
            values.get("outcome") or {},
            len(values["attempts"]),
            values["model_calls"],
            values["minutes"],
            already_finished=True,
        )
    if saved.values:
        show(f"(resuming at {', '.join(saved.next)})")
        start = None
    else:
        start = SetupState(
            instance_id=pr.instance_id,
            thread_id=thread,
            agent_model=deps.llm.name,
            facts=None,
            tests=[],
            tests_fallback=False,
            attempts=[],
            model_calls=0,
            minutes=0.0,
            proposal=None,
            file_requests=[],
            files={},
            feedback="",
            outcome=None,
            log=[],
        )
    for update in graph.stream(start, config, stream_mode="updates"):
        for changes in update.values():
            for line in (changes or {}).get("log", []):
                show(line)
    values = graph.get_state(config).values
    return SetupResult(
        pr.instance_id,
        values.get("outcome") or {},
        len(values["attempts"]),
        values["model_calls"],
        values["minutes"],
        already_finished=False,
    )
