"""
Tests for the setup agent's graph in setup_agent/agent.py: its routes, budget and model replies.

No Docker, database or real model: sandboxes, attempts and the recipe store are replaced by
fakes, and the model's replies are scripted. Each test plays one path through the graph:
- the standard recipe works, and the model is never called;
- the standard recipe fails, the model reads a file, then proposes a recipe that works;
- the model keeps failing until the budget is spent, and the setup gives up;
- the model's recipes keep failing with one error, and the setup stops early;
- the repo's tests run out of memory, and the setup stops without asking the model;
- the model repeats a recipe or writes broken JSON, and is told why;
- the PR has no related tests, so nothing is tried.
Also checks the reply schema and that prompts stay small.

How to run (from the project folder):
    pytest tests/setup_agent/test_agent.py
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.sandbox.runner import RunResult
from pr_reviewer.setup_agent import agent
from pr_reviewer.setup_agent.agent import (
    Proposal,
    SetupDeps,
    SetupResult,
    build_messages,
    run_setup,
    thread_for,
)
from pr_reviewer.setup_agent.attempt import Attempt, TryContext
from pr_reviewer.setup_agent.facts import RepoFacts
from pr_reviewer.setup_agent.selection import fallback_tests
from pr_reviewer.setup_agent.store import SetupRun
from tests.benchmark.test_dataset import make_row
from tests.fakes import ScriptedProvider, fake_client

PR = Instance.model_validate_json(json.dumps(make_row(7, year=2022, issues=())))

GOOD_REPLY = json.dumps(
    {
        "python_version": "3.9",
        "install_steps": ["uv pip install cython", "uv pip install -e .", "uv pip install pytest"],
        "test_command": "python -m pytest",
        "reason": "the CI uses 3.9",
    }
)
OTHER_REPLY = GOOD_REPLY.replace("3.9", "3.8")
READ_REPLY = json.dumps({"read_files": ["setup.cfg"], "install_steps": []})


class FakeSandbox:
    """Stands in for a sandbox: every command succeeds and prints a line of a file."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.log_dir = None

    def __enter__(self) -> "FakeSandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def run(self, command: Any, **kwargs: Any) -> RunResult:
        return RunResult(tuple(command), 0, 0.1, False, False, "[metadata]\nname = x\n", None)


@dataclass
class Fakes:
    """What the fakes saw, and how attempts should end.

    Attributes:
        outcomes (list[tuple[str, str | None]]): For each attempt in order, the source it
            must come from and its problem (None means it passes).
        tried (list[EnvRecipe]): The recipes tried, in order.
        runs (list[SetupRun]): The setups recorded.
        saved (list[str]): The sources of recipes kept.
    """

    outcomes: list[tuple[str, str | None]]
    tried: list[EnvRecipe] = field(default_factory=list)
    runs: list[SetupRun] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)


def attempt(source: str, recipe: EnvRecipe, problem: str | None) -> Attempt:
    """Make an attempt that ended with a problem, or passed.

    Args:
        source (str): Where the recipe came from.
        recipe (EnvRecipe): The recipe.
        problem (str | None): The failure category, or None for a pass.

    Returns:
        Attempt: The attempt.
    """
    counts = {"passed": 9, "failed": 1, "errors": 0, "skipped": 0}
    if problem is None:
        stage = "done"
    else:
        stage = "base_tests" if problem.startswith("tests_") else "install"
    return Attempt(
        source=source,
        recipe=recipe.model_dump(mode="json"),
        stage=stage,
        problem=problem,
        detail="" if problem is None else "error: command 'gcc' failed",
        base=counts,
        pr_side=counts if problem is None else None,
        minutes=2.0,
        image="pr-reviewer-env:x" if problem is None else None,
        log_dir=None,
    )


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[Fakes]:
    """Replace the graph's sandboxes, attempts and store with fakes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Swaps them for this test only.

    Yields:
        Fakes: The fakes' records. Tests set `outcomes` before running.
    """
    seen = Fakes(outcomes=[])

    def try_recipe(
        ctx: Any, pr: Any, recipe: EnvRecipe, source: str, tests: Any, **kwargs: Any
    ) -> Attempt:
        expected, problem = seen.outcomes.pop(0)
        assert source == expected, f"expected a {expected} attempt, got {source}"
        seen.tried.append(recipe)
        return attempt(source, recipe, problem)

    def save_recipe(conn: Any, recipe: EnvRecipe, year: int, source: str, image: Any) -> int:
        seen.saved.append(source)
        return 7

    facts = RepoFacts(python_requires=">=3.8", ci_pythons=["3.9"])
    monkeypatch.setattr(agent, "Sandbox", FakeSandbox)
    monkeypatch.setattr(agent, "checkout", lambda box, pr, side: ("/work/x", "a" * 40))
    monkeypatch.setattr(agent, "probe", lambda box, path: facts)
    monkeypatch.setattr(agent, "git", lambda box, args, timeout: "a.py\ntests/test_a.py")
    monkeypatch.setattr(agent, "try_recipe", try_recipe)
    monkeypatch.setattr(agent, "cached_recipes", lambda conn, repo, year: [])
    monkeypatch.setattr(agent, "save_recipe", save_recipe)
    monkeypatch.setattr(agent, "save_run", lambda conn, run: seen.runs.append(run))
    yield seen


def setup(replies: list[str | int]) -> tuple[SetupResult, list[str], ScriptedProvider]:
    """Run the graph on PR with scripted model replies.

    Args:
        replies (list[str | int]): The model's replies, in order.

    Returns:
        tuple[SetupResult, list[str], ScriptedProvider]: The result, the log lines and the
        fake provider (to count calls).
    """
    provider = ScriptedProvider(replies)
    deps = SetupDeps(
        ctx=TryContext(
            client=cast(Any, None), config=cast(Any, None), base_image="img", log_dir=None
        ),
        llm=fake_client(provider),
        conn=cast(Any, None),
        hosts=("pypi.org",),
    )
    lines: list[str] = []
    result = run_setup(deps, PR, InMemorySaver(), thread_for(PR), on_step=lines.append)
    return result, lines, provider


def test_standard_recipe_that_works_needs_no_model(fakes: Fakes) -> None:
    """Check the standard recipe is tried first, and the setup ends there when it passes."""
    fakes.outcomes = [("standard", None)]
    result, lines, provider = setup([])
    assert result.outcome == {"verified": True, "source": "standard", "recipe_id": 7}
    assert (result.model_calls, provider.requests) == (0, [])
    assert fakes.saved == ["standard"]
    assert fakes.runs[0].verified and fakes.runs[0].tests_run == 10
    assert lines[0] == "inspect: 1 related test files: tests/test_a.py"


def test_model_reads_a_file_then_its_recipe_works(fakes: Fakes) -> None:
    """Check: standard fails, the model reads setup.cfg, then proposes a recipe that passes."""
    fakes.outcomes = [("standard", "build_failed"), ("agent", None)]
    result, lines, provider = setup([READ_REPLY, GOOD_REPLY])
    assert result.outcome["source"] == "agent"
    assert result.model_calls == 2
    assert fakes.tried[1].python_version == "3.9"
    assert fakes.saved == ["agent"]
    second_prompt = json.loads(provider.requests[1].content)["messages"][1]["content"]
    assert "--- setup.cfg (first 80 lines) ---" in second_prompt
    assert "ended at install: build_failed" in second_prompt
    assert any(line.startswith("read_files: setup.cfg") for line in lines)


def test_setup_gives_up_when_the_budget_is_spent(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that after the last allowed call fails, the setup gives up with the last cause."""
    monkeypatch.setattr(agent, "MAX_MODEL_CALLS", 2)
    fakes.outcomes = [
        ("standard", "build_failed"),
        ("agent", "dependency_resolution"),
        ("agent", "missing_system_library"),
    ]
    result, lines, _ = setup([GOOD_REPLY, OTHER_REPLY])
    assert result.outcome["verified"] is False
    assert result.outcome["failure_category"] == "missing_system_library"
    assert result.model_calls == 2
    assert not fakes.runs[0].verified
    assert fakes.runs[0].failure_category == "missing_system_library"
    assert lines[-2].startswith("give_up: missing_system_library")


def test_setup_stops_early_when_the_agent_repeats_one_error(fakes: Fakes) -> None:
    """Check the setup gives up after 3 agent attempts in a row fail with the same error."""
    fakes.outcomes = [("standard", "build_failed"), *[("agent", "build_failed")] * 3]
    third = GOOD_REPLY.replace("3.9", "3.11")
    result, lines, provider = setup([GOOD_REPLY, OTHER_REPLY, third])
    assert result.model_calls == 3 and len(provider.requests) == 3
    assert result.outcome["failure_category"] == "build_failed"
    assert result.outcome["note"].startswith("stuck: the last 3 attempts failed the same way")
    assert lines[-2].startswith("give_up: build_failed (stuck")


def test_setup_stops_when_the_repos_tests_run_out_of_memory(fakes: Fakes) -> None:
    """Check base tests running out of memory end the setup with no model call."""
    fakes.outcomes = [("standard", "tests_out_of_memory")]
    result, lines, provider = setup([])
    assert (result.model_calls, provider.requests) == (0, [])
    assert result.outcome["failure_category"] == "tests_out_of_memory"
    assert result.outcome["note"].startswith("the repo's tests ran out of memory")
    assert lines[-2].startswith("give_up: tests_out_of_memory")


def test_failure_key_ignores_temporary_names_and_times() -> None:
    """Check two runs of one failure match, and a different error line doesn't."""
    recipe = EnvRecipe(
        repo="o/r", python_version="3.9", install_steps=("x",), test_command="python -m pytest"
    )

    def failed(detail: str) -> dict[str, Any]:
        data = asdict(attempt("agent", recipe, "build_failed"))
        return {**data, "detail": detail}

    one = failed("cd /home/.cache/uv/builds-v0/.tmpAb12Cd\nerror: no freetype (in 1.37s)\nhint: x")
    two = failed("cd /home/.cache/uv/builds-v0/.tmpZz9Yy8\nerror: no freetype (in 2.1s)\nhint: y")
    other = failed("error: no qhull")
    assert agent.failure_key(one) == agent.failure_key(two)
    assert agent.failure_key(one) != agent.failure_key(other)


def test_repeated_recipes_and_broken_json_are_sent_back(fakes: Fakes) -> None:
    """Check the model is told when it repeats a failed recipe or writes unusable JSON."""
    fakes.outcomes = [("standard", "build_failed"), ("agent", "tests_failing"), ("agent", None)]
    result, _, provider = setup([GOOD_REPLY, GOOD_REPLY, "not json", OTHER_REPLY])
    assert result.outcome["source"] == "agent"
    assert result.model_calls == 4
    prompts = [json.loads(r.content)["messages"][1]["content"] for r in provider.requests]
    assert "the same as attempt 2, which failed" in prompts[2]
    assert "the reply has no JSON object" in prompts[3]


def test_pr_without_related_tests_tries_nothing(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check a PR whose changes have no tests nearby ends at once, recorded as such."""
    monkeypatch.setattr(agent, "git", lambda box, args, timeout: "a.py\nREADME.py")
    result, _, provider = setup([])
    assert result.outcome == {"verified": False, "failure_category": "no_tests_found"}
    assert result.attempts == 0 and provider.requests == []
    assert fakes.runs[0].failure_category == "no_tests_found"


def test_pr_without_python_changes_is_checked_with_the_repos_tests(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check a PR changing only config gets the repo's smallest tests, and that's recorded."""
    config_only = PR.model_copy(
        update={
            "pr_commits": tuple(commit.model_copy(update={"diff": ()}) for commit in PR.pr_commits)
        }
    )
    blob = "100644 blob " + "1" * 40

    def git(box: Any, args: list[str], timeout: float) -> str:
        if "ls-tree" in args:
            assert args[-1] == "HEAD:pkg/tests"
            return f"{blob} 90000\ttest_big.py\n{blob}   300\ttest_x.py\n{blob}   200\ttest_y.py"
        return "pkg/tests/test_big.py\npkg/tests/test_x.py\npkg/tests/test_y.py"

    monkeypatch.setattr(agent, "git", git)
    monkeypatch.setattr(agent, "fallback_tests", partial(fallback_tests, count=2))
    fakes.outcomes = [("standard", None)]
    deps = SetupDeps(
        ctx=TryContext(cast(Any, None), cast(Any, None), base_image="img", log_dir=None),
        llm=fake_client(ScriptedProvider([])),
        conn=cast(Any, None),
        hosts=("pypi.org",),
    )
    lines: list[str] = []
    result = run_setup(deps, config_only, InMemorySaver(), "t", on_step=lines.append)
    assert result.outcome["verified"]
    assert lines[0] == (
        "inspect: 2 of the repo's test files (no related ones): "
        "pkg/tests/test_x.py, pkg/tests/test_y.py"
    )
    assert fakes.runs[0].details["tests_fallback"] is True


@pytest.mark.parametrize(
    "reply",
    [
        {"python_version": "3.6", "install_steps": ["x"]},  # older than we can provide
        {"python_version": "3.9", "install_steps": ["x"], "env": {"PATH": "/x"}},  # ours
        {"read_files": ["/etc/passwd"]},  # outside the repo
        {"read_files": ["../x"]},
        {"reason": "nothing to do"},  # neither a recipe nor files
    ],
)
def test_unusable_replies_are_refused(reply: dict[str, Any]) -> None:
    """Check the reply schema refuses what our code shouldn't run or read."""
    with pytest.raises(ValidationError):
        Proposal.model_validate(reply)


def test_prompts_stay_small() -> None:
    """Check a prompt with big facts and long failures stays under ~4,000 tokens."""
    facts = RepoFacts(
        excerpts={f"file{i}.txt": "x" * 2000 for i in range(10)},
        ci_install_lines=["pip install -e .[test]"] * 10,
    )
    long_attempt = attempt(
        "agent",
        EnvRecipe(
            repo=PR.repo,
            python_version="3.9",
            install_steps=("uv pip install -e .",) * 12,
            test_command="python -m pytest",
        ),
        "build_failed",
    ).__dict__ | {"detail": "y" * 5000}
    state = cast(
        agent.SetupState,
        {
            "facts": facts.model_dump(),
            "tests": ["tests/test_a.py"] * 5,
            "attempts": [long_attempt] * 6,
            "files": {"setup.cfg": "z\n" * 80},
            "feedback": "",
        },
    )
    text = "".join(m["content"] for m in build_messages(state, PR, ("pypi.org",)))
    assert len(text) / 4 < 4000
