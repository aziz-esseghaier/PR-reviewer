"""Running a repo's own tests in a sandbox, and reading how many passed.

A setup counts as working only when the repo's own tests say so, never on the setup agent's
word. The related test files (selection.py) run with the network off, through the recipe's
test command, and the counts come from the runner's own summary line:
- pytest: "5 passed, 1 failed, 2 skipped in 3.2s";
- Django's runtests.py: "Ran 25 tests" and "FAILED (failures=1, errors=2, skipped=3)".
Skipped tests don't count either way. The environment passes when at least one test passed and
at least MIN_PASS_RATE of the rest did.

Imported by other modules; the file isn't run directly.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pr_reviewer.core.schemas import MIN_PASS_RATE
from pr_reviewer.sandbox.runner import RunResult, Sandbox, SandboxError
from pr_reviewer.setup_agent.selection import django_labels

# Words in pytest's summary line, and what they count as.
_PYTEST_WORDS = {
    "passed": "passed",
    "xpassed": "passed",
    "xfailed": "skipped",
    "skipped": "skipped",
    "deselected": "skipped",
    "failed": "failed",
    "error": "errors",
    "errors": "errors",
}


@dataclass(frozen=True)
class CheckOutcome:
    """How a test run went.

    Attributes:
        passed (int): Tests that passed.
        failed (int): Tests that failed.
        errors (int): Tests that couldn't run, such as files that failed to import.
        skipped (int): Tests skipped; they count neither way.
        problem (str | None): Why the environment doesn't pass, as a category such as
            "tests_failing", or None when it passes.
        detail (str): The end of the test output, for the setup agent to read.
        log_path (str | None): The file with the whole output.
    """

    passed: int
    failed: int
    errors: int
    skipped: int
    problem: str | None
    detail: str
    log_path: str | None

    @property
    def ok(self) -> bool:
        """Whether the environment passes."""
        return self.problem is None

    @property
    def pass_rate(self) -> float | None:
        """Share of the tests that ran which passed, or None when none ran."""
        total = self.passed + self.failed + self.errors
        return self.passed / total if total else None


def parse_pytest(output: str) -> tuple[int, int, int, int] | None:
    """Read pytest's last summary line.

    Args:
        output (str): The run's output.

    Returns:
        tuple[int, int, int, int] | None: Passed, failed, errors and skipped, or None when
        there's no summary line.
    """
    for line in reversed(output.splitlines()):
        found = re.findall(
            r"(\d+) (passed|xpassed|xfailed|skipped|deselected|failed|errors?)\b", line
        )
        if found and (" in " in line or line.strip().startswith("=")):
            counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
            for number, word in found:
                counts[_PYTEST_WORDS[word]] += int(number)
            return counts["passed"], counts["failed"], counts["errors"], counts["skipped"]
    return None


def parse_django(output: str) -> tuple[int, int, int, int] | None:
    """Read Django's test runner summary.

    Args:
        output (str): The run's output.

    Returns:
        tuple[int, int, int, int] | None: Passed, failed, errors and skipped, or None when
        there's no "Ran N tests" line.
    """
    ran = re.findall(r"^Ran (\d+) tests?", output, re.MULTILINE)
    if not ran:
        return None
    total = int(ran[-1])

    def count(name: str) -> int:
        found = re.findall(rf"\b{name}=(\d+)", output.rsplit("Ran ", 1)[-1])
        return int(found[-1]) if found else 0

    failed, errors = count("failures"), count("errors")
    skipped = count("skipped") + count("expected failures")
    return max(total - failed - errors - skipped, 0), failed, errors, skipped


def judge_run(result: RunResult, runner: str) -> CheckOutcome:
    """Turn a test run's result into counts and a verdict.

    Args:
        result (RunResult): The finished test command.
        runner (str): "pytest" or "django".

    Returns:
        CheckOutcome: The counts, and the problem when it doesn't pass.
    """
    detail = result.output[-1500:]
    log = str(result.log_path) if result.log_path else None
    counts = (parse_django if runner == "django" else parse_pytest)(result.output)
    passed, failed, errors, skipped = counts or (0, 0, 0, 0)
    problem: str | None = None
    if result.timed_out:
        problem = "tests_timeout"
    elif result.out_of_memory:
        problem = "tests_out_of_memory"
    elif counts is None:
        problem = "test_command_error"
    elif passed + failed + errors == 0:
        problem = "no_tests_ran"
    elif passed == 0 or passed < MIN_PASS_RATE * (passed + failed + errors):
        imports = re.search(r"ModuleNotFoundError|ImportError", result.output)
        problem = "test_import_errors" if imports else "tests_failing"
    return CheckOutcome(passed, failed, errors, skipped, problem, detail, log)


def run_tests(
    sandbox: Sandbox,
    test_command: str,
    path: str,
    test_files: Sequence[str],
    runner: str,
    *,
    timeout: float,
    env: Mapping[str, str],
) -> CheckOutcome:
    """Run some test files in a sandbox with the network off.

    Args:
        sandbox (Sandbox): The sandbox, with the environment installed.
        test_command (str): The recipe's test command, such as "python -m pytest".
        path (str): The repo's working copy in the sandbox.
        test_files (Sequence[str]): The test files to run, relative to the repo root.
        runner (str): "pytest" or "django"; Django's runner takes labels instead of files.
        timeout (float): Seconds the run may take.
        env (Mapping[str, str]): Environment variables, with the virtual environment's.

    Returns:
        CheckOutcome: The counts and the verdict.

    Raises:
        SandboxError: If the network is on: tests must never run with it.
    """
    if sandbox.online:
        raise SandboxError("Tests must run with the network off")
    if runner == "django":
        targets = django_labels(test_files)
        extra = "--parallel 1"
    else:
        targets = list(test_files)
        extra = "-p no:cacheprovider -q"
    # The test files go in as arguments to the shell, never pasted into the command text.
    script = f'{test_command} {extra} "$@"'
    result = sandbox.run(
        ["sh", "-c", script, "tests", *targets],
        timeout=timeout,
        label="tests",
        workdir=path,
        env=env,
    )
    return judge_run(result, runner)


def existing(sandbox: Sandbox, path: str, files: Sequence[str]) -> list[str]:
    """Keep the files that exist in the working copy right now.

    Args:
        sandbox (Sandbox): The sandbox.
        path (str): The working copy.
        files (Sequence[str]): Paths relative to the repo root.

    Returns:
        list[str]: The ones that exist, in the same order.
    """
    if not files:
        return []
    script = 'cd "$0" && for f do [ -f "$f" ] && echo "$f"; done; true'
    result = sandbox.run(["sh", "-c", script, path, *files], timeout=60, label="exists")
    found = set(result.output.split("\n"))
    return [f for f in files if f in found]
