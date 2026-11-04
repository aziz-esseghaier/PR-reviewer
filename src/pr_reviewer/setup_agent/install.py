"""Installing a recipe in a sandbox, and naming why an install failed.

install() runs a recipe with the network on (through the proxy): uv installs the recipe's
Python version (3.7 is built into the image instead), makes a virtual environment at
/work/venv, then the recipe's steps run in order at the repo root. Every step runs with:
- the virtual environment first on PATH;
- UV_EXCLUDE_NEWER set to the PR's date, so `uv pip install` picks package releases that
  existed then. An old commit installed with today's libraries often fails for reasons that
  have nothing to do with the PR. One exception: for PRs before September 2022, setuptools
  may come from up to then, the first versions that can do editable installs (uv_settings()).
The network is turned off again afterwards, whatever happened.

failure_category() reads a failed step's output and names the cause, such as
"dependency_resolution" or "missing_system_library". Our code names it from the log, not the
model, so the setup report's failure counts can be trusted.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt
import re
from dataclasses import dataclass

from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.sandbox.runner import RunResult, Sandbox
from pr_reviewer.setup_agent.versions import BUILT_IN

# Where every environment's virtual environment lives.
VENV = "/work/venv"

# The image's PATH, with the virtual environment in front.
PATH = f"{VENV}/bin:/usr/lib/ccache:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# setuptools can do the editable installs uv uses (`uv pip install -e .`) from version 64 on;
# 65.3 (2022-08-28) is the first steady release. For older PRs, setuptools alone may be as
# new as this date: it builds the code but isn't the code under test.
EDITABLE_SETUPTOOLS = dt.datetime(2022, 9, 1, tzinfo=dt.UTC)

# The uv settings file our code writes into every sandbox.
UV_CONFIG = "/tmp/pr_reviewer_uv.toml"

# Variables our code sets; a recipe may not change them.
RESERVED = frozenset(
    {"PATH", "VIRTUAL_ENV", "UV_EXCLUDE_NEWER", "UV_CONFIG_FILE", "HOME", "USER"}
    | {f"{p}_PROXY" for p in ("HTTP", "HTTPS", "NO", "ALL")}
    | {f"{p}_proxy" for p in ("http", "https", "no", "all")}
)

# Output patterns of each failure cause, checked in this order.
_CAUSES: tuple[tuple[str, str], ...] = (
    ("network_blocked", r"403 Filtered|Tunnel connection failed: 403|Access denied by proxy"),
    ("python_unavailable", r"No download found for request|No interpreter found for Python"),
    (
        "dependency_resolution",
        r"No solution found when resolving|ResolutionImpossible|we can conclude that|"
        r"Could not find a version that satisfies",
    ),
    (
        "missing_system_library",
        r"fatal error: [\w/.+-]+\.h: No such file|cannot find -l\w+|"
        r"Package '?[\w.+-]+'? was not found in the pkg-config|Dependency \"[\w-]+\" not found",
    ),
    (
        "build_failed",
        r"Failed to build|Failed to prepare|subprocess-exited-with-error|"
        r"error: command '[^']+' failed|ninja: build stopped|Cython\.Compiler\.Errors|"
        r"Building (wheel|editable) for .* failed|error: subprocess-exited",
    ),
    ("network_error", r"Failed to fetch|Could not connect|Connection (refused|reset)|timed out"),
)


@dataclass(frozen=True)
class InstallOutcome:
    """How installing a recipe went.

    Attributes:
        problem (str | None): The failure cause, such as "build_failed", or None when every
            step worked.
        step (str): The step that failed, or "" when none did.
        detail (str): The end of the failed step's output, for the setup agent to read.
        log_path (str | None): The failed step's log file.
    """

    problem: str | None
    step: str
    detail: str
    log_path: str | None

    @property
    def ok(self) -> bool:
        """Whether every step worked."""
        return self.problem is None


def environment(recipe: EnvRecipe, pr_date: dt.datetime) -> dict[str, str]:
    """Give the environment variables every install step and test run gets.

    Args:
        recipe (EnvRecipe): The recipe, with its own extra variables.
        pr_date (dt.datetime): When the PR was opened.

    Returns:
        dict[str, str]: The variables. The recipe's can't override ours.
    """
    when = pr_date.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ours = {
        "VIRTUAL_ENV": VENV,
        "PATH": PATH,
        "UV_EXCLUDE_NEWER": when,
        "UV_CONFIG_FILE": UV_CONFIG,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    extra = {k: v for k, v in recipe.env.items() if k not in RESERVED}
    return {**extra, **ours}


def uv_settings(pr_date: dt.datetime) -> str:
    """Write uv's settings for a PR: setuptools may be newer than an old PR's date.

    Args:
        pr_date (dt.datetime): When the PR was opened.

    Returns:
        str: The uv.toml text; empty for PRs from EDITABLE_SETUPTOOLS on.
    """
    if pr_date >= EDITABLE_SETUPTOOLS:
        return ""
    when = EDITABLE_SETUPTOOLS.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'exclude-newer-package = {{ setuptools = "{when}" }}\n'


def failure_category(result: RunResult, stage: str = "install") -> str:
    """Name why a step failed, from its output.

    Args:
        result (RunResult): The failed step.
        stage (str): "install", for the fallback names.

    Returns:
        str: Such as "dependency_resolution", "install_timeout" or "install_failed".
    """
    if result.timed_out:
        return f"{stage}_timeout"
    if result.out_of_memory:
        return f"{stage}_out_of_memory"
    for cause, pattern in _CAUSES:
        if re.search(pattern, result.output):
            return cause
    return f"{stage}_failed"


def install(
    sandbox: Sandbox,
    recipe: EnvRecipe,
    path: str,
    pr_date: dt.datetime,
    *,
    timeout: float,
    steps_only: bool = False,
) -> InstallOutcome:
    """Install a recipe in a sandbox, with the network on only while it runs.

    Args:
        sandbox (Sandbox): The sandbox, with the repo checked out.
        recipe (EnvRecipe): The recipe.
        path (str): The repo's working copy in the sandbox.
        pr_date (dt.datetime): When the PR was opened; packages are picked as of then.
        timeout (float): Seconds each step may take.
        steps_only (bool): Skip installing Python and making the virtual environment, as when
            the repo is reinstalled at another commit in the same sandbox.

    Returns:
        InstallOutcome: The failure cause and details, or success.
    """
    env = environment(recipe, pr_date)
    version = recipe.python_version
    # uv installs 3.8 and newer; older versions are built into the image.
    python = BUILT_IN.get(version, version)
    commands: list[tuple[str, list[str]]] = []
    if not steps_only:
        if version not in BUILT_IN:
            commands.append((f"uv python install {version}", ["uv", "python", "install", version]))
        has_venv = sandbox.run(
            ["sh", "-c", f'"{VENV}/bin/python" -c "import sys; print(sys.version[:5])"'],
            timeout=60,
            label="venv",
        )
        if not (has_venv.ok and has_venv.output.strip().startswith(f"{version}.")):
            commands.append(
                (
                    f"uv venv --seed --python {python} {VENV}",
                    ["uv", "venv", "--clear", "--seed", "--python", python, VENV],
                )
            )
    commands += [(step, ["sh", "-c", step]) for step in recipe.install_steps]
    sandbox.write_files("/tmp", {UV_CONFIG.rsplit("/", 1)[1]: uv_settings(pr_date)})
    sandbox.set_network(True)
    try:
        for step, command in commands:
            result = sandbox.run(command, timeout=timeout, label="install", workdir=path, env=env)
            if not result.ok:
                return InstallOutcome(
                    problem=failure_category(result),
                    step=step,
                    detail=result.output[-1500:],
                    log_path=str(result.log_path) if result.log_path else None,
                )
    finally:
        sandbox.set_network(False)
    return InstallOutcome(problem=None, step="", detail="", log_path=None)
