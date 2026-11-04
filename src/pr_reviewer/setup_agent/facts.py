"""Repo facts: what a repo's own files say about installing and testing it, at one commit.

probe() copies probe.py into a sandbox and runs it at the repo root with the sandbox image's
Python. Its output is the repo's own text, so it's checked with pydantic and cut to size
before anything uses it. summary() turns the facts into the short text the setup agent's model
reads, kept small because the model's free tier allows only 8,000 tokens a minute.

Imported by other modules; the file isn't run directly.
"""

import json
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pr_reviewer.sandbox.repos import RepoError
from pr_reviewer.sandbox.runner import Sandbox

# Where probe.py goes inside the sandbox.
PROBE_PATH = "/tmp/pr_reviewer_probe.py"

# Most characters of excerpts shown to the model in total.
EXCERPT_BUDGET = 4000


class BuildFacts(BaseModel):
    """How a repo is built.

    Attributes:
        pyproject (bool): It has a pyproject.toml.
        setup_py (bool): It has a setup.py.
        setup_cfg (bool): It has a setup.cfg.
        meson (bool): It builds with Meson.
        setup_imports_numpy (bool): setup.py imports numpy, so numpy must be installed first.
        build_backend (str | None): The pyproject build backend, such as "setuptools.build_meta".
        build_requires (list[str]): The pyproject build requirements.
    """

    model_config = ConfigDict(extra="ignore")

    pyproject: bool = False
    setup_py: bool = False
    setup_cfg: bool = False
    meson: bool = False
    setup_imports_numpy: bool = False
    build_backend: str | None = Field(default=None, max_length=200)
    build_requires: list[str] = Field(default_factory=list, max_length=30)


class RepoFacts(BaseModel):
    """What a repo's files say about installing and testing it (see probe.py).

    Attributes:
        python_requires (str | None): Declared Python versions, such as ">=3.8".
        ci_pythons (list[str]): Python versions its CI runs, such as ["3.8", "3.9"].
        ci_install_lines (list[str]): Install commands from its CI config.
        requirement_files (list[str]): Requirement and environment files.
        extras (list[str]): Optional dependency groups, such as ["test"].
        build (BuildFacts): How it's built.
        test_runner (str): "pytest", or "django" for Django's own runner.
        excerpts (dict[str, str]): Short pieces of its setup files, by file name.
    """

    model_config = ConfigDict(extra="ignore")

    python_requires: str | None = Field(default=None, max_length=200)
    ci_pythons: list[str] = Field(default_factory=list, max_length=30)
    ci_install_lines: list[str] = Field(default_factory=list, max_length=20)
    requirement_files: list[str] = Field(default_factory=list, max_length=40)
    extras: list[str] = Field(default_factory=list, max_length=40)
    build: BuildFacts = Field(default_factory=BuildFacts)
    test_runner: str = Field(default="pytest", pattern=r"^(pytest|django)$")
    excerpts: dict[str, str] = Field(default_factory=dict)


def probe(sandbox: Sandbox, path: str) -> RepoFacts:
    """Read a repo's setup files inside a sandbox.

    Args:
        sandbox (Sandbox): The sandbox, with the repo checked out.
        path (str): The repo's working copy in the sandbox.

    Returns:
        RepoFacts: The checked facts.

    Raises:
        RepoError: If the probe fails or prints something that isn't the expected JSON.
    """
    script = resources.files("pr_reviewer.setup_agent").joinpath("probe.py").read_text("utf-8")
    sandbox.write_files("/tmp", {PROBE_PATH.rsplit("/", 1)[1]: script})
    result = sandbox.run(["python3", PROBE_PATH], timeout=120, label="probe", workdir=path)
    if not result.ok:
        raise RepoError(f"The repo probe failed: {result.describe()}: {result.output[-300:]}")
    try:
        return RepoFacts.model_validate(json.loads(result.output))
    except (json.JSONDecodeError, ValidationError) as exc:
        start = result.output[:200]
        raise RepoError(f"The repo probe printed something unexpected ({exc}): {start!r}") from None


def summary(facts: RepoFacts) -> str:
    """Write the facts as short text for the setup agent's model.

    Args:
        facts (RepoFacts): The facts.

    Returns:
        str: A few labelled lines, then excerpts of the setup files, within EXCERPT_BUDGET.
    """
    build = facts.build
    present = (
        ("pyproject.toml", build.pyproject),
        ("setup.py", build.setup_py),
        ("setup.cfg", build.setup_cfg),
        ("meson.build", build.meson),
    )
    kinds = [name for name, there in present if there]
    installs = [f"  {line}" for line in facts.ci_install_lines] or ["  none found"]
    lines = [
        f"Declared Python versions: {facts.python_requires or 'not declared'}",
        f"Python versions in CI: {', '.join(facts.ci_pythons) or 'none found'}",
        f"Build files: {', '.join(kinds) or 'none'}; backend: {build.build_backend or 'none'}",
        f"Build requirements: {', '.join(build.build_requires) or 'none'}",
        f"setup.py imports numpy: {'yes' if build.setup_imports_numpy else 'no'}",
        f"Extras: {', '.join(facts.extras) or 'none'}",
        f"Requirement files: {', '.join(facts.requirement_files) or 'none'}",
        f"Test runner: {facts.test_runner}",
        "CI install commands:",
        *installs,
    ]
    budget = EXCERPT_BUDGET
    for name, text in facts.excerpts.items():
        piece = f"--- {name} ---\n{text}"
        if len(piece) > budget:
            break
        lines.append(piece)
        budget -= len(piece)
    return "\n".join(lines)
