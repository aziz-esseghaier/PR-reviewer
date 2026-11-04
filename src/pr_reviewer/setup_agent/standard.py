"""The standard recipe: the usual way to install a Python repo, tried before asking the model.

Most repos install the same way: the right Python, the repo in editable mode with its test
extra, its test requirements file, and pytest. Trying that first costs no model calls, and the
setup agent only gets involved when it fails. The steps use `uv pip install`, so packages are
picked as of the PR's date (see install.py).

These special cases are handled:
- an old setup.py that imports numpy (scikit-learn, for example) needs its build requirements
  installed before the build, which must then use them (no build isolation). The requirements
  come from the repo's pyproject.toml (numpy, Cython, scipy...), or a usual set when it has
  none. A version pin on setuptools is dropped, so the newer setuptools allowed for old PRs
  can do the editable install (see install.py), and SETUPTOOLS_USE_DISTUTILS=stdlib keeps
  numpy's old build tools working with it;
- a meson-python build (matplotlib 3.9+, scikit-learn 1.5+) is done the same way, plus ninja.
  Its editable install rebuilds the changed code on import, with the ninja found at install
  time; built in isolation, that ninja is in a temporary folder that's gone by then. It
  compiles one file at a time, to fit the sandbox's memory, and its tests run with plain
  `pytest` (see standard_recipe());
- a build that caps setuptools (pylint 2.15 asks for setuptools~=62.6) is done the same way,
  without that pin: built in isolation it would get a setuptools too old for editable
  installs, which need version 64;
- Django runs its tests with its own tests/runtests.py, not pytest.

Imported by other modules; the file isn't run directly.
"""

import re
import shlex

from packaging.requirements import InvalidRequirement, Requirement

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.setup_agent.facts import RepoFacts
from pr_reviewer.setup_agent.versions import choose_python

# Extra names that usually hold the test dependencies, best first.
TEST_EXTRAS = ("test", "tests", "testing", "dev")

# Build requirements of a setup.py that imports numpy, when pyproject.toml lists none.
USUAL_BUILD_REQUIRES = ("numpy", "cython", "scipy")

# The pyproject build backend of meson-python.
MESON_BACKEND = "mesonpy"

# The meson-python setting for one compile job at a time; the rebuild on import uses it too.
# ninja otherwise runs 3 jobs on the sandbox's 2 CPUs, and three compilers on scikit-learn's or
# matplotlib's biggest generated files don't fit in 2 GB.
MESON_ONE_JOB = "-C compile-args=-j1"

# A setuptools version far in the future. A build pin that rules it out caps setuptools.
FAR_SETUPTOOLS = "1000"

# Django's test requirements file.
DJANGO_REQUIREMENTS = "tests/requirements/py3.txt"


def caps_setuptools(facts: RepoFacts) -> bool:
    """Tell whether the build requirements put an upper limit on setuptools.

    Args:
        facts (RepoFacts): The repo's facts, with its pyproject build requirements.

    Returns:
        bool: True when a setuptools requirement rules out new versions, such as "~=62.6",
        "<64" or "==62.6.0"; False for a lower limit only, such as ">=40".
    """
    for text in facts.build.build_requires:
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            continue
        specifier = requirement.specifier
        if requirement.name.lower() == "setuptools" and not specifier.contains(FAR_SETUPTOOLS):
            return True
    return False


def build_requirements(facts: RepoFacts) -> list[str]:
    """Give the packages to install before a build without isolation.

    Args:
        facts (RepoFacts): The repo's facts, with its pyproject build requirements.

    Returns:
        list[str]: The requirements, with any setuptools version pin dropped, plus setuptools
        and wheel.
    """
    declared = list(facts.build.build_requires) or list(USUAL_BUILD_REQUIRES)
    unpinned = [r for r in declared if not re.match(r"^\s*(setuptools|wheel)\b", r, re.IGNORECASE)]
    return [*unpinned, "setuptools", "wheel"]


def requirement_files_for_tests(facts: RepoFacts) -> list[str]:
    """Pick the requirement files that hold test dependencies.

    Args:
        facts (RepoFacts): The repo's facts.

    Returns:
        list[str]: Up to two .txt files, test ones first, then dev ones.
    """
    txt = [f for f in facts.requirement_files if f.endswith(".txt")]
    # Django keeps its test requirements in tests/requirements/py3.txt. The other files there
    # (mysql.txt, oracle.txt...) are for database servers the tests don't use by default.
    if DJANGO_REQUIREMENTS in txt:
        return [DJANGO_REQUIREMENTS]
    tests = [f for f in txt if re.search(r"test", f, re.IGNORECASE)]
    dev = [f for f in txt if re.search(r"dev", f, re.IGNORECASE) and f not in tests]
    return (tests or dev)[:2]


def standard_recipe(pr: Instance, facts: RepoFacts) -> EnvRecipe:
    """Write the standard recipe for a PR's repo.

    Args:
        pr (Instance): The PR, for its repo and date.
        facts (RepoFacts): What the repo's files say, at the PR's base.

    Returns:
        EnvRecipe: The recipe.
    """
    python = choose_python(facts.python_requires, facts.ci_pythons, pr.created_at.date())
    extra = next((e for e in TEST_EXTRAS if e in facts.extras), None)
    target = f'".[{extra}]"' if extra else "."
    steps = []
    env: dict[str, str] = {}
    meson = facts.build.build_backend == MESON_BACKEND
    if facts.build.setup_imports_numpy or meson or caps_setuptools(facts):
        needs = build_requirements(facts)
        if meson and not any(re.match(r"^\s*ninja\b", r, re.IGNORECASE) for r in needs):
            needs.append("ninja")
        # Requirement texts come from the repo, so each is quoted for the shell.
        steps.append(f"uv pip install {' '.join(shlex.quote(r) for r in needs)}")
        options = f"{MESON_ONE_JOB} " if meson else ""
        steps.append(f"uv pip install --no-build-isolation {options}-e {target}")
        if facts.build.setup_imports_numpy and not meson:
            env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    else:
        steps.append(f"uv pip install -e {target}")
    # File names come from the repo, so they're quoted for the shell.
    for name in requirement_files_for_tests(facts):
        steps.append(f"uv pip install -r {shlex.quote(name)}")
    if facts.test_runner == "pytest":
        steps.append("uv pip install pytest")
        # `python -m` puts the repo folder on the import path, so pytest can import the source
        # copy of the package instead of the installed one (matplotlib's lib/matplotlib, as
        # lib.matplotlib). A meson-python source folder lacks the files the build makes, such
        # as _version.py, so that import fails.
        command = "pytest" if meson else "python -m pytest"
    else:
        command = "python tests/runtests.py"
    return EnvRecipe(
        repo=pr.repo,
        python_version=python,
        install_steps=tuple(steps),
        test_command=command,
        env=env,
    )
