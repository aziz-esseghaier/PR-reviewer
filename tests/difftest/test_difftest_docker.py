"""
End-to-end tests of the differential runner (difftest/) in real sandboxes.

A tiny package is built inside a container, with three commits:
- base: add() and name();
- the PR: breaks add() (a - b) and adds mul(), with a test for it, and a test in a base class
  that only a Test class inheriting it runs;
- a later fix: repairs add() and adds a test for it, like a known bug's fix in SWR-Bench.
Then:
- both sides are prepared as images, and found again only while the recipe matches;
- one run gives every outcome: pass_fail (add), fail_pass (mul, an AttributeError at the
  base), pass_pass, fail_fail, flaky (a test that fails on every second run), a test file
  that doesn't import at the base (an ImportError, from collection), and one skipped while
  loading (not_run, skipped);
- watched lines (the PR's changed lines) show which ones each test ran, and a failure keeps
  its traceback and its exception's classes;
- a test ID pytest can't find stops its whole run, which shows as a problem;
- a test that hangs is not_run, with the timeout as the reason;
- the validation cases are built from git: the PR's own new tests (mul, and the base class's
  test under the class that runs it) and the fix's test (add), which fails at the PR state
  and passes at the fix: the bug is caught, and provable from the base.
These need Docker, the images built, and an internet connection (PyPI, through the proxy),
and take several minutes.

How to run (from the project folder):
    pytest tests/difftest/test_difftest_docker.py
"""

import json
from collections.abc import Iterator

import pytest
from docker import DockerClient

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.difftest.environments import (
    PrEnvironment,
    PreparedSide,
    find_prepared,
    prepare,
)
from pr_reviewer.difftest.runner import DiffResult, FileSpec, SideSpec, run_diff
from pr_reviewer.difftest.validation import build_cases, judge_case
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.repos import update_repo
from pr_reviewer.sandbox.runner import Sandbox
from pr_reviewer.setup_agent.attempt import TryContext
from tests.benchmark.test_dataset import make_row

pytestmark = pytest.mark.docker

REPO = "owner/difftoy"
UPSTREAM = "/repos/_upstream/difftoy"

RECIPE = EnvRecipe(
    repo=REPO,
    python_version="3.9",
    install_steps=("uv pip install -e .", "uv pip install pytest"),
    test_command="python -m pytest",
)

# Builds the package's repo: base, the PR, then a fix; prints the three commit IDs.
BUILD = f"""
set -e
rm -rf {UPSTREAM} /repos/owner__difftoy.git
mkdir -p {UPSTREAM}/difftoy {UPSTREAM}/tests && cd {UPSTREAM}
git init -q -b main && git config user.email t@example.com && git config user.name T
cat > setup.py <<'EOF'
from setuptools import setup
setup(name="difftoy", version="1.0", packages=["difftoy"])
EOF
cat > difftoy/__init__.py <<'EOF'
def add(a, b):
    return a + b


def name():
    return "toy"
EOF
cat > tests/test_difftoy.py <<'EOF'
import difftoy


def test_name():
    assert difftoy.name() == "toy"
EOF
git add -A && git commit -q -m base && BASE=$(git rev-parse HEAD)
cat > difftoy/__init__.py <<'EOF'
def add(a, b):
    return a - b


def name():
    return "toy"


def mul(a, b):
    return a * b
EOF
cat >> tests/test_difftoy.py <<'EOF'


def test_mul():
    assert difftoy.mul(2, 3) == 6


class NameChecks:
    def test_inherited(self):
        assert difftoy.name() == "toy"


class TestNames(NameChecks):
    pass
EOF
git commit -q -am "Add mul" && PR=$(git rev-parse HEAD)
sed -i 's/a - b/a + b/' difftoy/__init__.py
cat >> tests/test_difftoy.py <<'EOF'


def test_add():
    assert difftoy.add(2, 2) == 4
EOF
git commit -q -am "Fix add" && FIX=$(git rev-parse HEAD)
git update-ref refs/pull/1/head $FIX
echo BASE=$BASE PR=$PR FIX=$FIX
"""

# Tests written at both sides: one for each outcome.
CASES = """\
import os

import difftoy


def test_add():
    assert difftoy.add(2, 2) == 4


def test_mul():
    assert difftoy.mul(2, 3) == 6


def test_name():
    assert difftoy.name() == "toy"


def test_wrong():
    assert difftoy.name() == "other"


def test_flaky():
    path = "/tmp/flaky_count"
    count = int(open(path).read()) if os.path.exists(path) else 0
    with open(path, "w") as handle:
        handle.write(str(count + 1))
    assert count != 1
"""

# A test file that doesn't import at the base, where mul() doesn't exist yet.
IMPORTS = """\
from difftoy import mul


def test_import():
    assert mul(1, 1) == 1
"""

# A test file skipped while loading, like one needing an optional package that isn't there.
SKIPS = """\
import pytest

pytest.importorskip("no_such_optional_package")


def test_optional():
    pass
"""


@pytest.fixture(scope="module")
def toy(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> Iterator[tuple[TryContext, Instance, dict[PreparedSide, str], str]]:
    """Build the package's repo, describe its PR and prepare both sides' images.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.

    Yields:
        tuple[TryContext, Instance, dict[PreparedSide, str], str]: The context, the PR, its
        prepared images and the fix commit.
    """
    with Sandbox(
        docker, test_config, sandbox_image, purpose="test-writer", repos_writable=True
    ) as box:
        built = box.run(["sh", "-c", BUILD], timeout=120)
        assert built.ok, built.output
        sha = dict(pair.split("=") for pair in built.output.split())
        row = make_row(1, year=2022, issues=("F.2 Logic",), repo=REPO)
        row["pr_commits"][0]["sha"] = sha["PR"]
        row["pr_commits"][0]["diff"] = [
            {"file": "difftoy/__init__.py", "patch": "@@"},
            {"file": "tests/test_difftoy.py", "patch": "@@"},
        ]
        row["all_commits"][-1]["sha"] = sha["FIX"]
        row["changes"][0]["change_introducing"]["commit_sha"] = sha["PR"]
        row["changes"][0]["change_resolve_info"]["commit_sha"] = sha["FIX"]
        pr = Instance.model_validate_json(json.dumps(row))
        update_repo(box, REPO, [pr], timeout=120, source_url=lambda _: f"file://{UPSTREAM}")
    ctx = TryContext(docker, test_config, sandbox_image, None)
    images = prepare(ctx, pr, PrEnvironment(pr.instance_id, RECIPE, None))
    yield ctx, pr, images, sha["FIX"]
    for image in docker.images.list(filters={"label": f"pr-reviewer.pr={pr.instance_id}"}):
        if image.id:
            docker.images.remove(image.id, force=True)


def by_test(result: DiffResult) -> dict[str, str]:
    """Give each test's outcome by its function name.

    Args:
        result (DiffResult): The run.

    Returns:
        dict[str, str]: Outcomes, by the part of the test ID after the last "::".
    """
    return {test.test.rsplit("::", 1)[-1]: test.outcome for test in result.tests}


def test_prepared_images_are_found_while_the_recipe_matches(
    toy: tuple[TryContext, Instance, dict[PreparedSide, str], str],
) -> None:
    """Check both sides' images exist, and a changed recipe asks for new ones."""
    ctx, pr, images, _ = toy
    assert images == {
        "base": "pr-reviewer-pr:owner__difftoy-1-base",
        "pr": "pr-reviewer-pr:owner__difftoy-1-pr",
    }
    assert find_prepared(ctx, pr, RECIPE) == images
    other = RECIPE.model_copy(update={"python_version": "3.10"})
    assert find_prepared(ctx, pr, other) is None


def test_one_run_gives_every_outcome_with_its_failures(
    toy: tuple[TryContext, Instance, dict[PreparedSide, str], str],
) -> None:
    """Check the four outcomes, a flaky test, a file that doesn't import at the base and one
    skipped while loading. Asking for a test inside either by ID stops pytest, so the runner
    runs the others again."""
    ctx, pr, images, _ = toy
    files = (
        FileSpec("tests/test_probe_cases.py", CASES),
        FileSpec("tests/test_probe_import.py", IMPORTS),
        FileSpec("tests/test_probe_skip.py", SKIPS),
    )
    result = run_diff(
        ctx,
        pr,
        RECIPE,
        SideSpec("base", images["base"]),
        SideSpec("pr", images["pr"]),
        files,
        (
            "tests/test_probe_cases.py",
            "tests/test_probe_import.py::test_import",
            "tests/test_probe_skip.py::test_optional",
        ),
    )
    assert by_test(result) == {
        "test_add": "pass_fail",
        "test_mul": "fail_pass",
        "test_name": "pass_pass",
        "test_wrong": "fail_fail",
        "test_flaky": "flaky",
        "test_import": "fail_pass",
        "test_optional": "not_run",
    }
    tests = {test.test.rsplit("::", 1)[-1]: test for test in result.tests}
    broken = tests["test_add"].new.failure
    assert broken is not None
    assert (broken.exc_type, broken.where) == ("AssertionError", "tests/test_probe_cases.py:7")
    missing = tests["test_mul"].old.failure
    assert missing is not None and missing.exc_type == "AttributeError"
    unimported = tests["test_import"].old.failure
    assert unimported is not None
    assert (unimported.exc_type, unimported.phase) == ("ImportError", "collect")
    assert tests["test_flaky"].old.statuses == ("passed", "failed", "passed")
    assert (tests["test_optional"].old.reason, tests["test_optional"].new.reason) == (
        "skipped",
        "skipped",
    )
    assert result.problems == {}


def test_watched_lines_show_which_changed_lines_a_test_ran(
    toy: tuple[TryContext, Instance, dict[PreparedSide, str], str],
) -> None:
    """Check the plugin notes which watched lines (here the PR's add() and mul()) each test
    ran at the PR, and gives a failure's traceback with paths relative to the repo."""
    ctx, pr, images, _ = toy
    result = run_diff(
        ctx,
        pr,
        RECIPE,
        SideSpec("base", images["base"]),
        SideSpec("pr", images["pr"]),
        (FileSpec("tests/test_probe_cases.py", CASES),),
        ("tests/test_probe_cases.py::test_add", "tests/test_probe_cases.py::test_name"),
        runs=1,
        watch={"difftoy/__init__.py": [2, 9, 10]},
    )
    tests = {test.test.rsplit("::", 1)[-1]: test for test in result.tests}
    assert tests["test_add"].new.hits == ("difftoy/__init__.py:2",)
    assert tests["test_name"].new.hits == (), "name() is on none of the watched lines"
    failure = tests["test_add"].new.failure
    assert failure is not None and failure.frames == ("tests/test_probe_cases.py:7",)
    assert failure.classes[:2] == ("AssertionError", "Exception")


def test_a_test_id_pytest_cant_find_stops_the_run(
    toy: tuple[TryContext, Instance, dict[PreparedSide, str], str],
) -> None:
    """Check pytest refusing an ID (it stops the whole run) is a problem, not "not collected"."""
    ctx, pr, images, _ = toy
    result = run_diff(
        ctx,
        pr,
        RECIPE,
        SideSpec("base", images["base"]),
        SideSpec("pr", images["pr"]),
        (),
        ("tests/test_difftoy.py::test_name", "tests/test_difftoy.py::test_not_there"),
        runs=1,
    )
    assert {test.outcome for test in result.tests} == {"not_run"}
    assert {test.old.reason for test in result.tests} == {"usage_error"}
    assert "not found" in result.problems["base"][0]


def test_a_hanging_test_is_not_run(
    toy: tuple[TryContext, Instance, dict[PreparedSide, str], str],
) -> None:
    """Check a test that never ends is stopped at the time limit and counts as not_run."""
    ctx, pr, images, _ = toy
    hang = "import time\n\n\ndef test_hang():\n    time.sleep(600)\n"
    result = run_diff(
        ctx,
        pr,
        RECIPE,
        SideSpec("base", images["base"]),
        SideSpec("pr", images["pr"]),
        (FileSpec("tests/test_probe_hang.py", hang),),
        ("tests/test_probe_hang.py::test_hang",),
        runs=1,
        timeout=10,
    )
    (test,) = result.tests
    assert (test.outcome, test.old.reason) == ("not_run", "timeout")
    assert set(result.problems) == {"base", "pr"}


def test_a_known_bug_is_caught_by_its_fix_test(
    toy: tuple[TryContext, Instance, dict[PreparedSide, str], str],
) -> None:
    """Check the cases built from git, and that the fix's test catches the PR's bug."""
    ctx, pr, images, fix = toy
    cases = {case.kind: case for case in build_cases(ctx, pr, "pytest")[0]}
    assert set(cases) == {"pr_tests", "fix_tests"}
    assert cases["pr_tests"].targets == (
        "tests/test_difftoy.py::test_mul",
        "tests/test_difftoy.py::TestNames::test_inherited",
    )
    fix_case = cases["fix_tests"]
    assert (fix_case.targets, fix_case.fix_commit) == (("tests/test_difftoy.py::test_add",), fix)
    base, head = SideSpec("base", images["base"]), SideSpec("pr", images["pr"])
    base_pr = run_diff(ctx, pr, RECIPE, base, head, fix_case.files, fix_case.targets, runs=1)
    pr_fix = run_diff(
        ctx,
        pr,
        RECIPE,
        head,
        SideSpec("fix", images["pr"], fix),
        fix_case.files,
        fix_case.targets,
        runs=1,
    )
    judged = judge_case(fix_case, REPO, base_pr, pr_fix)
    assert (judged.caught, judged.provable, judged.unexpected) == (True, 1, [])
    own = cases["pr_tests"]
    own_run = run_diff(ctx, pr, RECIPE, base, head, own.files, own.targets, runs=1)
    assert by_test(own_run) == {"test_mul": "fail_pass", "test_inherited": "pass_pass"}
