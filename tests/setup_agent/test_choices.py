"""
Tests for the setup agent's fixed choices: the Python version (versions.py), the related tests
(selection.py), the standard recipe (standard.py) and the repo probe (probe.py).

None of them needs Docker: the probe is plain Python, run here on a small folder of files.

How to run (from the project folder):
    pytest tests/setup_agent/test_choices.py
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.setup_agent.facts import BuildFacts, RepoFacts, summary
from pr_reviewer.setup_agent.selection import (
    django_labels,
    fallback_tests,
    main_test_folder,
    related_tests,
    tree_sizes,
)
from pr_reviewer.setup_agent.standard import (
    caps_setuptools,
    requirement_files_for_tests,
    standard_recipe,
)
from pr_reviewer.setup_agent.versions import allowed_by, choose_python
from tests.benchmark.test_dataset import make_row

PROBE = Path(__file__).resolve().parents[2] / "src" / "pr_reviewer" / "setup_agent" / "probe.py"


@pytest.mark.parametrize(
    ("requires", "ci", "when", "expected"),
    [
        (None, [], dt.date(2022, 6, 1), "3.9"),  # 3.10 came out less than a year before
        (None, ["3.7", "3.8"], dt.date(2022, 6, 1), "3.8"),  # the newest CI version wins
        (">=3.6, <3.9", [], dt.date(2023, 1, 1), "3.8"),  # the project's limit wins
        (None, [], dt.date(2019, 3, 1), "3.7"),  # nothing older than the oldest we have
        (">=3.12", [], dt.date(2023, 3, 1), "3.12"),  # too new to have settled: oldest allowed
    ],
)
def test_python_follows_the_project_ci_and_date(
    requires: str | None, ci: list[str], when: dt.date, expected: str
) -> None:
    """Check the version satisfies the project, prefers CI's, and existed a year earlier."""
    assert choose_python(requires, ci, when) == expected


def test_unreadable_requirements_allow_any_version() -> None:
    """Check a garbled python_requires doesn't block every version."""
    assert allowed_by("not a spec", "3.8")
    assert allowed_by(">=3.8.1", "3.8")
    assert not allowed_by(">=3.9", "3.8")


def test_related_tests_prefer_edited_then_name_matched_tests() -> None:
    """Check edited tests come first, then test files named after changed modules."""
    repo = [
        "sklearn/linear_model/_ridge.py",
        "sklearn/linear_model/tests/test_ridge.py",
        "sklearn/svm/tests/test_ridge.py",
        "sklearn/utils/tests/test_validation.py",
        "sklearn/utils/validation.py",
    ]
    changed = ["sklearn/linear_model/_ridge.py", "sklearn/utils/tests/test_validation.py"]
    assert related_tests(changed, repo) == [
        "sklearn/utils/tests/test_validation.py",
        "sklearn/linear_model/tests/test_ridge.py",
        "sklearn/svm/tests/test_ridge.py",
    ]


def test_related_tests_fall_back_to_nearby_and_django_style_folders() -> None:
    """Check the nearest tests folder, then a top-level tests/ folder named after the code."""
    nearby = ["pkg/core/engine.py", "pkg/core/tests/test_misc.py", "pkg/core/tests/conftest.py"]
    assert related_tests(["pkg/core/engine.py"], nearby) == ["pkg/core/tests/test_misc.py"]
    django = ["django/contrib/admin/options.py", "tests/admin_views/tests.py", "tests/x/tests.py"]
    assert related_tests(["django/contrib/admin/options.py"], django) == [
        "tests/admin_views/tests.py"
    ]
    assert django_labels(["tests/admin_views/tests.py", "other.py"]) == ["admin_views.tests"]
    assert related_tests(["README.md"], django) == []


def test_fallback_tests_are_the_main_folders_smallest() -> None:
    """Check the fallback picks the smallest files of the folder with the most test files."""
    repo = ["a/tests/test_1.py", "b/tests/test_3.py", "b/tests/test_2.py", "b/tests/test_4.py"]
    assert main_test_folder(repo) == "b/tests"
    sizes = {"b/tests/test_2.py": 90_000, "b/tests/test_3.py": 800, "b/tests/test_4.py": 500}
    assert fallback_tests(repo, sizes, count=2) == ["b/tests/test_3.py", "b/tests/test_4.py"]
    # With no sizes, path order.
    assert fallback_tests(repo, {}, count=2) == ["b/tests/test_2.py", "b/tests/test_3.py"]
    assert fallback_tests(["setup.py"], {}) == []
    assert main_test_folder(["setup.py"]) is None


def test_tree_sizes_read_git_ls_tree_output() -> None:
    """Check sizes come out by repo path, skipping folders and unreadable lines."""
    listing = (
        "100644 blob 1111111111111111111111111111111111111111    1234\ttest_a.py\n"
        "040000 tree 2222222222222222222222222222222222222222       -\tbaseline_images\n"
        "100644 blob 3333333333333333333333333333333333333333      56\tconftest.py\n"
        "garbage"
    )
    assert tree_sizes(listing, "lib/tests") == {
        "lib/tests/test_a.py": 1234,
        "lib/tests/conftest.py": 56,
    }
    assert tree_sizes(listing, ".") == {"test_a.py": 1234, "conftest.py": 56}


def test_related_tests_find_pylint_style_mirrored_unittest_files() -> None:
    """Check pylint's layout: unittest_*.py files in a tests/ tree mirroring the package."""
    repo = [
        "pylint/checkers/classes/class_checker.py",
        "tests/checkers/unittest_base.py",
        "tests/checkers/unittest_classes.py",
        "tests/checkers/unittest_format.py",
        "tests/checkers/unittest_imports.py",
        "tests/test_self.py",
    ]
    assert related_tests(["pylint/checkers/classes/class_checker.py"], repo) == [
        "tests/checkers/unittest_classes.py",
        "tests/checkers/unittest_base.py",
        "tests/checkers/unittest_format.py",
    ]
    assert related_tests(["pylint/checkers/format.py"], repo) == [
        "tests/checkers/unittest_format.py"
    ]


def facts(**changes: object) -> RepoFacts:
    """Make repo facts with some fields set.

    Args:
        **changes (object): Fields to set.

    Returns:
        RepoFacts: The facts.
    """
    return RepoFacts.model_validate(changes)


def pr_of(year: int = 2022) -> Instance:
    """Make a PR from the given year.

    Args:
        year (int): The year it was opened.

    Returns:
        Instance: The PR.
    """
    return Instance.model_validate_json(json.dumps(make_row(1, year=year, issues=())))


def test_standard_recipe_installs_the_test_extra_and_requirements() -> None:
    """Check the usual install: editable with the test extra, the test requirements, pytest."""
    recipe = standard_recipe(
        pr_of(2022),
        facts(
            extras=["docs", "tests"],
            requirement_files=["requirements/docs.txt", "requirements/test.txt", "x; rm.txt"],
            ci_pythons=["3.8", "3.9"],
        ),
    )
    assert recipe.python_version == "3.9"
    assert recipe.install_steps == (
        'uv pip install -e ".[tests]"',
        "uv pip install -r requirements/test.txt",
        "uv pip install pytest",
    )
    assert recipe.test_command == "python -m pytest"


def test_standard_recipe_handles_numpy_builds_and_django() -> None:
    """Check build requirements go in first for numpy and meson builds; Django uses runtests.py.

    A meson build also compiles one file at a time and tests with plain pytest.
    """
    numpy = standard_recipe(pr_of(), facts(build=BuildFacts(setup_imports_numpy=True)))
    assert numpy.install_steps[:2] == (
        "uv pip install numpy cython scipy setuptools wheel",
        "uv pip install --no-build-isolation -e .",
    )
    assert numpy.env == {"SETUPTOOLS_USE_DISTUTILS": "stdlib"}
    mesonpy = BuildFacts(
        build_backend="mesonpy", build_requires=["meson-python>=0.13.1", "setuptools_scm>=7"]
    )
    meson = standard_recipe(pr_of(2024), facts(build=mesonpy))
    assert meson.install_steps[:2] == (
        "uv pip install 'meson-python>=0.13.1' 'setuptools_scm>=7' setuptools wheel ninja",
        "uv pip install --no-build-isolation -C compile-args=-j1 -e .",
    )
    assert meson.env == {}
    assert meson.test_command == "pytest"
    assert numpy.test_command == "python -m pytest"
    declared = BuildFacts(
        setup_imports_numpy=True,
        build_requires=["setuptools<60.0", "wheel", "Cython>=0.28.5", "oldest-supported-numpy"],
    )
    sklearn = standard_recipe(pr_of(), facts(build=declared))
    assert sklearn.install_steps[0] == (
        "uv pip install 'Cython>=0.28.5' oldest-supported-numpy setuptools wheel"
    )
    django = standard_recipe(
        pr_of(), facts(test_runner="django", requirement_files=["tests/requirements/py3.txt"])
    )
    assert django.test_command == "python tests/runtests.py"
    assert "uv pip install -r tests/requirements/py3.txt" in django.install_steps
    assert "uv pip install pytest" not in django.install_steps


def test_a_setuptools_cap_is_dropped_and_built_without_isolation() -> None:
    """Check pylint's setuptools~=62.6: no pin, no isolation, no numpy setting; >=40 is kept."""
    capped = BuildFacts(build_requires=["setuptools~=62.6", "wheel~=0.37.1"])
    assert caps_setuptools(facts(build=capped))
    pylint = standard_recipe(pr_of(2022), facts(build=capped))
    assert pylint.install_steps[:2] == (
        "uv pip install setuptools wheel",
        "uv pip install --no-build-isolation -e .",
    )
    assert pylint.env == {}
    assert pylint.test_command == "python -m pytest"
    for pins in (["setuptools<64"], ["setuptools==62.6.0"], ["Setuptools >=61, <70"]):
        assert caps_setuptools(facts(build=BuildFacts(build_requires=pins))), pins
    for pins in (["setuptools>=40", "setuptools_scm"], ["setuptools"], ["not a requirement!"]):
        assert not caps_setuptools(facts(build=BuildFacts(build_requires=pins))), pins
    lower = standard_recipe(pr_of(), facts(build=BuildFacts(build_requires=["setuptools>=40"])))
    assert lower.install_steps[0] == "uv pip install -e ."


def test_requirement_file_names_are_quoted_for_the_shell() -> None:
    """Check a repo-made file name can't add a shell command to a step."""
    recipe = standard_recipe(pr_of(), facts(requirement_files=["test; rm -rf ~.txt"]))
    assert "uv pip install -r 'test; rm -rf ~.txt'" in recipe.install_steps
    assert requirement_files_for_tests(facts(requirement_files=["a.txt", "dev.txt"])) == ["dev.txt"]


def test_django_installs_only_its_py3_requirements() -> None:
    """Check Django's database-server files (mysql.txt, oracle.txt...) are left out."""
    folder = "tests/requirements/"
    files = [folder + name for name in ("mysql.txt", "oracle.txt", "postgres.txt", "py3.txt")]
    assert requirement_files_for_tests(facts(requirement_files=files)) == [folder + "py3.txt"]


def test_probe_reads_a_repos_setup_files(tmp_path: Path) -> None:
    """Check the probe on a small repo: declared Python, CI versions, extras, files, numpy."""
    (tmp_path / "setup.py").write_text(
        "import numpy\nsetup(python_requires='>=3.7', extras_require={'test': ['pytest']})\n",
        encoding="utf-8",
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "jobs:\n  t:\n    strategy:\n      matrix:\n        python-version: ['3.8', '3.10']\n"
        "    steps:\n      - run: pip install -e .[test]\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "test.txt").write_text("pytest\n", encoding="utf-8")
    output = subprocess.run(
        [sys.executable, "-I", str(PROBE)], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    found = RepoFacts.model_validate_json(output)
    assert found.python_requires == ">=3.7"
    assert found.ci_pythons == ["3.8", "3.10"]
    assert found.ci_install_lines == ["run: pip install -e .[test]"]
    assert found.extras == ["test"]
    assert found.requirement_files == ["requirements/test.txt"]
    assert found.build.setup_imports_numpy
    assert found.test_runner == "pytest"
    text = summary(found)
    assert "Declared Python versions: >=3.7" in text
    assert "--- setup.py ---" in text
