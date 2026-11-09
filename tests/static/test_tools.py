"""Tests for the static tools' commands, settings and result files (static/tools.py).

Checks, without Docker, that each tool's command uses the fixed rules and ignores the repo's
own settings, that target versions are raised to what a tool accepts, that the settings key
changes with the rules and the tools image, that the tools image's tag covers its pinned
requirements, and that a result survives a round trip through its file.

How to run (from the project folder):
    pytest tests/static/test_tools.py
"""

from pathlib import Path

from pr_reviewer.sandbox.image import ImageSpec
from pr_reviewer.static.tools import (
    TOOLS_DIR,
    StaticResult,
    StaticSettings,
    ToolRun,
    _version_flag,
    load_result,
    save_result,
    settings_key,
    tool_command,
)
from tests.static.test_warnings import warning

FILES = ["pkg/cart.py", "pkg/io.py"]


def test_each_command_uses_fixed_rules_and_ignores_repo_settings() -> None:
    """Check the rules and the options that keep each tool off the repo's own config."""
    settings = StaticSettings()
    ruff = tool_command("ruff", FILES, "/tmp/r.json", "3.10", settings)
    assert ruff[:2] == [f"{TOOLS_DIR}/bin/ruff", "check"]
    assert "--isolated" in ruff and ruff[ruff.index("--select") + 1] == "F,B,PLE"
    assert ruff[ruff.index("--target-version") + 1] == "py310"
    assert ruff[-3:] == ["--", *FILES]
    bandit = tool_command("bandit", FILES, "/tmp/b.json", "3.10", settings)
    assert bandit[bandit.index("--severity-level") + 1] == "medium"
    semgrep = tool_command("semgrep", FILES, "/tmp/s.json", "3.10", settings)
    assert semgrep[semgrep.index("--config") + 1] == f"{TOOLS_DIR}/semgrep-rules"
    assert semgrep[semgrep.index("--metrics") + 1] == "off"
    mypy = tool_command("mypy", FILES, "/tmp/m.json", "3.8", settings, "skip")
    assert mypy[:4] == ["sh", "-c", 'out="$1"; shift; exec "$@" > "$out"', "sh"]
    assert mypy[4] == "/tmp/m.json"
    assert mypy[mypy.index("--config-file") + 1] == f"{TOOLS_DIR}/mypy.ini"
    assert mypy[mypy.index("--python-executable") + 1] == "/work/venv/bin/python"
    # mypy can't target 3.8, so the oldest version it accepts is used.
    assert mypy[mypy.index("--python-version") + 1] == "3.10"
    assert mypy[mypy.index("--follow-imports") + 1] == "skip"


def test_target_versions_are_raised_to_the_oldest_accepted() -> None:
    """Check a version below a tool's oldest is raised, and others are kept."""
    assert _version_flag("3.7", (3, 9)) == "3.9"
    assert _version_flag("3.12", (3, 9)) == "3.12"


def test_the_settings_key_changes_with_rules_and_tools() -> None:
    """Check other rules or another tools image give another key."""
    key = settings_key(StaticSettings(), "pr-reviewer-tools:aaa")
    assert key == settings_key(StaticSettings(), "pr-reviewer-tools:aaa")
    assert key != settings_key(StaticSettings(ruff_select="F"), "pr-reviewer-tools:aaa")
    assert key != settings_key(StaticSettings(), "pr-reviewer-tools:bbb")


def test_the_image_tag_covers_the_files_the_build_copies(tmp_path: Path) -> None:
    """Check editing requirements.txt changes the tag, and files are only counted if named."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM x\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("ruff==1\n", encoding="utf-8")
    plain = ImageSpec("img", dockerfile, "base:1")
    with_files = ImageSpec("img", dockerfile, "base:1", ("requirements.txt",))
    before = with_files.tag
    assert plain.tag != before
    (tmp_path / "requirements.txt").write_text("ruff==2\n", encoding="utf-8")
    assert with_files.tag != before
    assert plain.tag == ImageSpec("img", dockerfile, "base:1").tag


def test_a_result_survives_its_file(tmp_path: Path) -> None:
    """Check a result written to its file reads back the same."""
    result = StaticResult(
        instance_id="psf__requests-1",
        version=1,
        image="pr-reviewer-pr:psf__requests-1-pr",
        base="a" * 40,
        head="b" * 40,
        changed={"pkg/cart.py": [3, 4]},
        runs=[ToolRun(tool="mypy", status="ok", seconds=1.5, mode="silent", warnings=2)],
        warnings=[warning(line=3), warning(line=20)],
        kept=[warning(line=3)],
        seconds=9.0,
    )
    path = tmp_path / "static" / "psf__requests-1.json"
    save_result(path, result)
    assert load_result(path) == result
