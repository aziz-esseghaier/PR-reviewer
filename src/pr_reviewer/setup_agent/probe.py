"""Repo probe: read a repo's setup files and print what they say about installing and testing it.

This file runs inside a sandbox, with the sandbox image's own Python and its standard library
only; inspect.py copies it in and runs it at the repo root. It reads the files that tell how a
repo wants to be installed and tested, and prints one JSON object:
- python_requires: the Python versions the project declares (pyproject, setup.cfg, setup.py);
- ci_pythons: Python versions named in its CI config (GitHub Actions, Azure, Travis, tox);
- ci_install_lines: the install commands its CI runs;
- requirement_files: requirement and environment files, such as requirements/test.txt;
- extras: optional dependency groups, such as "test";
- build: how it's built (pyproject, setup.py, meson, and whether setup.py imports numpy);
- test_runner: "pytest", or "django" for Django's own runner;
- excerpts: short pieces of these files, for the setup agent's model to read.

Everything it reads is the repo's own text, so the caller treats the output as untrusted data.

How to run (inside a sandbox, at a repo root):
    python3 probe.py
"""

import configparser
import json
import re
import sys
import tomllib
from pathlib import Path

# Most lines kept per excerpt, and most characters per line.
EXCERPT_LINES = 25
LINE_CHARS = 160

# Where CI configs usually live.
CI_GLOBS = (
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "azure-pipelines.yml",
    ".azure-pipelines/*.yml",
    ".travis.yml",
    ".circleci/config.yml",
    "tox.ini",
)

# Requirement and environment files, at the root and one level down.
REQUIREMENT_GLOBS = (
    "requirements*.txt",
    "*requirements*.txt",
    "requirements/*.txt",
    "requirements/*.in",
    "tests/requirements/*.txt",
    "ci/requirements/*.yml",
    "ci/*.yml",
    "environment*.yml",
    "build_tools/*requirements*.txt",
)


def read(path: Path) -> str:
    """Read a text file, or give "" when it's missing or unreadable.

    Args:
        path (Path): The file.

    Returns:
        str: Its text.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def excerpt(text: str, pattern: str | None = None) -> str:
    """Cut a short piece of a file: the first lines, or the lines around a pattern.

    Args:
        text (str): The file's text.
        pattern (str | None): A regular expression; the excerpt starts at its first match.

    Returns:
        str: At most EXCERPT_LINES lines of at most LINE_CHARS characters.
    """
    lines = text.splitlines()
    start = 0
    if pattern is not None:
        for number, line in enumerate(lines):
            if re.search(pattern, line):
                start = number
                break
    return "\n".join(line[:LINE_CHARS] for line in lines[start : start + EXCERPT_LINES])


def python_requires(root: Path) -> str | None:
    """Find the Python versions the project declares.

    Args:
        root (Path): The repo root.

    Returns:
        str | None: Such as ">=3.8", or None when nothing declares it.
    """
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(read(pyproject))
            value = data.get("project", {}).get("requires-python")
            if isinstance(value, str):
                return value
        except tomllib.TOMLDecodeError:
            pass
    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(read(setup_cfg))
            if parser.has_option("options", "python_requires"):
                return parser.get("options", "python_requires").strip()
        except configparser.Error:
            pass
    match = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", read(root / "setup.py"))
    return match.group(1) if match else None


def extras(root: Path) -> list[str]:
    """List the project's optional dependency groups.

    Args:
        root (Path): The repo root.

    Returns:
        list[str]: Group names, such as ["docs", "test"].
    """
    found: set[str] = set()
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(read(pyproject))
            found.update(data.get("project", {}).get("optional-dependencies", {}))
        except tomllib.TOMLDecodeError:
            pass
    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(read(setup_cfg))
            if parser.has_section("options.extras_require"):
                found.update(parser.options("options.extras_require"))
        except configparser.Error:
            pass
    setup_py = read(root / "setup.py")
    block = re.search(r"extras_require\s*=\s*\{(.*?)\}", setup_py, re.DOTALL)
    if block:
        found.update(re.findall(r"['\"]([A-Za-z0-9_.-]+)['\"]\s*:", block.group(1)))
    return sorted(found)


def ci_facts(root: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """Read the CI configs for Python versions and install commands.

    Args:
        root (Path): The repo root.

    Returns:
        tuple[list[str], list[str], dict[str, str]]: The Python versions named, such as
        ["3.8", "3.9"]; up to 10 install commands; and an excerpt of each config.
    """
    versions: set[str] = set()
    installs: list[str] = []
    excerpts: dict[str, str] = {}
    for pattern in CI_GLOBS:
        for path in sorted(root.glob(pattern))[:6]:
            text = read(path)
            name = path.relative_to(root).as_posix()
            for line in text.splitlines():
                lowered = line.lower()
                if "python" in lowered or "py3" in lowered or "envlist" in lowered:
                    versions.update(re.findall(r"(?<![\d.])3\.(\d{1,2})(?![\d])", line))
                    versions.update(re.findall(r"\bpy3(\d{1,2})\b", line))
                if "pip install" in lowered or "uv pip" in lowered:
                    command = line.strip().lstrip("-").strip()
                    if command and command not in installs:
                        installs.append(command[:LINE_CHARS])
            excerpts[name] = excerpt(text, r"pip install|python-version|envlist")
    return sorted({f"3.{v}" for v in versions}, key=lambda v: int(v[2:])), installs[:10], excerpts


def requirement_files(root: Path) -> tuple[list[str], dict[str, str]]:
    """List the requirement and environment files, with short excerpts of test-related ones.

    Args:
        root (Path): The repo root.

    Returns:
        tuple[list[str], dict[str, str]]: The files, and excerpts of up to four of them.
    """
    files = sorted(
        {path.relative_to(root).as_posix() for p in REQUIREMENT_GLOBS for path in root.glob(p)}
    )
    testy = [f for f in files if re.search(r"test|dev|ci|environment", f)] or files
    return files[:30], {f: excerpt(read(root / f)) for f in testy[:4]}


def build_facts(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    """Describe how the project is built.

    Args:
        root (Path): The repo root.

    Returns:
        tuple[dict[str, object], dict[str, str]]: The facts, and excerpts of the build files.
    """
    pyproject = read(root / "pyproject.toml")
    setup_py = read(root / "setup.py")
    facts: dict[str, object] = {
        "pyproject": bool(pyproject),
        "setup_py": bool(setup_py),
        "setup_cfg": (root / "setup.cfg").is_file(),
        "meson": (root / "meson.build").is_file(),
        "setup_imports_numpy": bool(re.search(r"^\s*(import numpy|from numpy)", setup_py, re.M)),
        "build_backend": None,
        "build_requires": [],
    }
    if pyproject:
        try:
            system = tomllib.loads(pyproject).get("build-system", {})
            facts["build_backend"] = system.get("build-backend")
            facts["build_requires"] = list(system.get("requires", []))[:15]
        except tomllib.TOMLDecodeError:
            pass
    excerpts = {}
    if pyproject:
        excerpts["pyproject.toml"] = excerpt(pyproject, r"\[build-system\]|\[project\]")
    if setup_py:
        excerpts["setup.py"] = excerpt(setup_py, r"install_requires|setup\(")
    return facts, excerpts


def main() -> int:
    """Probe the repo in the current folder and print the facts as JSON.

    Returns:
        int: Exit code, always 0.
    """
    root = Path.cwd()
    pythons, installs, ci_excerpts = ci_facts(root)
    files, file_excerpts = requirement_files(root)
    build, build_excerpts = build_facts(root)
    runner = "django" if (root / "tests" / "runtests.py").is_file() else "pytest"
    facts = {
        "python_requires": python_requires(root),
        "ci_pythons": pythons,
        "ci_install_lines": installs,
        "requirement_files": files,
        "extras": extras(root),
        "build": build,
        "test_runner": runner,
        "excerpts": {**build_excerpts, **file_excerpts, **dict(list(ci_excerpts.items())[:3])},
    }
    json.dump(facts, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
