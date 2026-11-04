"""Choosing a Python version for a repo at a past date.

A PR from March 2020 was written and tested on the Python versions of that time, and its
dependencies' old releases may not install on a much newer Python. So the version must:
- satisfy what the project declares (python_requires), when it declares something;
- have been out for at least a year by the PR's date, so the libraries of the time supported it;
- be one the sandbox can provide: uv installs 3.8 and newer, and 3.7 is built into the image.
Among those, a version the repo's CI ran is preferred, then the newest.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

# When each Python version came out.
RELEASES = {
    "3.6": dt.date(2016, 12, 23),
    "3.7": dt.date(2018, 6, 27),
    "3.8": dt.date(2019, 10, 14),
    "3.9": dt.date(2020, 10, 5),
    "3.10": dt.date(2021, 10, 4),
    "3.11": dt.date(2022, 10, 24),
    "3.12": dt.date(2023, 10, 2),
    "3.13": dt.date(2024, 10, 7),
    "3.14": dt.date(2025, 10, 7),
}

# Versions uv can't install, built into the sandbox image instead: `uv python list
# --all-versions` (uv 0.13.1) lists nothing older than 3.8.
BUILT_IN = {"3.7": "/opt/python/3.7/bin/python3.7"}

# The oldest version a recipe can use.
OLDEST_PYTHON = "3.7"

# How long a Python version must have been out before a PR's date to be chosen.
SETTLING = dt.timedelta(days=365)


def _key(version: str) -> tuple[int, int]:
    """Sort key for "3.x" versions.

    Args:
        version (str): Such as "3.10".

    Returns:
        tuple[int, int]: Such as (3, 10).
    """
    major, minor = version.split(".")
    return int(major), int(minor)


def allowed_by(requires: str | None, version: str) -> bool:
    """Tell whether a declared requirement allows a Python version.

    Args:
        requires (str | None): Such as ">=3.8, <4". None or unreadable allows everything.
        version (str): Such as "3.9".

    Returns:
        bool: True when the version satisfies it.
    """
    if not requires:
        return True
    try:
        spec = SpecifierSet(requires.replace(" ", ""))
    except InvalidSpecifier:
        return True
    # A requirement like ">=3.8.1" is about patch releases; test with a late patch.
    return spec.contains(Version(f"{version}.99"), prereleases=True)


def choose_python(requires: str | None, ci_versions: list[str], when: dt.date) -> str:
    """Pick the Python version for a repo at a date.

    Args:
        requires (str | None): The project's declared Python versions, if any.
        ci_versions (list[str]): Versions its CI ran at that commit, such as ["3.8", "3.9"].
        when (dt.date): The PR's date.

    Returns:
        str: Such as "3.8".
    """
    usable = [v for v in RELEASES if _key(v) >= _key(OLDEST_PYTHON)]
    settled = [v for v in usable if RELEASES[v] + SETTLING <= when and allowed_by(requires, v)]
    if not settled:
        # Nothing old enough fits; take the oldest version the project allows.
        allowed = [v for v in usable if allowed_by(requires, v)]
        return min(allowed or usable, key=_key)
    in_ci = [v for v in settled if v in ci_versions]
    return max(in_ci or settled, key=_key)
