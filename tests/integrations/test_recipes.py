"""Tests for carrying recipes between Action runs (integrations/recipes.py).

Only the file checks; saving and loading go through the recipe cache, tested in setup_agent.

How to run (from the project folder):
    pytest tests/integrations/test_recipes.py
"""

import json
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.integrations.recipes import RecipeFileError, read_recipes


def item(repo: str = "octo/demo", source: str = "standard") -> dict[str, Any]:
    """Build one entry of a recipe file.

    Args:
        repo (str): The recipe's repo.
        source (str): Where it came from.

    Returns:
        dict[str, Any]: The entry.
    """
    recipe = {
        "repo": repo,
        "python_version": "3.12",
        "install_steps": ['uv pip install -e ".[test]"', "uv pip install pytest"],
        "test_command": "python -m pytest",
    }
    return {"year": 2026, "source": source, "recipe": recipe}


def write(tmp_path: Path, content: object) -> Path:
    """Write a recipe file.

    Args:
        tmp_path (Path): pytest's temporary folder.
        content (object): The JSON content.

    Returns:
        Path: The file.
    """
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_a_valid_file_is_read(tmp_path: Path) -> None:
    """Check the year, source and recipe come back."""
    [(year, source, recipe)] = read_recipes(write(tmp_path, [item()]), "octo/demo")
    assert (year, source, recipe.python_version) == (2026, "standard", "3.12")


@pytest.mark.parametrize(
    "content",
    [
        {"not": "a list"},
        [item(repo="evil/other")],
        [item(source="model")],
        [{"year": 2026, "source": "agent", "recipe": {"repo": "octo/demo"}}],
    ],
)
def test_a_bad_file_is_refused(tmp_path: Path, content: object) -> None:
    """Check a file of another shape, another repo's recipe, or a broken recipe is refused."""
    with pytest.raises(RecipeFileError):
        read_recipes(write(tmp_path, content), "octo/demo")
