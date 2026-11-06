"""The candidate context of a PR: what the probe found, checked, and kept in a file.

The probe (probe.py) answers from inside a sandbox that holds untrusted PR code, so its answer
is checked here before anything uses it: known piece kinds, sane line numbers, paths inside the
repo, and sizes within limits. The checked candidates are saved as one JSON file per PR under
data/contexts/<settings key>/, and the database (context/store.py) keeps the path.

Imported by other modules; the file isn't run directly.
"""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The kinds of pieces, in the order they're packed into a request.
PieceKind = Literal["changed", "called", "caller", "test", "changed_test", "file"]
KIND_ORDER: tuple[PieceKind, ...] = (
    "changed",
    "called",
    "caller",
    "test",
    "changed_test",
    "file",
)

# Most pieces accepted in one answer. The probe's caps keep real answers far below it.
MAX_PIECES = 2000

# Longest file path or qualified name accepted.
MAX_NAME = 500

# Longest piece text accepted. The probe skips pieces bigger than the request budget, which
# is far smaller.
MAX_TEXT = 200_000

# The probe's answer for a commit ID.
SHA_PATTERN = r"^[0-9a-f]{40}$"


def _check_path(path: str) -> str:
    """Refuse a path that would point outside the repo.

    Args:
        path (str): A path relative to the repo root.

    Returns:
        str: The unchanged path.

    Raises:
        ValueError: If it's empty, absolute, goes up with "..", or uses backslashes.
    """
    parts = path.split("/")
    if not path or path.startswith("/") or ".." in parts or "\\" in path:
        raise ValueError(f"paths must stay inside the repo: {path!r}")
    return path


class Piece(BaseModel):
    """One piece of context: some lines of a file, and why they're relevant.

    Attributes:
        kind (PieceKind): changed (a source function the PR changed, whole), called (a
            definition a changed function calls), caller (a function calling a changed one),
            test (an existing test using a changed function), changed_test (a test function
            the PR edited, whole), or file (a changed file, whole).
        file (str): The file, relative to the repo root.
        start (int): First line of the function (or file) the piece comes from.
        end (int): Last line of it.
        qualname (str): The function the piece comes from, such as "Dataset.where"; "" for a
            whole file or module-level lines.
        target (str): The changed function it relates to ("" for a whole file).
        rank (int): Its order among the pieces of its kind for the same changed function.
        whole (bool): Whether the text is the whole function (False: signature and docstring,
            or the lines around a call).
        text (str): The code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PieceKind
    file: str = Field(max_length=MAX_NAME)
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    qualname: str = Field(max_length=MAX_NAME)
    target: str = Field(max_length=MAX_NAME)
    rank: int = Field(ge=0)
    whole: bool
    text: str = Field(max_length=MAX_TEXT)

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Check the path and the line range.

        Returns:
            Self: The unchanged piece.

        Raises:
            ValueError: If the path leaves the repo or the lines are out of order.
        """
        _check_path(self.file)
        if self.end < self.start:
            raise ValueError(f"lines out of order: {self.start}-{self.end}")
        return self

    def label(self) -> str:
        """Name the piece in a few words, for logs and summaries.

        Returns:
            str: Such as "caller xarray/core/merge.py:120-150 merge_core".
        """
        name = f" {self.qualname}" if self.qualname else ""
        return f"{self.kind} {self.file}:{self.start}-{self.end}{name}"


class ChangedFunction(BaseModel):
    """A function or method the PR changed, at its last commit.

    Attributes:
        file (str): The file.
        qualname (str): Its qualified name.
        start (int): First line, decorators included.
        end (int): Last line.
        fully_added (bool): Whether the PR added all of it (the diff then shows it whole).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(max_length=MAX_NAME)
    qualname: str = Field(max_length=MAX_NAME)
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    fully_added: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Check the path.

        Returns:
            Self: The unchanged entry.
        """
        _check_path(self.file)
        return self


class Candidates(BaseModel):
    """Everything the probe found for one PR.

    Attributes:
        instance_id (str): The PR.
        version (int): The probe's VERSION.
        base (str): The base commit the changes were taken from.
        head (str): The PR's last reviewed commit, where the code was read.
        changed_files (tuple[str, ...]): The Python files the PR changed.
        changed (tuple[ChangedFunction, ...]): The functions it changed.
        pieces (tuple[Piece, ...]): Every candidate piece, in packing order.
        stats (dict[str, int]): Counts: files indexed, calls resolved by ast or Jedi or not at
            all, callers and tests found and kept, and so on.
        jedi_seconds (float): Time spent in Jedi.
        seconds (float): Time the probe took.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    version: int
    base: str = Field(pattern=SHA_PATTERN)
    head: str = Field(pattern=SHA_PATTERN)
    changed_files: tuple[str, ...]
    changed: tuple[ChangedFunction, ...]
    pieces: tuple[Piece, ...] = Field(max_length=MAX_PIECES)
    stats: dict[str, int]
    jedi_seconds: float
    seconds: float

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Check the changed files' paths.

        Returns:
            Self: The unchanged candidates.
        """
        for path in self.changed_files:
            _check_path(path)
        return self

    def counts(self) -> dict[str, int]:
        """Count the pieces of each kind.

        Returns:
            dict[str, int]: Every kind, with 0 for kinds with no piece.
        """
        return {kind: sum(p.kind == kind for p in self.pieces) for kind in KIND_ORDER}


class FixDef(BaseModel):
    """A function a fix commit changed.

    Attributes:
        file (str): The file, before the fix.
        qualname (str): The function's qualified name.
        test (bool): Whether the file holds tests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(max_length=MAX_NAME)
    qualname: str = Field(max_length=MAX_NAME)
    test: bool


class FixFunctions(BaseModel):
    """What one fix commit changed.

    Attributes:
        defs (tuple[FixDef, ...]): The functions it changed, as they were just before it.
        outside_lines (int): Changed lines of existing Python files outside every function.
        new_files (tuple[str, ...]): Python files it added (or renamed to).
        error (str): Why the commit couldn't be read; "" when it could.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    defs: tuple[FixDef, ...]
    outside_lines: int = Field(ge=0)
    new_files: tuple[str, ...]
    error: str


def save_candidates(path: Path, candidates: Candidates) -> None:
    """Write a PR's candidates to a JSON file.

    Args:
        path (Path): The file. Its folder is created if needed.
        candidates (Candidates): The candidates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidates.model_dump_json(), encoding="utf-8", newline="\n")


def load_candidates(path: Path) -> Candidates:
    """Read a PR's candidates from their JSON file.

    Args:
        path (Path): The file.

    Returns:
        Candidates: The candidates, checked again.
    """
    return Candidates.model_validate(json.loads(path.read_text(encoding="utf-8")))
