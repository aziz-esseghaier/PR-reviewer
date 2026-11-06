"""Shared test data for the context tests: pieces and candidates.

Imported by the test files in this folder; the file isn't run directly.
"""

from typing import Any

from pr_reviewer.context.candidates import Candidates, Piece

BASE = "a" * 40
HEAD = "b" * 40


def piece(kind: str = "caller", size: int = 50, **changes: Any) -> Piece:
    """Make one piece of context.

    Args:
        kind (str): Its kind.
        size (int): Length of its code text.
        **changes (Any): Fields to change, such as file="pkg/b.py".

    Returns:
        Piece: The piece.
    """
    fields: dict[str, Any] = {
        "kind": kind,
        "file": "pkg/a.py",
        "start": 1,
        "end": 5,
        "qualname": "f",
        "target": "g",
        "rank": 0,
        "whole": True,
        "text": "x" * size,
        **changes,
    }
    return Piece.model_validate(fields)


def candidates(*pieces: Piece, instance_id: str = "psf__requests-1", **changes: Any) -> Candidates:
    """Make a PR's candidates.

    Args:
        *pieces (Piece): The pieces.
        instance_id (str): The PR.
        **changes (Any): Fields to change, such as changed=[...].

    Returns:
        Candidates: The candidates.
    """
    fields: dict[str, Any] = {
        "instance_id": instance_id,
        "version": 1,
        "base": BASE,
        "head": HEAD,
        "changed_files": ["pkg/a.py"],
        "changed": [],
        "pieces": [p.model_dump() for p in pieces],
        "stats": {"calls": 1},
        "jedi_seconds": 0.0,
        "seconds": 1.0,
        **changes,
    }
    return Candidates.model_validate(fields)
