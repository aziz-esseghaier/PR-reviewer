"""Tests for checking the probe's answer (context/candidates.py) and packing it (context/pack.py).

Checks that an answer from the sandbox with a path out of the repo or lines out of order is
refused, that candidates survive a save and load, that packing follows the kinds' order and
keeps whole pieces only (a smaller piece still fits after a big one is left out), and the
header each piece gets.

How to run (from the project folder):
    pytest tests/context/test_pack.py
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pr_reviewer.context.candidates import load_candidates, save_candidates
from pr_reviewer.context.pack import format_piece, pack_context
from tests.context.helpers import candidates, piece


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd", "a\\b.py", ""])
def test_a_path_out_of_the_repo_is_refused(path: str) -> None:
    """Check the sandbox can't make us read or name a file outside the repo."""
    with pytest.raises(ValidationError, match="inside the repo"):
        piece(file=path)


def test_lines_out_of_order_are_refused() -> None:
    """Check a piece's last line can't come before its first."""
    with pytest.raises(ValidationError, match="out of order"):
        piece(start=9, end=3)


def test_candidates_read_back_unchanged(tmp_path: Path) -> None:
    """Check a save and load gives the same candidates, with LF line endings."""
    found = candidates(piece("changed"), piece("test"))
    path = tmp_path / "ctx" / "pr.json"
    save_candidates(path, found)
    assert load_candidates(path) == found
    assert b"\r\n" not in path.read_bytes()
    assert found.counts()["changed"] == 1


def test_pieces_are_packed_by_kind_whole_only() -> None:
    """Check kinds go in order, and a small piece still fits after a big one is left out."""
    pieces = [
        piece("test", 10, qualname="t"),
        piece("caller", 500, qualname="big"),
        piece("caller", 10, qualname="small"),
        piece("changed", 10, qualname="c"),
    ]
    sizes = {p.qualname: len(format_piece(p)) for p in pieces}
    room = sizes["c"] + sizes["small"] + sizes["t"]
    packed = pack_context(pieces, room)
    assert [p.qualname for p in packed.included] == ["c", "small", "t"]
    assert [p.qualname for p in packed.left_out] == ["big"]
    assert len(packed.text) == room
    assert packed.summary()["left_out"] == ["caller pkg/a.py:1-5 big"]


def test_no_room_packs_nothing() -> None:
    """Check negative room is treated as none."""
    packed = pack_context([piece()], -50)
    assert (packed.text, packed.included, packed.room) == ("", (), 0)


def test_each_piece_says_where_it_is_from_and_why() -> None:
    """Check the header names the file, the lines and the reason."""
    text = format_piece(piece("called", 5, whole=False, qualname="Cart.total", target="avg"))
    assert text.startswith(
        "### pkg/a.py (lines 1-5): `Cart.total`, called by `avg` (signature and docstring)\n"
    )
    assert text.endswith("```python\nxxxxx\n```\n")
