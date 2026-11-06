"""Fitting candidate context into the room a request has left, whole pieces only.

pack_context() goes through the pieces in a fixed order of kinds: the changed source
functions whole, then the definitions they call, their callers, related tests, the test
functions the PR edited, and last the changed files whole. Within a kind it keeps the
probe's order, which takes each changed function's closest caller (or test) before anyone's
second. Each piece goes in whole, with a header
saying where it's from and why, or is left out when it doesn't fit; smaller pieces after it
still get their chance. A piece is never cut, so the model never sees half a function.

A whole changed file may repeat functions already shown; it comes last, and fits only for
small files.

Imported by other modules; the file isn't run directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pr_reviewer.context.candidates import KIND_ORDER, Piece


@dataclass(frozen=True)
class PackedContext:
    """The context that fits.

    Attributes:
        text (str): The pieces, formatted, one after the other. Empty when none fits.
        included (tuple[Piece, ...]): The pieces in the text, in order.
        left_out (tuple[Piece, ...]): The pieces that didn't fit.
        room (int): The characters there were room for.
    """

    text: str
    included: tuple[Piece, ...]
    left_out: tuple[Piece, ...]
    room: int

    def summary(self) -> dict[str, object]:
        """Describe what went in, for the saved review.

        Returns:
            dict[str, object]: Room, characters used, and the included and left-out pieces'
            labels.
        """
        return {
            "room": self.room,
            "chars": len(self.text),
            "included": [piece.label() for piece in self.included],
            "left_out": [piece.label() for piece in self.left_out],
        }


def _why(piece: Piece) -> str:
    """Say in a few words why a piece is in the context.

    Args:
        piece (Piece): The piece.

    Returns:
        str: Such as "`merge_core`, which calls `Dataset.where` (around the call)".
    """
    name = f"`{piece.qualname}`" if piece.qualname else "module-level code"
    target = f"`{piece.target}`"
    part = "" if piece.whole else " (excerpt)"
    if piece.kind == "changed":
        return f"the whole function {name}, which the pull request changes"
    if piece.kind == "changed_test":
        return f"the whole test {name}, which the pull request changes"
    if piece.kind == "called":
        part = "" if piece.whole else " (signature and docstring)"
        return f"{name}, called by {target}{part}"
    if piece.kind == "caller":
        return f"{name}, which calls {target}{part}"
    if piece.kind == "test":
        return f"test code {name}, which uses {target}{part}"
    return "the whole file, as the pull request leaves it"


def format_piece(piece: Piece) -> str:
    """Write one piece with its header, as it goes into the request.

    Args:
        piece (Piece): The piece.

    Returns:
        str: A header line naming the file, lines and reason, then the code in a fence.
    """
    return (
        f"### {piece.file} (lines {piece.start}-{piece.end}): {_why(piece)}\n"
        f"```python\n{piece.text}\n```\n"
    )


def ordered(pieces: Sequence[Piece]) -> list[Piece]:
    """Put pieces in packing order: by kind, keeping the probe's order within a kind.

    Args:
        pieces (Sequence[Piece]): The candidate pieces.

    Returns:
        list[Piece]: The same pieces, in packing order.
    """
    return sorted(pieces, key=lambda piece: KIND_ORDER.index(piece.kind))


def pack_context(pieces: Sequence[Piece], room: int) -> PackedContext:
    """Fit as many whole pieces as the room allows, in packing order.

    Args:
        pieces (Sequence[Piece]): The candidate pieces.
        room (int): Characters available for the formatted pieces.

    Returns:
        PackedContext: The text, and which pieces went in or were left out.
    """
    left = max(0, room)
    blocks: list[str] = []
    included: list[Piece] = []
    left_out: list[Piece] = []
    for piece in ordered(pieces):
        block = format_piece(piece)
        if len(block) <= left:
            blocks.append(block)
            included.append(piece)
            left -= len(block)
        else:
            left_out.append(piece)
    return PackedContext("".join(blocks), tuple(included), tuple(left_out), max(0, room))
