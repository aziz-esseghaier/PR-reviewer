"""Tests for the reviewer model's lookups (context/lookups.py, context/lookup_tool.py).

Checks the lookups as the model writes them (bad paths and names refused with a reason), and
runs the lookup tool on a small repo written to a temporary folder, the way it runs inside a
sandbox: read_file shows numbered lines within the limits and never leaves the repo, and
search_symbol finds definitions (a dotted name narrows them) and uses, cut to size.

How to run (from the project folder):
    pytest tests/context/test_lookups.py
"""

from pathlib import Path

from pr_reviewer.context.lookup_tool import answer, fit, read_file
from pr_reviewer.context.lookups import Lookup, LookupLimits, parse_lookups

FILES = {
    "pkg/__init__.py": "",
    "pkg/cart.py": (
        "class Cart:\n"
        "    def total(self):\n"
        "        return sum(self.items)\n\n\n"
        "def total(items):\n"
        "    return sum(items)\n"
    ),
    "pkg/use.py": "from pkg.cart import Cart\n\n\ndef bill(cart):\n    return cart.total()\n",
    "tests/test_cart.py": (
        "from pkg.cart import total\n\n\ndef test_total():\n    assert total([1]) == 1\n"
    ),
}

LIMITS = {"read_lines": 60, "max_defs": 8, "max_uses": 15, "max_chars": 2000}


def make_repo(root: Path) -> Path:
    """Write the small repo.

    Args:
        root (Path): The folder.

    Returns:
        Path: The same folder, holding the files.
    """
    for name, text in FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_lookups_are_checked_before_running() -> None:
    """Check bad paths, line ranges, names and tools get a reason, and good ones none."""
    assert Lookup(tool="read_file", path="pkg/cart.py", start_line=1, end_line=5).check() == ""
    assert (
        "inside the repository"
        in Lookup(tool="read_file", path="../x.py", start_line=1, end_line=2).check()
    )
    assert (
        "inside the repository"
        in Lookup(tool="read_file", path="/etc/passwd", start_line=1, end_line=2).check()
    )
    assert "start_line" in Lookup(tool="read_file", path="a.py", start_line=5, end_line=2).check()
    assert Lookup(tool="search_symbol", symbol="Dataset.where").check() == ""
    assert "Python name" in Lookup(tool="search_symbol", symbol="os.system('x')").check()
    assert "read_file or search_symbol" in Lookup(tool="run_shell").check()


def test_only_the_first_lookups_are_kept() -> None:
    """Check the limit, and an item that isn't a lookup becomes a refused one."""
    raw = [{"tool": "search_symbol", "symbol": f"f{n}"} for n in range(6)]
    assert [lookup.symbol for lookup in parse_lookups(raw, 4)] == ["f0", "f1", "f2", "f3"]
    (odd,) = parse_lookups(["not an object"], 4)
    assert odd.check() == "the tool must be read_file or search_symbol"


def test_read_file_shows_numbered_lines_within_the_limits(tmp_path: Path) -> None:
    """Check the header, numbering, the line limit and files that can't be read."""
    root = str(make_repo(tmp_path / "repo"))
    shown = read_file(root, "pkg/cart.py", 2, 3, LIMITS)
    assert shown == {
        "ok": True,
        "text": "read_file pkg/cart.py, lines 2-3 of 7:\n2:     def total(self):\n"
        "3:         return sum(self.items)",
    }
    capped = read_file(root, "pkg/cart.py", 1, 100, {**LIMITS, "read_lines": 2})
    assert capped["text"].startswith("read_file pkg/cart.py, lines 1-2 of 7:")
    assert read_file(root, "pkg/cart.py", 50, 60, LIMITS)["ok"] is False
    assert read_file(root, "missing.py", 1, 2, LIMITS)["ok"] is False
    # A path that climbs out of the repo is refused even if the file exists.
    (tmp_path / "secret.txt").write_text("key", encoding="utf-8")
    outside = read_file(root, "../secret.txt", 1, 1, LIMITS)
    assert outside == {
        "ok": False,
        "text": "read_file ../secret.txt: no such file in the repository",
    }


def test_search_symbol_finds_definitions_and_uses(tmp_path: Path) -> None:
    """Check a plain name lists every definition, a dotted one narrows them, and uses show."""
    root = str(make_repo(tmp_path))
    lookups = [
        {"tool": "search_symbol", "symbol": "total", "path": "", "start_line": 0, "end_line": 0},
        {
            "tool": "search_symbol",
            "symbol": "Cart.total",
            "path": "",
            "start_line": 0,
            "end_line": 0,
        },
        {
            "tool": "search_symbol",
            "symbol": "nothing_here",
            "path": "",
            "start_line": 0,
            "end_line": 0,
        },
        {"tool": "read_file", "symbol": "", "path": "pkg/use.py", "start_line": 4, "end_line": 5},
    ]
    results = answer({"root": root, "lookups": lookups, **LIMITS})["results"]
    plain, dotted, missing, read = (item["text"] for item in results)
    assert plain.startswith("search_symbol total: 2 definitions")
    assert "- pkg/cart.py:2-3 method Cart.total: def total(self):" in plain
    assert "- pkg/cart.py:6-7 function total: def total(items):" in plain
    assert "- pkg/use.py:5 in bill: return cart.total()" in plain
    assert "- tests/test_cart.py:5 in test_total: assert total([1]) == 1" in plain
    assert dotted.startswith("search_symbol Cart.total: 1 definitions")
    assert "function total" not in dotted
    assert missing == "search_symbol nothing_here: no definition or use found in the repository"
    assert results[2]["ok"] is False
    assert read.startswith("read_file pkg/use.py, lines 4-5 of 5:")


def test_answers_are_cut_at_whole_lines() -> None:
    """Check a long answer is cut at a line break with a note."""
    text = "\n".join(f"line {n}" for n in range(100))
    cut = fit(text, 40)
    assert len(cut) <= 40 and cut.endswith("[... cut to fit]")
    assert cut.split("\n")[-2].startswith("line ")
    assert fit("short", 40) == "short"
    assert LookupLimits().read_lines == 60
