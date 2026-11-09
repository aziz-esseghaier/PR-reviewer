"""Answer the reviewer model's lookups inside a sandbox: read_file and search_symbol.

This file is copied into an indexer sandbox, next to probe.py (whose index it reuses), and run
there with the network off, at the PR's last commit. It reads files and parses them with
Python's `ast`; it never runs the repo's code. Two tools:
- read_file: some lines of one file in the repo, numbered;
- search_symbol: where a function, class or method is defined, and where it's called (or, in
  tests, named), found by name. Uses are matched by name only, so a common method name may
  list calls of other classes' methods; the answer says so.
Every answer is cut to a size limit, so the model's request stays under its token budget.

    python lookup_tool.py <request.json> <answer.json>
        The request has "root" (the working copy), "lookups" (each with "tool", "path",
        "start_line", "end_line" and "symbol") and the limits; the answer has one result per
        lookup: {"ok": ..., "text": ...}.

Run by context/lookups.py inside a sandbox; our own code imports it only in tests.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    from pr_reviewer.context import probe
except ImportError:  # inside the sandbox, probe.py sits next to this file
    import probe  # type: ignore[no-redef, import-not-found]

# Raised whenever what the tools answer changes.
VERSION = 1

# Biggest file read_file opens, in bytes.
MAX_READ_BYTES = 5_000_000

# Longest line kept in an answer, in characters.
LINE_CHARS = 160


def _cut_line(text: str) -> str:
    """Shorten one line of code for an answer.

    Args:
        text (str): The line.

    Returns:
        str: The line, cut with "..." past LINE_CHARS.
    """
    text = text.rstrip()
    return text if len(text) <= LINE_CHARS else text[: LINE_CHARS - 3] + "..."


def fit(text: str, max_chars: int) -> str:
    """Cut an answer to a size, at a whole line, saying so.

    Args:
        text (str): The answer.
        max_chars (int): Most characters.

    Returns:
        str: The answer, or its first lines and a note.
    """
    if len(text) <= max_chars:
        return text
    note = "\n[... cut to fit]"
    kept = text[: max(0, max_chars - len(note))]
    if "\n" in kept:
        kept = kept[: kept.rindex("\n")]
    return kept + note


def inside(root: str, path: str) -> str | None:
    """Find a file inside the repo, refusing paths that lead out of it.

    Args:
        root (str): The repo's folder.
        path (str): The path the model gave, relative to the root.

    Returns:
        str | None: The file's real path, or None when it's outside the repo (symbolic
        links included).
    """
    real_root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(real_root, path))
    return full if full.startswith(real_root + os.sep) else None


def read_file(root: str, path: str, start: int, end: int, limits: dict[str, int]) -> dict[str, Any]:
    """Give some lines of a file, numbered.

    Args:
        root (str): The repo's folder.
        path (str): The file, relative to the root.
        start (int): First line wanted, from 1.
        end (int): Last line wanted.
        limits (dict[str, int]): "read_lines" (most lines shown) and "max_chars".

    Returns:
        dict[str, Any]: {"ok": bool, "text": the lines or why there are none}.
    """
    full = inside(root, path)
    if full is None or not os.path.isfile(full):
        return {"ok": False, "text": f"read_file {path}: no such file in the repository"}
    if os.path.getsize(full) > MAX_READ_BYTES:
        return {"ok": False, "text": f"read_file {path}: the file is too big to read"}
    with open(full, "rb") as file:
        data = file.read()
    if b"\x00" in data[:8192]:
        return {"ok": False, "text": f"read_file {path}: not a text file"}
    lines = data.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return {"ok": True, "text": f"read_file {path}: the file is empty"}
    start = max(1, start)
    if start > len(lines):
        return {"ok": False, "text": f"read_file {path}: the file has only {len(lines)} lines"}
    end = min(len(lines), max(start, end), start + limits["read_lines"] - 1)
    header = f"read_file {path}, lines {start}-{end} of {len(lines)}:"
    body = "\n".join(f"{n}: {_cut_line(lines[n - 1])}" for n in range(start, end + 1))
    return {"ok": True, "text": fit(f"{header}\n{body}", limits["max_chars"])}


def _matches(index: Any, item: Any, name: str) -> bool:
    """Tell whether a definition is the one a dotted name points at.

    Args:
        index (Any): The repo's index (probe.Index).
        item (Any): A definition with the name's last part (probe.Def).
        name (str): The name asked for, such as "Dataset.where" or "pkg.io.read".

    Returns:
        bool: True when the qualname, or module plus qualname, ends with the name.
    """
    if "." not in name:
        return True
    module = index.files[item.file].module
    full = f"{module}.{item.qualname}" if module else item.qualname
    return item.qualname == name or full.endswith("." + name) or full == name


def search_symbols(
    root: str, names: list[str], limits: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Find where some names are defined and used, indexing the repo once.

    Args:
        root (str): The repo's folder.
        names (list[str]): The names asked for, plain or dotted.
        limits (dict[str, int]): "max_defs", "max_uses" and "max_chars".

    Returns:
        dict[str, dict[str, Any]]: By name: {"ok": bool, "text": the answer}.
    """
    index = probe.Index(root)
    wanted = {name.split(".")[-1] for name in names}
    for path in index.python_files():
        tree = index.add_file(path)
        if tree is not None:
            index.collect_uses(path, tree, wanted)
    answers: dict[str, dict[str, Any]] = {}
    for name in names:
        last = name.split(".")[-1]
        every = index.by_name.get(last, [])
        defs = [item for item in every if _matches(index, item, name)]
        note = ""
        if every and not defs:
            defs, note = every, f" (none matches {name} exactly; every `{last}` is listed)"
        uses = sorted(
            index.uses.get(last, []),
            key=lambda use: (probe.is_test_file(use.file), use.file, use.line),
        )
        lines = [
            f"search_symbol {name}: {len(defs)} definitions{note}, {len(uses)} uses "
            f"(calls, and names in tests; matched by name only)"
        ]
        if defs:
            lines.append("Definitions:")
        for item in defs[: limits["max_defs"]]:
            text = index.text(item.file)
            first = text[item.line - 1].strip() if 0 < item.line <= len(text) else ""
            lines.append(
                f"- {item.file}:{item.start}-{item.end} {item.kind} {item.qualname}: "
                f"{_cut_line(first)}"
            )
        if len(defs) > limits["max_defs"]:
            lines.append(f"- [{len(defs) - limits['max_defs']} more definitions]")
        if uses:
            lines.append("Uses:")
        for use in uses[: limits["max_uses"]]:
            text = index.text(use.file)
            code = text[use.line - 1].strip() if 0 < use.line <= len(text) else ""
            info = index.files.get(use.file)
            holder = probe.outer_function(info.defs, use.line) if info is not None else None
            where = f" in {holder.qualname}" if holder is not None else ""
            lines.append(f"- {use.file}:{use.line}{where}: {_cut_line(code)}")
        if len(uses) > limits["max_uses"]:
            lines.append(f"- [{len(uses) - limits['max_uses']} more uses]")
        found = bool(defs or uses)
        if not found:
            lines = [f"search_symbol {name}: no definition or use found in the repository"]
        answers[name] = {"ok": found, "text": fit("\n".join(lines), limits["max_chars"])}
    return answers


def answer(request: dict[str, Any]) -> dict[str, Any]:
    """Answer every lookup of a request, in order.

    Args:
        request (dict[str, Any]): "root", "lookups" and the limits ("read_lines",
            "max_defs", "max_uses", "max_chars").

    Returns:
        dict[str, Any]: {"version": VERSION, "results": one {"ok", "text"} per lookup}.
    """
    root = str(request["root"])
    limits = {key: int(request[key]) for key in ("read_lines", "max_defs", "max_uses", "max_chars")}
    lookups = list(request["lookups"])
    names = [str(item["symbol"]) for item in lookups if item.get("tool") == "search_symbol"]
    symbols = search_symbols(root, sorted(set(names)), limits) if names else {}
    results = []
    for item in lookups:
        if item.get("tool") == "read_file":
            results.append(
                read_file(
                    root,
                    str(item["path"]),
                    int(item["start_line"]),
                    int(item["end_line"]),
                    limits,
                )
            )
        elif item.get("tool") == "search_symbol":
            results.append(symbols[str(item["symbol"])])
        else:
            results.append({"ok": False, "text": f"unknown tool {item.get('tool')!r}"})
    return {"version": VERSION, "results": results}


def main(argv: list[str]) -> int:
    """Read the request, answer it, write the answer.

    Args:
        argv (list[str]): The request file and the answer file.

    Returns:
        int: Exit code, 0 on success.
    """
    if len(argv) != 2:
        print("usage: lookup_tool.py <request.json> <answer.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as file:
        request = json.load(file)
    with open(argv[1], "w", encoding="utf-8") as file:
        json.dump(answer(request), file, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
