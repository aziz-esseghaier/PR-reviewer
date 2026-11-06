"""Find the code around a PR's change, inside a sandbox: index the repo, then pick context.

This file is copied into an indexer sandbox (context/retrieve.py) and run there by the
sandbox's own Python (3.11), with the network off. It reads the repo and parses each file with
Python's `ast`, which never runs the code. Jedi, pinned in the indexer image, is asked only
about calls `ast` can't resolve, such as a name re-exported through a package's __init__.py.
Without Jedi those calls stay unresolved. Apart from Jedi it uses only the standard library.

Two modes, each reading a JSON request and writing a JSON answer:

    python probe.py retrieve <request.json> <answer.json>
        At the PR's last commit (checked out), find the lines the PR changed (git diff from the
        base), then collect candidate pieces of context:
        - changed: each source function or method the PR changed, whole (skipped when the
          PR added all of it, since the diff already shows it);
        - called: the definitions the changed functions call, as signature and docstring (up
          to 10 docstring lines), or whole when short;
        - caller: up to 5 functions that call each changed function;
        - test: up to 3 existing test functions that use each changed function;
        - changed_test: each test function the PR edited, whole;
        - file: each changed Python file, whole.
        The host (context/pack.py) later fits as many as the size budget allows, in that order.

    python probe.py fixdefs <request.json> <answer.json>
        For each given commit (a later fix of a known issue), list the functions it changed,
        as they were just before it. Used only to measure retrieval, never as review input.

The index never leaves the sandbox: a big repo's index (sympy) would be tens of MB, while the
answer holds only the pieces found for one PR.

Run by context/retrieve.py inside a sandbox; our own code imports it only in tests.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Raised whenever what the probe finds changes, so stored answers aren't reused.
VERSION = 1

# Folders never indexed: version control, environments, builds and caches.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        ".eggs",
        "site-packages",
        ".mypy_cache",
        ".pytest_cache",
    }
)

# Bigger files are generated code or data, not worth indexing.
MAX_FILE_BYTES = 1_000_000

# A method called on an object of unknown type, such as obj.copy(), is matched to a changed
# method only when no other definition has its name. With self or cls as the object, up to
# this many definitions may share the name (subclasses often override a method).
AMBIGUOUS_SELF = 3

# Lines shown on each side of a call, in a caller or test too long to show whole.
WINDOW = 5

# Folder names that hold tests (the same rule as reviewer/budget.py is_test_file()).
TEST_FOLDERS = frozenset({"test", "tests", "testing"})

# A full commit ID.
SHA = re.compile(r"^[0-9a-f]{40}$")

# A hunk header of a diff made with --unified=0.
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# Names Python provides, which never point at the repo's code.
BUILTIN_NAMES = frozenset(dir(builtins))

DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

# The stats counter for each way a call ends up: found by ast, found by Jedi, not found in
# the repo although it may be there, or outside the repo (builtins, libraries).
RESOLUTION_STATS = {
    "ast": "resolved_ast",
    "jedi": "resolved_jedi",
    "unresolved": "unresolved",
    "external": "external",
}


class JediTimeout(BaseException):
    """Raised by the alarm when one Jedi question takes too long.

    A BaseException, so Jedi's own `except Exception` blocks don't swallow it.
    """


@dataclass
class Def:
    """One function, method or class in the repo.

    Attributes:
        file (str): Its file, relative to the repo root.
        qualname (str): Its name with its enclosing classes and functions, such as
            "Dataset.where".
        name (str): Its own name.
        kind (str): "function", "method" or "class".
        start (int): Its first line, decorators included.
        line (int): The line of its `def` or `class` keyword.
        end (int): Its last line.
        sig_end (int): The last line of its signature.
        doc_end (int): The last line of its signature and docstring.
        cls (str): The qualname of the class it's a method of, or "".
    """

    file: str
    qualname: str
    name: str
    kind: str
    start: int
    line: int
    end: int
    sig_end: int
    doc_end: int
    cls: str

    @property
    def size(self) -> int:
        """Its number of lines."""
        return self.end - self.start + 1

    def key(self) -> tuple[str, str]:
        """Identify it by file and qualname.

        Returns:
            tuple[str, str]: (file, qualname).
        """
        return (self.file, self.qualname)


@dataclass
class Use:
    """A place that calls or names a function of interest.

    Attributes:
        file (str): The file.
        line (int): The line.
        form (str): "name" for f(...), "attr" for something.f(...).
        receiver (str): For "attr", what comes before the dot, such as "self" or "np"; "" when
            it isn't a plain dotted name.
        name (str): The name used.
        call (bool): Whether it's a call (False: the name is only mentioned, such as passed
            as an argument in a test).
    """

    file: str
    line: int
    form: str
    receiver: str
    name: str
    call: bool


@dataclass
class FileInfo:
    """What the index knows about one file.

    Attributes:
        module (str): Its dotted module name, such as "xarray.core.dataset".
        is_package (bool): Whether it's a package's __init__.py.
        defs (list[Def]): Its functions, methods and classes, in order.
        top (dict[str, Def]): Its top-level functions and classes, by name.
        methods (dict[str, dict[str, Def]]): Methods by class qualname, then by name.
        imports (dict[str, tuple[str, str, str]]): Local names bound by imports: ("module",
            module, "") for `import m as x`, ("from", module, name) for `from m import n as x`.
        stars (list[str]): Modules imported with `from m import *`.
    """

    module: str
    is_package: bool
    defs: list[Def] = field(default_factory=list)
    top: dict[str, Def] = field(default_factory=dict)
    methods: dict[str, dict[str, Def]] = field(default_factory=dict)
    imports: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    stars: list[str] = field(default_factory=list)


def is_test_file(path: str) -> bool:
    """Tell whether a file holds tests, from its path.

    Args:
        path (str): The file's path in the repo, such as "pkg/tests/test_io.py".

    Returns:
        bool: True for files in a tests folder, and for test_*.py, *_test.py and conftest.py.
    """
    parts = path.split("/")
    name = parts[-1]
    return (
        any(part in TEST_FOLDERS for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


# ---------------------------------------------------------------------------------------------
# Diffs
# ---------------------------------------------------------------------------------------------


def _diff_path(line: str) -> str | None:
    """Read the path from a diff's "--- a/..." or "+++ b/..." line.

    Args:
        line (str): The line.

    Returns:
        str | None: The path, or None for /dev/null (a file added or deleted).
    """
    path = line[4:].rstrip("\n")
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path[2:] if path[:2] in ("a/", "b/") else path


def parse_diff(text: str, side: str) -> dict[str, dict[str, set[int]]]:
    """Read which lines a diff made with --unified=0 touches, on one side.

    Args:
        text (str): The diff.
        side (str): "new" for the lines after the change (where they are now), "old" for the
            lines before it.

    Returns:
        dict[str, dict[str, set[int]]]: By file path on that side: "lines", the lines that
        are new or changed (on the old side: removed or changed); "added", the same lines;
        and "between", for a change with no line on this side (a pure insertion or deletion),
        the line after which it happened. Files that don't exist on that side are left out.
    """
    files: dict[str, dict[str, set[int]]] = {}
    empty: dict[str, set[int]]
    old_path: str | None = None
    current: dict[str, set[int]] | None = None
    # Lines of the current hunk still to come. Inside a hunk, a removed line such as
    # "-- note" looks like a file header, so they're skipped by count.
    pending = 0
    for line in text.splitlines():
        if pending:
            if line[:1] in ("-", "+"):
                pending -= 1
            continue
        if line.startswith("--- "):
            old_path = _diff_path(line)
        elif line.startswith("+++ "):
            new_path = _diff_path(line)
            path = new_path if side == "new" else old_path
            empty = {"lines": set(), "added": set(), "between": set()}
            current = None if path is None else files.setdefault(path, empty)
        elif line.startswith("@@"):
            match = HUNK.match(line)
            if match is None:
                continue
            old_start, old_count, new_start, new_count = match.groups()
            olds = 1 if old_count is None else int(old_count)
            news = 1 if new_count is None else int(new_count)
            pending = olds + news
            if current is None:
                continue
            start, count = (int(new_start), news) if side == "new" else (int(old_start), olds)
            if count:
                block = set(range(start, start + count))
                current["lines"] |= block
                current["added"] |= block
            else:
                # Nothing on this side: the change sits between `start` and the next line.
                current["between"].add(start)
    return {path: info for path, info in files.items() if path.endswith(".py")}


def touched_functions(defs: list[Def], change: dict[str, set[int]]) -> list[Def]:
    """Find the outermost functions a file's changes touch, in line order.

    A changed line touches the function holding it. A pure insertion or deletion touches a
    function only when the lines on both sides of it are in that function: code added just
    after a function's last line is new code, not a change to the function.

    Args:
        defs (list[Def]): The file's definitions.
        change (dict[str, set[int]]): parse_diff()'s entry for the file.

    Returns:
        list[Def]: Each touched function once.
    """
    found: dict[str, tuple[int, Def]] = {}
    for line in change["lines"]:
        item = outer_function(defs, line)
        if item is not None and item.qualname not in found:
            found[item.qualname] = (line, item)
    for line in change.get("between", set()):
        item = outer_function(defs, line)
        if item is not None and item is outer_function(defs, line + 1):
            found.setdefault(item.qualname, (line, item))
    return [item for _, item in sorted(found.values(), key=lambda pair: pair[0])]


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


def parse(text: str) -> ast.Module | None:
    """Parse Python source without running it.

    Args:
        text (str): The source.

    Returns:
        ast.Module | None: The tree, or None if it isn't valid Python for this interpreter.
    """
    with warnings.catch_warnings():
        # Old code has escapes such as "\d" in plain strings; Python only warns about them.
        warnings.simplefilter("ignore")
        try:
            return ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return None


def _child_defs(node: ast.AST) -> list[ast.AST]:
    """Find the definitions directly inside a node, also under if, try, with and loops.

    Args:
        node (ast.AST): A module, class or function.

    Returns:
        list[ast.AST]: The function and class definitions, in order.
    """
    found: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, DEF_TYPES):
            found.append(child)
        elif isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
            found.extend(_child_defs(child))
    return found


def _doc_lines(node: Any) -> tuple[int, int]:
    """Find where a definition's signature, and its signature plus docstring, end.

    Args:
        node (Any): A function or class definition.

    Returns:
        tuple[int, int]: The last line of the signature, and of the signature with the
        docstring.
    """
    first = node.body[0]
    sig_end = max(node.lineno, first.lineno - 1)
    is_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    doc_end = first.end_lineno if is_doc and first.end_lineno is not None else sig_end
    return sig_end, doc_end


def walk_defs(tree: ast.Module, file: str) -> list[Def]:
    """List every function, method and class of a file, outer ones first.

    Args:
        tree (ast.Module): The file's tree.
        file (str): The file's path, relative to the repo root.

    Returns:
        list[Def]: The definitions. A method's qualname is "Class.method"; a function inside
        a function is "outer.inner".
    """
    defs: list[Def] = []

    def visit(node: ast.AST, prefix: str, cls: str) -> None:
        for child in _child_defs(node):
            assert isinstance(child, DEF_TYPES)
            qualname = f"{prefix}.{child.name}" if prefix else child.name
            kind = "method" if cls else "function"
            if isinstance(child, ast.ClassDef):
                kind = "class"
            decorators = [d.lineno for d in child.decorator_list]
            sig_end, doc_end = _doc_lines(child)
            defs.append(
                Def(
                    file=file,
                    qualname=qualname,
                    name=child.name,
                    kind=kind,
                    start=min([child.lineno, *decorators]),
                    line=child.lineno,
                    end=child.end_lineno or child.lineno,
                    sig_end=sig_end,
                    doc_end=doc_end,
                    cls=cls,
                )
            )
            visit(child, qualname, qualname if isinstance(child, ast.ClassDef) else "")

    visit(tree, "", "")
    return defs


def outer_function(defs: list[Def], line: int) -> Def | None:
    """Find the outermost function or method that holds a line.

    Args:
        defs (list[Def]): A file's definitions.
        line (int): The line.

    Returns:
        Def | None: The function, or None when the line is outside every function (module or
        class level).
    """
    holding = [d for d in defs if d.kind != "class" and d.start <= line <= d.end]
    return min(holding, key=lambda d: d.start) if holding else None


def inner_def(defs: list[Def], line: int) -> Def | None:
    """Find the innermost function or method that holds a line.

    Args:
        defs (list[Def]): A file's definitions.
        line (int): The line.

    Returns:
        Def | None: The function, or None when the line is outside every function.
    """
    holding = [d for d in defs if d.kind != "class" and d.start <= line <= d.end]
    return max(holding, key=lambda d: d.start) if holding else None


def module_of(root: str, path: str) -> tuple[str, bool, str]:
    """Give a file's module name, the way Python would import it.

    The package starts at the highest folder of an unbroken chain of folders with an
    __init__.py, so "lib/matplotlib/axes/_base.py" is "matplotlib.axes._base".

    Args:
        root (str): The repo's folder.
        path (str): The file's path, relative to the repo root.

    Returns:
        tuple[str, bool, str]: The module name ("" if none), whether it's a package's
        __init__.py, and the folder the package starts in ("" for the repo root).
    """
    folders = path.split("/")[:-1]
    stem = path.split("/")[-1][:-3]
    top = len(folders)
    while top > 0 and os.path.isfile(os.path.join(root, *folders[:top], "__init__.py")):
        top -= 1
    is_package = stem == "__init__"
    parts = folders[top:] + ([] if is_package else [stem])
    return ".".join(parts), is_package, "/".join(folders[:top])


def _relative_base(module: str, is_package: bool, level: int, name: str | None) -> str | None:
    """Turn a relative import's target into an absolute module name.

    Args:
        module (str): The importing file's module.
        is_package (bool): Whether the importing file is a package's __init__.py.
        level (int): How many dots the import starts with (0 for an absolute import).
        name (str | None): The module after the dots, if any.

    Returns:
        str | None: The absolute module name, or None if it goes above the top package.
    """
    if level == 0:
        return name
    package = module.split(".") if is_package else module.split(".")[:-1]
    if level - 1 > len(package):
        return None
    base = package[: len(package) - (level - 1)]
    if name:
        base = [*base, *name.split(".")]
    return ".".join(base) or None


def read_imports(tree: ast.Module, info: FileInfo) -> None:
    """Record the names a file's imports bind.

    Args:
        tree (ast.Module): The file's tree.
        info (FileInfo): The file's entry, filled in place.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    info.imports[alias.asname] = ("module", alias.name, "")
                else:
                    first = alias.name.split(".")[0]
                    info.imports.setdefault(first, ("module", first, ""))
        elif isinstance(node, ast.ImportFrom):
            base = _relative_base(info.module, info.is_package, node.level, node.module)
            if base is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    info.stars.append(base)
                else:
                    info.imports[alias.asname or alias.name] = ("from", base, alias.name)


def dotted(node: ast.AST) -> str:
    """Write an expression like a.b.c as text, or "" when it isn't a plain dotted name.

    Args:
        node (ast.AST): The expression.

    Returns:
        str: Such as "self" or "np.linalg".
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = dotted(node.value)
        return f"{inner}.{node.attr}" if inner else ""
    return ""


# ---------------------------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------------------------


class Index:
    """Every Python file of a repo: definitions, imports, and the uses of chosen names.

    Args:
        root (str): The repo's folder.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self.files: dict[str, FileInfo] = {}
        self.modules: dict[str, str] = {}
        self.by_name: dict[str, list[Def]] = defaultdict(list)
        self.uses: dict[str, list[Use]] = defaultdict(list)
        self.package_roots: set[str] = set()
        self.parse_errors = 0
        self.skipped_big = 0
        self._lines: dict[str, list[str]] = {}
        self._trees: dict[str, ast.Module] = {}

    def python_files(self) -> list[str]:
        """List the repo's Python files, skipping environments, builds and caches.

        Returns:
            list[str]: Paths relative to the root, sorted.
        """
        found: list[str] = []
        for folder, dirs, names in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            relative = os.path.relpath(folder, self.root).replace(os.sep, "/")
            for name in names:
                if name.endswith(".py"):
                    found.append(name if relative == "." else f"{relative}/{name}")
        return sorted(found)

    def text(self, path: str) -> list[str]:
        """Read a file's lines.

        Args:
            path (str): The file, relative to the root.

        Returns:
            list[str]: Its lines, without line endings. Empty when it can't be read.
        """
        if path not in self._lines:
            try:
                with open(os.path.join(self.root, path), encoding="utf-8", errors="replace") as f:
                    self._lines[path] = f.read().splitlines()
            except OSError:
                self._lines[path] = []
        return self._lines[path]

    def add_file(self, path: str, keep_tree: bool = False) -> ast.Module | None:
        """Parse one file and add its definitions and imports.

        Args:
            path (str): The file, relative to the root.
            keep_tree (bool): Keep its tree, for files whose code is looked at again.

        Returns:
            ast.Module | None: Its tree, or None if it's too big or doesn't parse.
        """
        if path in self._trees:
            return self._trees[path]
        full = os.path.join(self.root, path)
        try:
            if os.path.getsize(full) > MAX_FILE_BYTES:
                self.skipped_big += 1
                return None
        except OSError:
            return None
        tree = parse("\n".join(self.text(path)))
        if tree is None:
            self.parse_errors += 1
            return None
        if path not in self.files:
            module, is_package, package_root = module_of(self.root, path)
            info = FileInfo(module=module, is_package=is_package)
            info.defs = walk_defs(tree, path)
            for item in info.defs:
                if "." not in item.qualname:
                    info.top[item.name] = item
                if item.kind == "method":
                    info.methods.setdefault(item.cls, {})[item.name] = item
                self.by_name[item.name].append(item)
            read_imports(tree, info)
            self.files[path] = info
            if module:
                self.modules.setdefault(module, path)
                self.package_roots.add(package_root)
        if keep_tree:
            self._trees[path] = tree
        return tree

    def collect_uses(self, path: str, tree: ast.Module, names: set[str]) -> None:
        """Record where a file calls (or, in test files, names) any of the given names.

        Args:
            path (str): The file.
            tree (ast.Module): Its tree.
            names (set[str]): The names of interest.
        """
        in_test = is_test_file(path)
        called: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in names:
                    self.uses[func.id].append(Use(path, node.lineno, "name", "", func.id, True))
                    called.add(id(func))
                elif isinstance(func, ast.Attribute) and func.attr in names:
                    use = Use(path, node.lineno, "attr", dotted(func.value), func.attr, True)
                    self.uses[func.attr].append(use)
                    called.add(id(func))
        if not in_test:
            return
        for node in ast.walk(tree):
            if id(node) in called:
                continue
            if isinstance(node, ast.Name) and node.id in names:
                self.uses[node.id].append(Use(path, node.lineno, "name", "", node.id, False))
            elif isinstance(node, ast.Attribute) and node.attr in names:
                use = Use(path, node.lineno, "attr", dotted(node.value), node.attr, False)
                self.uses[node.attr].append(use)

    def top_def(self, module: str, name: str) -> Def | None:
        """Find a module's top-level function or class.

        Args:
            module (str): The module name.
            name (str): The function or class name.

        Returns:
            Def | None: It, or None if the module isn't in the repo or doesn't define it.
        """
        path = self.modules.get(module)
        return self.files[path].top.get(name) if path is not None else None

    def method(self, cls: Def, name: str) -> Def | None:
        """Find a method a class defines itself (not one it inherits).

        Args:
            cls (Def): The class.
            name (str): The method name.

        Returns:
            Def | None: The method, or None.
        """
        return self.files[cls.file].methods.get(cls.qualname, {}).get(name)


# ---------------------------------------------------------------------------------------------
# Jedi
# ---------------------------------------------------------------------------------------------


class JediResolver:
    """Ask Jedi where a name is defined, within a time budget.

    Args:
        root (str): The repo's folder.
        package_roots (set[str]): Folders packages start in, relative to the root, added to
            Jedi's search path.
        total_seconds (float): Most time spent in Jedi for one PR.
        call_seconds (float): Most time one question may take.
    """

    def __init__(
        self, root: str, package_roots: set[str], total_seconds: float, call_seconds: float
    ) -> None:
        self.root = os.path.realpath(root)
        self.total_seconds = total_seconds
        self.call_seconds = call_seconds
        self.spent = 0.0
        self.calls = 0
        self.timeouts = 0
        self.errors = 0
        self._scripts: dict[str, Any] = {}
        try:
            self._jedi: Any = importlib.import_module("jedi")
            environment = importlib.import_module("jedi.api.environment")
        except ImportError:
            self._jedi = None
            return
        paths = [os.path.join(self.root, folder) for folder in sorted(package_roots)]
        self._project = self._jedi.Project(self.root, added_sys_path=paths)
        # Inference runs in this process: no extra interpreter, and no C extensions from the
        # repo are ever loaded (load_unsafe_extensions stays off).
        self._environment = environment.InterpreterEnvironment()

    @property
    def available(self) -> bool:
        """Whether Jedi is installed."""
        return self._jedi is not None

    def goto(self, path: str, line: int, column: int) -> list[tuple[str, int]] | None:
        """Ask Jedi where the name at a position is defined, following imports.

        Args:
            path (str): The file, relative to the root.
            line (int): The name's line.
            column (int): The name's column, in characters.

        Returns:
            list[tuple[str, int]] | None: The definitions found, as (file relative to the
            root, line of its def/class keyword); files outside the repo are left out. None
            when Jedi isn't installed, is out of time, or fails.
        """
        left = self.total_seconds - self.spent
        if self._jedi is None or left <= 0:
            return None
        if sys.platform == "win32":
            # The time limit needs SIGALRM, which only Unix has; the probe runs on Linux.
            return None
        started = time.monotonic()
        self.calls += 1

        def alarm(signum: int, frame: Any) -> None:
            raise JediTimeout

        previous = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, min(self.call_seconds, left))
        try:
            script = self._scripts.get(path)
            if script is None:
                script = self._jedi.Script(
                    path=os.path.join(self.root, path),
                    project=self._project,
                    environment=self._environment,
                )
                self._scripts[path] = script
            names = script.goto(line, column, follow_imports=True)
        except JediTimeout:
            self.timeouts += 1
            return None
        except Exception:
            self.errors += 1
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            self.spent += time.monotonic() - started
        found: list[tuple[str, int]] = []
        for name in names:
            module_path = getattr(name, "module_path", None)
            if module_path is None or name.line is None:
                continue
            full = os.path.realpath(str(module_path))
            if full.startswith(self.root + os.sep):
                found.append((os.path.relpath(full, self.root).replace(os.sep, "/"), name.line))
        return found


# ---------------------------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------------------------


@dataclass
class Settings:
    """How much to collect.

    Attributes:
        max_piece_chars (int): Pieces bigger than this are skipped: they could never fit.
        callers (int): Most callers kept per changed function.
        tests (int): Most tests kept per changed function.
        short_lines (int): A called function, caller or test up to this many lines is shown
            whole; a longer one as signature and docstring, or around the call.
        doc_lines (int): Most docstring lines shown with a signature; longer docstrings are
            cut, so one long docstring can't use the room callers and tests need.
        jedi_seconds (float): Most time spent in Jedi per PR.
        jedi_call_seconds (float): Most time one Jedi question may take.
    """

    max_piece_chars: int = 12_000
    callers: int = 5
    tests: int = 3
    short_lines: int = 30
    doc_lines: int = 10
    jedi_seconds: float = 120.0
    jedi_call_seconds: float = 10.0


@dataclass
class Changed:
    """A function or method the PR changed.

    Attributes:
        item (Def): The function.
        fully_added (bool): Whether the PR added all of it.
        target (str): The name its callers use: its own, or its class's for __init__. ""
            when callers can't be found by name (other dunder methods, nested functions).
    """

    item: Def
    fully_added: bool
    target: str


class Retriever:
    """Collect the candidate context pieces for one PR.

    Args:
        index (Index): The repo's index.
        settings (Settings): How much to collect.
        jedi (JediResolver | None): Jedi, for calls `ast` can't resolve.
    """

    def __init__(self, index: Index, settings: Settings, jedi: JediResolver | None) -> None:
        self.index = index
        self.settings = settings
        self.jedi = jedi
        self.stats: dict[str, int] = defaultdict(int)

    # -- changed functions ------------------------------------------------------------------

    def changed_functions(self, changes: dict[str, dict[str, set[int]]]) -> list[Changed]:
        """Find the functions and methods holding the PR's changed lines.

        Args:
            changes (dict[str, dict[str, set[int]]]): parse_diff()'s new side.

        Returns:
            list[Changed]: The changed functions, by file and line.
        """
        found: list[Changed] = []
        # Source files first: their functions matter more than changed tests.
        for path in sorted(changes, key=lambda path: (is_test_file(path), path)):
            info = self.index.files.get(path)
            if info is None:
                continue
            added = changes[path]["added"]
            for item in touched_functions(info.defs, changes[path]):
                fully = all(n in added for n in range(item.start, item.end + 1))
                found.append(Changed(item, fully, self._target_name(item)))
        return found

    def _target_name(self, item: Def) -> str:
        """Give the name callers of a function use.

        Args:
            item (Def): The changed function.

        Returns:
            str: Its name; its class's name for __init__; "" for other dunder methods,
            functions nested in functions (which can't be found by name) and tests.
        """
        if is_test_file(item.file):
            return ""
        if item.kind == "method" and item.name == "__init__":
            return item.cls.split(".")[-1]
        if item.name.startswith("__") and item.name.endswith("__"):
            self.stats["dunder_skipped"] += 1
            return ""
        nested = item.qualname.count(".") > (1 if item.kind == "method" else 0)
        return "" if nested else item.name

    # -- definitions the changed functions call ---------------------------------------------

    def _class_at(self, module: str, name: str) -> Def | None:
        """Find a class by module and name.

        Args:
            module (str): The module.
            name (str): The class name.

        Returns:
            Def | None: The class, or None.
        """
        item = self.index.top_def(module, name)
        return item if item is not None and item.kind == "class" else None

    def resolve(self, path: str, cls: str, func: ast.AST) -> tuple[str, Def | None]:
        """Find the definition a call goes to, with `ast` alone.

        Args:
            path (str): The calling file.
            cls (str): The qualname of the class the call is in ("" outside methods).
            func (ast.AST): The called expression.

        Returns:
            tuple[str, Def | None]: ("ast", the definition) when found; ("jedi", None) when
            it may be in the repo but needs Jedi (re-exports, inherited methods, unknown
            objects); ("external", None) when it's outside the repo (builtins, libraries).
        """
        index = self.index
        info = index.files[path]
        if isinstance(func, ast.Name):
            name = func.id
            if name in info.top:
                return "ast", info.top[name]
            bound = info.imports.get(name)
            if bound is not None:
                kind, module, original = bound
                if kind == "module" or module not in index.modules:
                    in_repo = kind == "from" and f"{module}.{original}" in index.modules
                    return ("jedi", None) if in_repo else ("external", None)
                found = index.top_def(module, original)
                return ("ast", found) if found is not None else ("jedi", None)
            for module in info.stars:
                found = index.top_def(module, name)
                if found is not None:
                    return "ast", found
            if name in BUILTIN_NAMES or name not in index.by_name:
                return "external", None
            return "jedi", None
        if not isinstance(func, ast.Attribute):
            return "external", None
        attr, receiver = func.attr, dotted(func.value)
        if receiver in ("self", "cls") and cls:
            own = info.methods.get(cls, {}).get(attr)
            if own is not None:
                return "ast", own
            return ("jedi", None) if attr in index.by_name else ("external", None)
        if receiver and "." not in receiver:
            bound = info.imports.get(receiver)
            if bound is not None:
                kind, module, original = bound
                target = module if kind == "module" else f"{module}.{original}"
                if target in index.modules:
                    found = index.top_def(target, attr)
                    return ("ast", found) if found is not None else ("jedi", None)
                if kind == "from" and module in index.modules:
                    owner = self._class_at(module, original)
                    if owner is not None and index.method(owner, attr) is not None:
                        return "ast", index.method(owner, attr)
                    return ("jedi", None) if attr in index.by_name else ("external", None)
                return "external", None
            owner = info.top.get(receiver)
            if owner is not None and owner.kind == "class":
                found = index.method(owner, attr)
                if found is not None:
                    return "ast", found
        if attr not in index.by_name:
            return "external", None
        return "jedi", None

    def _jedi_resolve(self, path: str, func: ast.AST) -> Def | None:
        """Ask Jedi for a call's definition.

        Args:
            path (str): The calling file.
            func (ast.AST): The called expression (a name or an attribute).

        Returns:
            Def | None: The definition in the repo, or None.
        """
        if self.jedi is None or not self.jedi.available:
            return None
        if isinstance(func, ast.Attribute):
            line = func.end_lineno or func.lineno
            byte_column = (func.end_col_offset or 0) - len(func.attr)
        elif isinstance(func, ast.Name):
            line, byte_column = func.lineno, func.col_offset
        else:
            return None
        lines = self.index.text(path)
        if not 0 < line <= len(lines):
            return None
        source = lines[line - 1].encode("utf-8")
        column = len(source[: max(0, byte_column)].decode("utf-8", errors="replace"))
        found = self.jedi.goto(path, line, column)
        if not found:
            return None
        for file, def_line in found:
            if file not in self.index.files:
                self.index.add_file(file)
            info = self.index.files.get(file)
            if info is None:
                continue
            for item in info.defs:
                if item.line == def_line:
                    return item
        return None

    def called_definitions(self, changed: list[Changed]) -> list[tuple[Def, str]]:
        """Find the definitions the changed functions call.

        Args:
            changed (list[Changed]): The changed functions.

        Returns:
            list[tuple[Def, str]]: Each definition, once, with the changed function that first
            calls it, in order of first call. Changed functions, and functions nested in them,
            are left out.
        """
        changed_keys = {c.item.key() for c in changed}
        found: dict[tuple[str, str], tuple[Def, str]] = {}

        def inside_changed(target: Def) -> bool:
            return any(
                c.item.file == target.file and c.item.start <= target.start <= c.item.end
                for c in changed
            )

        for entry in changed:
            item = entry.item
            tree = self.index.add_file(item.file, keep_tree=True)
            if tree is None:
                continue
            node = _find_def_node(tree, item)
            if node is None:
                continue
            cls = item.cls
            calls = sorted(
                (n for n in ast.walk(node) if isinstance(n, ast.Call)),
                key=lambda n: (n.lineno, n.col_offset),
            )
            for call in calls:
                self.stats["calls"] += 1
                how, target = self.resolve(item.file, cls, call.func)
                if how == "jedi":
                    target = self._jedi_resolve(item.file, call.func)
                    how = "jedi" if target is not None else "unresolved"
                self.stats[RESOLUTION_STATS[how]] += 1
                if target is None or target.key() in changed_keys or target.key() in found:
                    continue
                if inside_changed(target):
                    # A function nested in a changed one is already shown with it.
                    continue
                found[target.key()] = (target, item.qualname)
        return list(found.values())

    # -- callers and tests ------------------------------------------------------------------

    def _accepts(self, use: Use, target: Changed) -> bool:
        """Tell whether a use of a name refers to a changed function, with `ast` alone.

        Args:
            use (Use): Where the name is used.
            target (Changed): The changed function.

        Returns:
            bool: True when the use refers to it. Uses that may or may not (a method name
            many classes share, on an object of unknown type) are refused and counted.
        """
        item = target.item
        index = self.index
        own = index.files[item.file]
        info = index.files.get(use.file)
        if info is None:
            return False
        top_level = item.kind == "function" or item.name == "__init__"
        if use.form == "name":
            if not top_level:
                return False
            if use.file == item.file:
                return True
            bound = info.imports.get(use.name)
            if bound is not None:
                kind, module, original = bound
                return (
                    kind == "from"
                    and original == target.target
                    and (module == own.module or own.module.startswith(module + "."))
                )
            return any(
                module == own.module or own.module.startswith(module + ".") for module in info.stars
            )
        receiver = use.receiver
        if top_level:
            # module.function(...), with the module imported under some name.
            if not receiver:
                return False
            if receiver == own.module:
                return True
            bound = info.imports.get(receiver) if "." not in receiver else None
            if bound is None:
                return False
            kind, module, original = bound
            return (module if kind == "module" else f"{module}.{original}") == own.module
        # A changed method, called as something.method(...).
        sharing = len(index.by_name.get(item.name, []))
        class_name = item.cls.split(".")[-1]
        if receiver == class_name:
            return True
        if receiver in ("self", "cls"):
            holder = inner_def(info.defs, use.line)
            if holder is not None and holder.cls == item.cls and use.file == item.file:
                return True
            if sharing <= AMBIGUOUS_SELF:
                return True
        elif sharing == 1:
            return True
        self.stats["ambiguous_skipped"] += 1
        return False

    def users(
        self, changed: list[Changed], *, tests: bool
    ) -> list[tuple[Use, Def | None, Changed, int]]:
        """Find the callers (or tests) of each changed function, closest first.

        Args:
            changed (list[Changed]): The changed functions.
            tests (bool): True for uses in test files, False for calls elsewhere.

        Returns:
            list[tuple[Use, Def | None, Changed, int]]: Each kept use, with the function it's
            in (None at module level), the changed function it uses, and its rank among that
            function's uses. Up to the settings' callers (or tests) per changed function,
            ordered by rank, then by changed function, so every changed function gets its
            closest ones before any gets more.
        """
        limit = self.settings.tests if tests else self.settings.callers
        changed_keys = {c.item.key() for c in changed}
        kept: list[tuple[int, int, Use, Def | None, Changed]] = []
        for number, target in enumerate(changed):
            if not target.target:
                continue
            item = target.item
            own_module = self.index.files[item.file].module
            candidates: list[tuple[tuple[int, str, int], Use, Def | None]] = []
            holders: set[tuple[str, str]] = set()
            for use in self.index.uses.get(target.target, []):
                if is_test_file(use.file) != tests or (not tests and not use.call):
                    continue
                info = self.index.files.get(use.file)
                if info is None:
                    continue
                holder = outer_function(info.defs, use.line)
                if holder is not None and holder.key() in changed_keys:
                    continue
                if not self._accepts(use, target):
                    continue
                self.stats["tests_found" if tests else "callers_found"] += 1
                where = holder.key() if holder is not None else (use.file, f"line {use.line}")
                if where in holders:
                    continue
                holders.add(where)
                rank = _closeness(use.file, item.file, own_module, tests)
                candidates.append(((rank, use.file, use.line), use, holder))
            candidates.sort(key=lambda entry: entry[0])
            for position, (_, use, holder) in enumerate(candidates[:limit]):
                kept.append((position, number, use, holder, target))
        kept.sort(key=lambda entry: (entry[0], entry[1]))
        self.stats["tests_kept" if tests else "callers_kept"] += len(kept)
        return [(use, holder, target, position) for position, _, use, holder, target in kept]

    # -- pieces -------------------------------------------------------------------------------

    def _lines(self, path: str, start: int, end: int) -> str:
        """Give some of a file's lines.

        Args:
            path (str): The file.
            start (int): First line (1-based).
            end (int): Last line.

        Returns:
            str: The lines, joined.
        """
        return "\n".join(self.index.text(path)[start - 1 : end])

    def summary_text(self, item: Def) -> tuple[str, bool]:
        """Show a definition as signature and docstring, or whole when it's short.

        A class shows its header, docstring and __init__ signature. A docstring longer than
        the settings' doc_lines is cut after them.

        Args:
            item (Def): The definition.

        Returns:
            tuple[str, bool]: The text, and whether it's the whole definition.
        """
        if item.size <= self.settings.short_lines:
            return self._lines(item.file, item.start, item.end), True
        lines = self.index.text(item.file)
        first = lines[item.line - 1] if 0 < item.line <= len(lines) else ""
        indent = first[: len(first) - len(first.lstrip())] + "    "
        doc_end = min(item.doc_end, item.sig_end + self.settings.doc_lines)
        text = self._lines(item.file, item.start, doc_end)
        if doc_end < item.doc_end:
            text += f"\n{indent}[... rest of the docstring]"
        if item.kind == "class":
            init = self.index.method(item, "__init__")
            if init is not None:
                text += f"\n{indent}...\n" + self._lines(item.file, init.start, init.sig_end)
        return f"{text}\n{indent}...", False

    def around_text(self, path: str, holder: Def | None, line: int) -> tuple[int, int, str, bool]:
        """Show the function a use is in: whole when short, otherwise around the use.

        Args:
            path (str): The file.
            holder (Def | None): The function holding the use, or None at module level.
            line (int): The use's line.

        Returns:
            tuple[int, int, str, bool]: First and last line shown, the text, and whether it's
            the whole function.
        """
        if holder is None:
            start = max(1, line - WINDOW)
            end = min(len(self.index.text(path)), line + WINDOW)
            return start, end, self._lines(path, start, end), False
        if holder.size <= self.settings.short_lines:
            return holder.start, holder.end, self._lines(path, holder.start, holder.end), True
        # The signature, then the lines around the use (a use in the signature itself, such
        # as a default value, shows the lines just after it).
        start = max(holder.sig_end + 1, line - WINDOW)
        end = min(holder.end, max(line, start) + WINDOW)
        text = self._lines(path, holder.start, holder.sig_end)
        if start > holder.sig_end + 1:
            text += "\n    ..."
        text += "\n" + self._lines(path, start, end)
        if end < holder.end:
            text += "\n    ..."
        return holder.start, holder.end, text, False

    def pieces(
        self, changes: dict[str, dict[str, set[int]]]
    ) -> tuple[list[dict[str, Any]], list[Changed]]:
        """Collect every candidate piece, in packing order.

        Args:
            changes (dict[str, dict[str, set[int]]]): parse_diff()'s new side.

        Returns:
            tuple[list[dict[str, Any]], list[Changed]]: The pieces (kind, file, start, end,
            qualname, target, rank, whole, text), and the changed functions.
        """
        changed = self.changed_functions(changes)
        limit = self.settings.max_piece_chars
        pieces: list[dict[str, Any]] = []

        def add(
            kind: str,
            path: str,
            start: int,
            end: int,
            qualname: str,
            target: str,
            rank: int,
            whole: bool,
            text: str,
        ) -> None:
            if len(text) > limit:
                self.stats["too_big"] += 1
                return
            pieces.append(
                {
                    "kind": kind,
                    "file": path,
                    "start": start,
                    "end": end,
                    "qualname": qualname,
                    "target": target,
                    "rank": rank,
                    "whole": whole,
                    "text": text,
                }
            )

        def add_changed(kind: str, entries: list[Changed]) -> None:
            for rank, entry in enumerate(entries):
                item = entry.item
                if entry.fully_added:
                    self.stats["changed_fully_added"] += 1
                    continue
                text = self._lines(item.file, item.start, item.end)
                add(
                    kind,
                    item.file,
                    item.start,
                    item.end,
                    item.qualname,
                    item.qualname,
                    rank,
                    True,
                    text,
                )

        # Changed source functions go first; tests the PR edited go after the related tests,
        # since the diff already shows their changed lines and bugs live in the source.
        add_changed("changed", [e for e in changed if not is_test_file(e.item.file)])
        for rank, (item, caller) in enumerate(self.called_definitions(changed)):
            text, whole = self.summary_text(item)
            add("called", item.file, item.start, item.end, item.qualname, caller, rank, whole, text)
        for tests in (False, True):
            for use, holder, target, rank in self.users(changed, tests=tests):
                start, end, text, whole = self.around_text(use.file, holder, use.line)
                qualname = holder.qualname if holder is not None else ""
                add(
                    "test" if tests else "caller",
                    use.file,
                    start,
                    end,
                    qualname,
                    target.item.qualname,
                    rank,
                    whole,
                    text,
                )
        add_changed("changed_test", [e for e in changed if is_test_file(e.item.file)])
        files = sorted(changes, key=lambda path: (is_test_file(path), path))
        for rank, path in enumerate(files):
            lines = self.index.text(path)
            if path in self.index.files and lines:
                add("file", path, 1, len(lines), "", "", rank, True, "\n".join(lines))
        return _dedupe(pieces), changed


def _closeness(path: str, target_file: str, target_module: str, tests: bool) -> int:
    """Rank a use by how close it is to the changed function.

    Args:
        path (str): The using file.
        target_file (str): The changed function's file.
        target_module (str): The changed function's module.
        tests (bool): Whether the use is in a test file.

    Returns:
        int: Lower is closer. Callers: same file 0, same folder 1, same top folder 2, else 3.
        Tests: the module's own test file (test_<module>.py) 0, same top folder 1, else 2.
    """
    if tests:
        stem = target_module.split(".")[-1]
        name = path.split("/")[-1]
        if name in (f"test_{stem}.py", f"{stem}_test.py"):
            return 0
        return 1 if path.split("/")[0] == target_file.split("/")[0] else 2
    if path == target_file:
        return 0
    if posixpath.dirname(path) == posixpath.dirname(target_file):
        return 1
    return 2 if path.split("/")[0] == target_file.split("/")[0] else 3


def _dedupe(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop pieces whose lines an earlier piece of the same file already covers.

    A full file is never dropped this way; pack.py handles it.

    Args:
        pieces (list[dict[str, Any]]): The pieces, in packing order.

    Returns:
        list[dict[str, Any]]: The pieces without repeats.
    """
    kept: list[dict[str, Any]] = []
    for piece in pieces:
        covered = piece["kind"] != "file" and any(
            other["kind"] != "file"
            and other["file"] == piece["file"]
            and other["start"] <= piece["start"]
            and piece["end"] <= other["end"]
            and (other["whole"] or not piece["whole"])
            for other in kept
        )
        if not covered:
            kept.append(piece)
    return kept


def _find_def_node(tree: ast.Module, item: Def) -> ast.AST | None:
    """Find a definition's node in its file's tree.

    Args:
        tree (ast.Module): The file's tree.
        item (Def): The definition.

    Returns:
        ast.AST | None: The node, or None.
    """
    for node in ast.walk(tree):
        if isinstance(node, DEF_TYPES) and node.lineno == item.line and node.name == item.name:
            return node
    return None


# ---------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------


def _git(root: str, *args: str) -> str:
    """Run a git command in the repo and give its output.

    Args:
        root (str): The repo's folder.
        *args (str): The git arguments.

    Returns:
        str: What it printed.

    Raises:
        RuntimeError: If git fails.
    """
    result = subprocess.run(
        ["git", "-C", root, "-c", "core.quotePath=false", *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {message}")
    return result.stdout.decode("utf-8", errors="replace")


def _check_sha(value: Any) -> str:
    """Check a commit ID from a request.

    Args:
        value (Any): The value.

    Returns:
        str: The commit ID.

    Raises:
        ValueError: If it isn't a full commit ID.
    """
    if not isinstance(value, str) or not SHA.match(value):
        raise ValueError(f"not a full commit ID: {value!r}")
    return value


def collect(
    root: str,
    changes: dict[str, dict[str, set[int]]],
    settings: Settings,
    use_jedi: bool = True,
) -> tuple[list[dict[str, Any]], list[Changed], dict[str, int], JediResolver]:
    """Index a repo and collect the candidate pieces for some changed lines.

    The changed files are parsed first, to know which names to look for; then every Python
    file of the repo is indexed, recording only uses of those names.

    Args:
        root (str): The repo's folder, at the PR's last commit.
        changes (dict[str, dict[str, set[int]]]): parse_diff()'s new side.
        settings (Settings): How much to collect.
        use_jedi (bool): Whether to ask Jedi about calls `ast` can't resolve.

    Returns:
        tuple[list[dict[str, Any]], list[Changed], dict[str, int], JediResolver]: The pieces,
        the changed functions, the counts, and the Jedi resolver (for its time and counts).
    """
    index = Index(root)
    for path in sorted(changes):
        if os.path.isfile(os.path.join(root, path)):
            index.add_file(path, keep_tree=True)
    targets: set[str] = set()
    for entry in Retriever(index, settings, None).changed_functions(changes):
        if entry.target:
            targets.add(entry.target)
    for path in index.python_files():
        tree = index.add_file(path, keep_tree=path in changes)
        if tree is not None and targets:
            index.collect_uses(path, tree, targets)
    jedi = JediResolver(
        root,
        index.package_roots,
        settings.jedi_seconds if use_jedi else 0.0,
        settings.jedi_call_seconds,
    )
    retriever = Retriever(index, settings, jedi)
    pieces, changed = retriever.pieces(changes)
    stats = dict(retriever.stats)
    stats.update(
        files_indexed=len(index.files),
        parse_errors=index.parse_errors,
        skipped_big=index.skipped_big,
        changed_files=len(changes),
        changed_functions=len(changed),
        jedi_available=int(jedi.available and use_jedi),
        jedi_calls=jedi.calls,
        jedi_timeouts=jedi.timeouts,
        jedi_errors=jedi.errors,
    )
    return pieces, changed, stats, jedi


def retrieve(request: dict[str, Any]) -> dict[str, Any]:
    """Collect a PR's candidate context, with the repo checked out at its last commit.

    Args:
        request (dict[str, Any]): "root" (the working copy), "base" and "head" (commit IDs),
            and optionally Settings' fields.

    Returns:
        dict[str, Any]: The answer: version, commits, changed files and functions, pieces
        and stats.
    """
    started = time.monotonic()
    root = str(request["root"])
    base, head = _check_sha(request["base"]), _check_sha(request["head"])
    known = Settings.__dataclass_fields__
    settings = Settings(**{k: v for k, v in request.items() if k in known})
    diff = _git(
        root, "diff", "--unified=0", "--no-color", "--no-ext-diff", "-M", base, head, "--", "*.py"
    )
    changes = parse_diff(diff, "new")
    pieces, changed, stats, jedi = collect(root, changes, settings)
    return {
        "version": VERSION,
        "base": base,
        "head": head,
        "changed_files": sorted(changes),
        "changed": [
            {
                "file": c.item.file,
                "qualname": c.item.qualname,
                "start": c.item.start,
                "end": c.item.end,
                "fully_added": c.fully_added,
            }
            for c in changed
        ],
        "pieces": pieces,
        "stats": stats,
        "jedi_seconds": round(jedi.spent, 1),
        "seconds": round(time.monotonic() - started, 1),
    }


def fix_functions(root: str, commit: str) -> dict[str, Any]:
    """List the functions one commit changed, as they were just before it.

    Args:
        root (str): The repo's folder.
        commit (str): The commit's full ID.

    Returns:
        dict[str, Any]: "defs" (file, qualname, test) changed in files that existed before;
        "outside_lines", changed lines of those files outside every function; "new_files",
        Python files the commit added; "error" when git failed.
    """
    try:
        parent = _git(root, "rev-parse", "--verify", "--quiet", f"{commit}^1").strip()
        diff = _git(
            root,
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "-M",
            parent,
            commit,
            "--",
            "*.py",
        )
    except RuntimeError as exc:
        return {"defs": [], "outside_lines": 0, "new_files": [], "error": str(exc)}
    old = parse_diff(diff, "old")
    new = parse_diff(diff, "new")
    defs: list[dict[str, Any]] = []
    outside = 0
    for path in sorted(old):
        try:
            text = _git(root, "show", f"{parent}:{path}")
        except RuntimeError:
            continue
        tree = parse(text)
        if tree is None:
            continue
        items = walk_defs(tree, path)
        outside += sum(outer_function(items, line) is None for line in old[path]["lines"])
        for item in touched_functions(items, old[path]):
            defs.append({"file": path, "qualname": item.qualname, "test": is_test_file(path)})
    # Files only on the new side: added, or renamed (their old name is in `old`).
    new_files = sorted(set(new) - set(old))
    return {"defs": defs, "outside_lines": outside, "new_files": new_files, "error": ""}


def fixdefs(request: dict[str, Any]) -> dict[str, Any]:
    """List the functions each fix commit changed.

    Args:
        request (dict[str, Any]): "root" (the working copy) and "commits" (full IDs).

    Returns:
        dict[str, Any]: The answer: version, and fix_functions() for each commit.
    """
    root = str(request["root"])
    commits = [_check_sha(commit) for commit in request["commits"]]
    return {"version": VERSION, "fixes": {c: fix_functions(root, c) for c in commits}}


def main(argv: list[str]) -> int:
    """Run one mode: read the request, write the answer.

    Args:
        argv (list[str]): The mode, the request file and the answer file.

    Returns:
        int: Exit code, 0 on success.
    """
    if len(argv) != 3 or argv[0] not in ("retrieve", "fixdefs"):
        print("usage: probe.py retrieve|fixdefs <request.json> <answer.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as file:
        request = json.load(file)
    answer = retrieve(request) if argv[0] == "retrieve" else fixdefs(request)
    with open(argv[2], "w", encoding="utf-8") as file:
        json.dump(answer, file, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
