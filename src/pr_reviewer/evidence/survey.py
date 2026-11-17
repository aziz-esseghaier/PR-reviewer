"""The survey of one PR: the code each candidate's test writer and verifier get to read.

For one PR, in one sandbox started from its prepared image, network off:
1. the working copy is put at the PR's last commit, and its base is found;
2. git lists the repo's Python files and gives the PR's diff twice: with no context lines
   (which lines changed, for the matcher and the lines the runner watches) and with three
   around each change (the hunks the model reads);
3. our code picks, for each candidate, where its test goes (next to the closest existing
   test) and reads the claimed file, that test file and the conftest.py files above it.
Files are read as text and parsed with Python's ast on this side, so no repo code runs.

The pure functions below then build each candidate's Materials:
- the function holding the claimed lines, with line numbers (or the lines around them);
- the diff hunks nearest the claimed lines;
- the test's path, an example test (the closest test file's imports and one of its tests) and
  the fixtures its folder's conftest.py files offer.
Our code chooses all of this, so the model gets useful context without spending a call.

Imported by other modules; the file isn't run directly.
"""

import ast
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.probe import parse_diff
from pr_reviewer.evidence.inputs import Candidate
from pr_reviewer.sandbox.checkout import base_of, checkout
from pr_reviewer.sandbox.runner import Sandbox, SandboxError
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.setup_agent.selection import is_test_file

# Where the survey keeps git's output inside the sandbox (outside the repo).
WORK = "/tmp/pr_reviewer_survey"

# Biggest file the survey reads, in bytes. Bigger ones are left out.
MAX_FILE_BYTES = 2_000_000

# Most characters of each part of a candidate's materials. The request builders cut them
# again to fit the room they have.
CODE_CHARS = 6_000
HUNK_CHARS = 4_000
EXAMPLE_CHARS = 3_000
IMPORT_CHARS = 1_200
MAX_FIXTURES = 30

# Lines shown around the claimed lines when they aren't inside a function.
AROUND_LINES = 25

# Seconds a git command may take.
GIT_SECONDS = 600

# The generated test's file name, next to the closest existing test.
TEST_NAME = "test_pr_reviewer_{finding_id}.py"

# A Python name, and the lines that define names: a function with its arguments, a class, an
# assignment (also to self.x), and a `from ... import` that brings names in.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\((.*)")
_CLASS = re.compile(r"^class\s+(\w+)")
_ASSIGN = re.compile(r"^(?:self\.|cls\.)?(\w+)\s*(?::[^=]+)?=(?!=)")
_FROM_IMPORT = re.compile(r"^from\s+[\w.]+\s+import\s+(.+)$")


class SurveyError(Exception):
    """Raised when a PR's survey can't be made, such as when git fails in the sandbox."""


class Materials(BaseModel):
    """What one candidate's test writer and verifier get to read.

    Attributes:
        finding_id (str): The candidate.
        function (str): The qualified name of the function holding the claimed lines, such
            as "Cart.total"; "" when they're outside any function or the file wasn't read.
        code (str): That function's lines (or the lines around the claim), numbered.
        hunks (str): The PR's diff hunks nearest the claimed lines.
        test_path (str | None): Where a generated test goes, relative to the repo root; None
            when the repo has no test file to put it next to.
        example (str): The closest test file's path, imports and one test, for the style.
        fixtures (tuple[str, ...]): The fixtures the test's folder offers, one per line, such
            as "simple_dataset(request): A small dataset with two variables."
    """

    model_config = ConfigDict(frozen=True)

    finding_id: str
    function: str = ""
    code: str = ""
    hunks: str = ""
    test_path: str | None = None
    example: str = ""
    fixtures: tuple[str, ...] = ()


class PrSurvey(BaseModel):
    """What the survey found for one PR.

    Attributes:
        instance_id (str): The PR.
        base (str): The base commit.
        head (str): The PR's last reviewed commit.
        changed (dict[str, list[int]]): The lines the PR added or changed in Python files, at
            its last commit, by file.
        added_names (tuple[str, ...]): The names the PR's added lines define (functions,
            arguments, classes, assigned and imported names) and the module names of files it
            added, sorted; they tell code the PR added apart.
        materials (dict[str, Materials]): Each candidate's materials, by finding ID.
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str
    base: str
    head: str
    changed: dict[str, list[int]]
    added_names: tuple[str, ...]
    materials: dict[str, Materials]


@dataclass(frozen=True)
class Span:
    """A function or method in a file.

    Attributes:
        qualname (str): Its qualified name, such as "Cart.total".
        start (int): Its first line, decorators included.
        end (int): Its last line.
    """

    qualname: str
    start: int
    end: int


def functions(source: str) -> list[Span]:
    """List a file's functions and methods, nested ones included.

    Args:
        source (str): The file's text.

    Returns:
        list[Span]: Every function, in file order. Empty when the text isn't valid Python.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[Span] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                found.append(Span(name, start, child.end_lineno or child.lineno))
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return sorted(found, key=lambda span: (span.start, -span.end))


def enclosing_function(source: str, line: int) -> Span | None:
    """Find the innermost function holding a line.

    Args:
        source (str): The file's text.
        line (int): The line, from 1.

    Returns:
        Span | None: The smallest function whose lines include it, or None.
    """
    holding = [span for span in functions(source) if span.start <= line <= span.end]
    return min(holding, key=lambda span: span.end - span.start, default=None)


def numbered(lines: Sequence[str], first: int, last: int) -> str:
    """Write some lines of a file with their numbers.

    Args:
        lines (Sequence[str]): The file's lines.
        first (int): The first line to write, from 1.
        last (int): The last line to write.

    Returns:
        str: Lines such as "  12 | return total", one per line.
    """
    first, last = max(1, first), min(len(lines), last)
    width = len(str(last))
    return "\n".join(f"{n:>{width}} | {lines[n - 1]}" for n in range(first, last + 1))


def window(
    lines: Sequence[str], first: int, last: int, focus: tuple[int, int], max_chars: int
) -> tuple[int, int]:
    """Choose the lines to show within a range, centered on some lines, within a size.

    Args:
        lines (Sequence[str]): The file's lines.
        first (int): The first line that may be shown.
        last (int): The last line that may be shown.
        focus (tuple[int, int]): The lines to center on (start, end).
        max_chars (int): Most characters, counting each line's number and newline.

    Returns:
        tuple[int, int]: The first and last line to show. Grows one line at a time on each
        side of the focus until the next line wouldn't fit.
    """
    first, last = max(1, first), min(len(lines), last)
    low = max(first, min(focus[0], last))
    high = min(last, max(focus[1], low))

    def cost(n: int) -> int:
        return len(lines[n - 1]) + 9

    used = sum(cost(n) for n in range(low, high + 1))
    while used > max_chars and high > low:
        used -= cost(high)
        high -= 1
    grew = True
    while grew:
        grew = False
        for step in (-1, 1):
            n = low - 1 if step < 0 else high + 1
            if first <= n <= last and used + cost(n) <= max_chars:
                used += cost(n)
                low, high = min(low, n), max(high, n)
                grew = True
    return low, high


def code_view(source: str, start: int, end: int, max_chars: int = CODE_CHARS) -> tuple[str, str]:
    """Give the code around a claim: its function, or the lines near it, numbered.

    Args:
        source (str): The claimed file's text at the PR's last commit.
        start (int): The claim's first line.
        end (int): The claim's last line.
        max_chars (int): Most characters.

    Returns:
        tuple[str, str]: The function's qualified name ("" outside any function) and the
        numbered lines: the whole function when it fits, else the part around the claim.
    """
    lines = source.splitlines()
    if not lines:
        return "", ""
    span = enclosing_function(source, start)
    if span is not None:
        low, high = window(lines, span.start, span.end, (start, end), max_chars)
        return span.qualname, numbered(lines, low, high)
    low, high = window(lines, start - AROUND_LINES, end + AROUND_LINES, (start, end), max_chars)
    return "", numbered(lines, low, high)


@dataclass(frozen=True)
class Hunk:
    """One hunk of a diff with context lines.

    Attributes:
        file (str): The file, as the PR leaves it.
        start (int): The hunk's first line at the PR's side.
        end (int): Its last line there.
        text (str): The hunk, from its @@ line on.
    """

    file: str
    start: int
    end: int
    text: str


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_hunks(diff: str) -> dict[str, list[Hunk]]:
    """Split a diff into hunks, by file.

    Args:
        diff (str): The output of `git diff` with context lines.

    Returns:
        dict[str, list[Hunk]]: Each file's hunks, in order. Deleted files are left out.
    """
    found: dict[str, list[Hunk]] = {}
    current: str | None = None
    hunk: list[str] = []
    start = count = 0

    def close() -> None:
        if current is not None and hunk:
            found.setdefault(current, []).append(
                Hunk(current, start, start + max(count, 1) - 1, "\n".join(hunk))
            )

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            close()
            hunk, current = [], None
            continue
        if line.startswith("+++ ") and not hunk:
            target = line[4:].strip()
            current = target[2:] if target.startswith("b/") else None
            continue
        match = _HUNK.match(line)
        if match:
            close()
            hunk = [line]
            start, count = int(match.group(1)), int(match.group(2) or "1")
            continue
        if hunk:
            hunk.append(line)
    close()
    return found


def near_hunks(hunks: Sequence[Hunk], start: int, end: int, max_chars: int = HUNK_CHARS) -> str:
    """Give the hunks nearest some lines, within a size, in file order.

    Args:
        hunks (Sequence[Hunk]): One file's hunks.
        start (int): The claim's first line.
        end (int): Its last line.
        max_chars (int): Most characters.

    Returns:
        str: The file's name, then the nearest hunks that fit, closest first chosen. A first
        hunk too big to fit is cut. "" when there are no hunks.
    """
    if not hunks:
        return ""

    def distance(hunk: Hunk) -> int:
        return max(0, hunk.start - end, start - hunk.end)

    header = f"--- {hunks[0].file}\n"
    room = max_chars - len(header)
    chosen: list[Hunk] = []
    for hunk in sorted(hunks, key=distance):
        if len(hunk.text) + 1 <= room:
            chosen.append(hunk)
            room -= len(hunk.text) + 1
        elif not chosen:
            cut = hunk.text[: max(0, room - 30)].rsplit("\n", 1)[0]
            chosen.append(Hunk(hunk.file, hunk.start, hunk.end, cut + "\n[... hunk cut ...]"))
            break
    return header + "\n".join(h.text for h in sorted(chosen, key=lambda h: h.start))


def defined_names(line: str) -> set[str]:
    """Collect the names one line of code defines.

    Args:
        line (str): The line, such as "def total(self, items, *, scale=1):".

    Returns:
        set[str]: A function's name and its arguments, a class's name, the target of an
        assignment (also `self.x = ...`), or the names a `from ... import` brings in.
    """
    text = line.strip()
    found = _DEF.match(text)
    if found:
        args = re.sub(r"\[[^\]]*\]|=[^,]*|:[^,]*", "", found.group(2))
        return {found.group(1)} | {a.strip(" *):") for a in args.split(",") if a.strip(" *):")}
    found = _CLASS.match(text) or _ASSIGN.match(text)
    if found:
        return {found.group(1)}
    found = _FROM_IMPORT.match(text)
    if found:
        return {part.split(" as ")[-1].strip(" ()") for part in found.group(1).split(",")} - {""}
    return set()


def added_names(diff: str) -> set[str]:
    """Collect the names a PR defines: on its added lines, and the modules of its new files.

    A test of code the PR adds fails at the base because one of these is missing. Names the
    added lines only use (such as "shape" in `x.shape`) don't count.

    Args:
        diff (str): The output of `git diff --unified=0`.

    Returns:
        set[str]: Names such as "total", "Cart" or a new keyword argument's name.
    """
    names: set[str] = set()
    new_file = False
    for line in diff.splitlines():
        if line.startswith("--- "):
            new_file = line.strip() == "--- /dev/null"
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if new_file and target.startswith("b/"):
                names.add(PurePosixPath(target[2:]).stem)
        elif line.startswith("+"):
            names.update(name for name in defined_names(line[1:]) if _NAME.fullmatch(name))
    return names


def _shared(a: Sequence[str], b: Sequence[str]) -> int:
    """Count the leading folder names two paths share.

    Args:
        a (Sequence[str]): One path's folders.
        b (Sequence[str]): The other's.

    Returns:
        int: How many leading folders are the same.
    """
    shared = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        shared += 1
    return shared


def choose_test_file(listing: Iterable[str], claim_file: str, hints: Sequence[str]) -> str | None:
    """Pick the existing test file a candidate's test goes next to.

    The first hint in the repo wins, whatever its name: hints are test files the retrieval step
    found using the changed function (some repos name them freely, such as pytest's
    testing/python/approx.py). Otherwise the test file closest to the claimed file in the
    folder tree, with a file named after its module (test_<module>.py) counting as two folders
    closer, and one in a tests, test or testing folder as one closer.

    Args:
        listing (Iterable[str]): The repo's Python files.
        claim_file (str): The claimed file.
        hints (Sequence[str]): Test files known to use the changed code, best first.

    Returns:
        str | None: The test file, or None when the repo has none.
    """
    files = set(listing)
    for hint in hints:
        if hint in files:
            return hint
    tests = sorted(path for path in files if is_test_file(path))
    if not tests:
        return None
    claim = PurePosixPath(claim_file)
    stem = claim.stem.lstrip("_")
    folders = claim.parent.parts

    def score(path: str) -> tuple[int, int, str]:
        test = PurePosixPath(path)
        shared = _shared(folders, test.parent.parts)
        distance = len(folders) + len(test.parent.parts) - 2 * shared
        named = test.stem in (f"test_{stem}", f"{stem}_test", f"unittest_{stem}")
        in_tests = any(part in ("tests", "test", "testing") for part in test.parent.parts)
        return (distance - (2 if named else 0) - (1 if in_tests else 0), len(path), path)

    return min(tests, key=score)


def place_test(test_file: str | None, finding_id: str) -> str | None:
    """Give the path of a candidate's generated test, next to an existing test file.

    Args:
        test_file (str | None): The existing test file, or None.
        finding_id (str): The candidate, such as "f3".

    Returns:
        str | None: Such as "pkg/tests/test_pr_reviewer_f3.py", or None without a test file.
    """
    if test_file is None:
        return None
    name = TEST_NAME.format(finding_id=finding_id)
    folder = posixpath.dirname(test_file)
    return f"{folder}/{name}" if folder else name


def conftests(listing: Iterable[str], test_path: str) -> list[str]:
    """List the conftest.py files pytest loads for a test file, nearest first.

    Args:
        listing (Iterable[str]): The repo's Python files.
        test_path (str): The test file.

    Returns:
        list[str]: The conftest.py files in its folder and every folder above it.
    """
    known = set(listing)
    found = []
    folder = posixpath.dirname(test_path)
    while True:
        candidate = f"{folder}/conftest.py" if folder else "conftest.py"
        if candidate in known:
            found.append(candidate)
        if not folder:
            return found
        folder = posixpath.dirname(folder)


def _is_fixture(decorator: ast.expr) -> bool:
    """Tell whether a decorator makes a pytest fixture.

    Args:
        decorator (ast.expr): The decorator, such as `@pytest.fixture(scope="module")`.

    Returns:
        bool: True for fixture, pytest.fixture and their called forms.
    """
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return target.attr == "fixture"
    return isinstance(target, ast.Name) and target.id == "fixture"


def fixtures(source: str) -> list[str]:
    """List the fixtures a conftest.py defines, with their arguments and docstring's first line.

    Args:
        source (str): The conftest.py's text.

    Returns:
        list[str]: Such as "tmp_cart(tmp_path): A cart saved in a temporary folder.", in
        file order. Empty when the text isn't valid Python.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not any(_is_fixture(d) for d in node.decorator_list):
            continue
        names = [a.arg for a in [*node.args.posonlyargs, *node.args.args]]
        doc = (ast.get_docstring(node) or "").strip().split("\n", 1)[0][:120]
        line = f"{node.name}({', '.join(names)})"
        found.append(f"{line}: {doc}" if doc else line)
    return found


def example_test(
    path: str, source: str, names: Iterable[str], max_chars: int = EXAMPLE_CHARS
) -> str:
    """Give a test file's imports and one of its tests, as an example of the repo's tests.

    Args:
        path (str): The test file.
        source (str): Its text.
        names (Iterable[str]): Names to prefer a test that uses, such as the changed
            function's.
        max_chars (int): Most characters.

    Returns:
        str: The path, the module-level imports and one test function (one using a name when
        there is one, else the first), cut to fit. "" when the text isn't valid Python.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    lines = source.splitlines()

    def text_of(node: ast.stmt) -> str:
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        return "\n".join(lines[start - 1 : (node.end_lineno or node.lineno)])

    imports = "\n".join(
        text_of(node) for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
    )[:IMPORT_CHARS]
    tests: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test"
        ):
            tests.append(node)
        elif isinstance(node, ast.ClassDef):
            tests.extend(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and item.name.startswith("test")
            )
    # The first name that some test uses decides: the function's own name before its class's.
    chosen = tests[:1]
    for name in (name for name in names if len(name) > 2):
        using = [t for t in tests if re.search(rf"\b{re.escape(name)}\b", text_of(t))]
        if using:
            chosen = using[:1]
            break
    body = text_of(chosen[0]) if chosen else ""
    text = f"# {path}\n{imports}\n\n\n{body}".rstrip()
    if len(text) > max_chars:
        text = text[: max_chars - 20].rsplit("\n", 1)[0] + "\n# [... cut ...]"
    return text


def short_names(qualname: str) -> list[str]:
    """Give the names a test of a function would mention.

    Args:
        qualname (str): Such as "Cart.total" or "Cart.__init__".

    Returns:
        list[str]: Such as ["total", "Cart"]; dunder methods give only their class.
    """
    parts = [part for part in qualname.split(".") if part]
    names = [part for part in reversed(parts) if not part.startswith("__")]
    return names[:2]


def build_materials(
    candidate: Candidate,
    sources: Mapping[str, str | None],
    hunks: Mapping[str, Sequence[Hunk]],
    test_file: str | None,
    test_source: str | None,
    fixture_lines: Sequence[str],
) -> Materials:
    """Put together one candidate's materials from what the survey read.

    Args:
        candidate (Candidate): The candidate.
        sources (Mapping[str, str | None]): The claimed files' text, None when unread.
        hunks (Mapping[str, Sequence[Hunk]]): The diff's hunks, by file.
        test_file (str | None): The existing test file the test goes next to.
        test_source (str | None): That file's text.
        fixture_lines (Sequence[str]): The fixtures its folder offers.

    Returns:
        Materials: The candidate's materials.
    """
    claim = candidate.claim
    source = sources.get(claim.file)
    function, code = ("", "")
    if source is not None:
        function, code = code_view(source, claim.start_line, claim.end_line)
    example = ""
    if test_file is not None and test_source is not None:
        example = example_test(test_file, test_source, short_names(function))
    return Materials(
        finding_id=candidate.finding_id,
        function=function,
        code=code,
        hunks=near_hunks(hunks.get(claim.file, []), claim.start_line, claim.end_line),
        test_path=place_test(test_file, candidate.finding_id),
        example=example,
        fixtures=tuple(fixture_lines[:MAX_FIXTURES]),
    )


def patch_hunks(pr: Instance, file: str) -> list[Hunk]:
    """Give a file's hunks from the PR's own commit diffs, without a sandbox.

    Args:
        pr (Instance): The PR.
        file (str): The file.

    Returns:
        list[Hunk]: The hunks of every reviewed commit that changed it, in commit order.
    """
    found: list[Hunk] = []
    for commit in pr.pr_commits:
        for patch in commit.diff:
            if patch.file == file:
                text = f"diff --git a/{file} b/{file}\n+++ b/{file}\n{patch.patch}"
                found.extend(parse_hunks(text).get(file, []))
    return found


def fallback_materials(pr: Instance, candidate: Candidate) -> Materials:
    """Give a candidate's materials when its PR couldn't be surveyed: the diff alone.

    Args:
        pr (Instance): The PR.
        candidate (Candidate): The candidate.

    Returns:
        Materials: The nearest hunks of the PR's own diff, with no code and no test place,
        so only the verifier checks it.
    """
    claim = candidate.claim
    hunks = patch_hunks(pr, claim.file)
    return Materials(
        finding_id=candidate.finding_id,
        hunks=near_hunks(hunks, claim.start_line, claim.end_line),
    )


def _git_to_text(box: Sandbox, root: str, args: Sequence[str], name: str) -> str:
    """Run git in the working copy, its output going to a file, and read the file.

    Args:
        box (Sandbox): The sandbox.
        root (str): The working copy.
        args (Sequence[str]): git's arguments, such as ["ls-files"].
        name (str): The output file's name.

    Returns:
        str: What git printed.

    Raises:
        SurveyError: If git fails.
        SandboxError: If the sandbox fails.
    """
    out = f"{WORK}/{name}"
    done = box.run(
        ["sh", "-c", 'out="$1"; shift; exec "$@" > "$out"', "sh", out, "git", "-C", root, *args],
        timeout=GIT_SECONDS,
        label=name.split(".")[0],
    )
    if not done.ok:
        raise SurveyError(f"git {args[0]} failed: {done.output.strip()[-300:]}")
    return box.read_file(out).decode("utf-8", errors="replace")


def _read_text(box: Sandbox, root: str, path: str) -> str | None:
    """Read a file of the working copy as text.

    Args:
        box (Sandbox): The sandbox.
        root (str): The working copy.
        path (str): The file, relative to the repo root.

    Returns:
        str | None: Its text, or None when it's missing or too big.
    """
    try:
        data = box.read_file(f"{root}/{path}", max_bytes=MAX_FILE_BYTES)
    except SandboxError:
        return None
    return data.decode("utf-8", errors="replace")


def survey_pr(
    ctx: TryContext,
    pr: Instance,
    image: str,
    candidates: Sequence[Candidate],
    hints: Mapping[str, Sequence[str]],
) -> PrSurvey:
    """Survey a PR in a sandbox from its prepared image, and build each candidate's materials.

    Args:
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        image (str): Its prepared image at the PR state.
        candidates (Sequence[Candidate]): The candidates to build materials for.
        hints (Mapping[str, Sequence[str]]): Test files known to use each changed function, by
            the function's qualified name, best first (from the retrieved context).

    Returns:
        PrSurvey: The changed lines, the added names and each candidate's materials.

    Raises:
        SurveyError: If git fails.
        RepoError, CheckoutError: If the PR's commits can't be checked out.
        SandboxError: If Docker or the sandbox fails.
    """
    with Sandbox(ctx.client, ctx.config, image, purpose="survey", log_dir=ctx.log_dir) as box:
        root, head = checkout(box, pr, "pr")
        base = base_of(box, root, pr)
        made = box.run(["mkdir", "-p", WORK], timeout=60, label="mkdir")
        if not made.ok:
            raise SandboxError(f"Can't make {WORK}: {made.output.strip()[-300:]}")
        diff_args = ["-c", "core.quotePath=false", "diff", "--no-color", "--no-ext-diff", "-M"]
        listing = _git_to_text(box, root, ["ls-files", "--", "*.py"], "files.txt").splitlines()
        changes = _git_to_text(
            box, root, [*diff_args, "--unified=0", base, head, "--", "*.py"], "changes.diff"
        )
        files = sorted({c.claim.file for c in candidates})
        context_diff = _git_to_text(
            box, root, [*diff_args, "--unified=3", base, head, "--", *files], "hunks.diff"
        )
        sources = {path: _read_text(box, root, path) for path in files}
        hunks = parse_hunks(context_diff)
        materials: dict[str, Materials] = {}
        read: dict[str, str | None] = {}
        for candidate in candidates:
            claim = candidate.claim
            source = sources.get(claim.file)
            span = enclosing_function(source, claim.start_line) if source else None
            wanted = list(hints.get(span.qualname, ())) if span else []
            test_file = choose_test_file(listing, claim.file, wanted)
            test_source = fixture_lines = None
            if test_file is not None:
                for path in [test_file, *conftests(listing, test_file)]:
                    if path not in read:
                        read[path] = _read_text(box, root, path)
                test_source = read[test_file]
                fixture_lines = [
                    line
                    for path in conftests(listing, test_file)
                    for line in fixtures(read[path] or "")
                ]
            materials[candidate.finding_id] = build_materials(
                candidate, sources, hunks, test_file, test_source, fixture_lines or []
            )
    changed = parse_diff(changes, "new")
    return PrSurvey(
        instance_id=pr.instance_id,
        base=base,
        head=head,
        changed={
            path: sorted(info["lines"]) for path, info in sorted(changed.items()) if info["lines"]
        },
        added_names=tuple(sorted(added_names(changes))),
        materials=materials,
    )
