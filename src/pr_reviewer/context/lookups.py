"""The reviewer model's lookups: checking what it asks for, and answering it in a sandbox.

The candidate reviewer (reviewer/candidates.py) may ask, once per PR, for up to a few lookups
in the repo at the PR's last commit:
- read_file: lines start_line to end_line of a file;
- search_symbol: where a name is defined and used.
The model only proposes them, as JSON; our code checks each one (a path must stay inside the
repo, a name must be a Python name), then answers them all in one indexer sandbox, network
off, with lookup_tool.py. A lookup that doesn't check out gets a short refusal instead of an
answer, so a bad lookup never costs a retry.

Imported by other modules; the file isn't run directly.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from docker import DockerClient
from pydantic import BaseModel, ConfigDict, ValidationError

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.retrieve import ContextError, probe_source
from pr_reviewer.core.schemas import RepoPath
from pr_reviewer.sandbox.checkout import checkout
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.runner import Sandbox, SandboxError

# The tools the model may ask for.
LookupTool = Literal["read_file", "search_symbol"]

# Where the lookup files go inside the sandbox (outside the repo).
LOOKUP_DIR = "/tmp/pr_reviewer_lookup"

# A name search_symbol accepts: a Python name, or several joined by dots.
SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,5}$")


@dataclass(frozen=True)
class LookupLimits:
    """How much a lookup may show.

    Attributes:
        read_lines (int): Most lines read_file shows.
        max_defs (int): Most definitions search_symbol lists.
        max_uses (int): Most uses search_symbol lists.
        max_chars (int): Most characters of one answer.
    """

    read_lines: int = 60
    max_defs: int = 8
    max_uses: int = 15
    max_chars: int = 2_000


class Lookup(BaseModel):
    """One lookup, as the model asked for it.

    Every field the reply format has is kept as written; check() says whether it can run.

    Attributes:
        tool (str): "read_file" or "search_symbol".
        path (str): For read_file, the file, relative to the repo root.
        start_line (int): For read_file, the first line.
        end_line (int): For read_file, the last line.
        symbol (str): For search_symbol, the name.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tool: str
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    symbol: str = ""

    def check(self) -> str:
        """Say why the lookup can't run.

        Returns:
            str: The reason, or "" when it can run.
        """
        if self.tool == "read_file":
            try:
                _PathCheck(path=self.path)
            except ValidationError:
                return "the path must be a file inside the repository, such as pkg/io.py"
            if self.start_line < 1 or self.end_line < self.start_line:
                return "start_line must be 1 or more, and end_line at least start_line"
            return ""
        if self.tool == "search_symbol":
            if not SYMBOL.match(self.symbol):
                return "the symbol must be a Python name, such as parse_header or Dataset.where"
            return ""
        return "the tool must be read_file or search_symbol"

    def describe(self) -> str:
        """Write the lookup in a few words.

        Returns:
            str: Such as "read_file pkg/io.py 10-40" or "search_symbol Dataset.where".
        """
        if self.tool == "read_file":
            return f"read_file {self.path} {self.start_line}-{self.end_line}"
        if self.tool == "search_symbol":
            return f"search_symbol {self.symbol}"
        return f"{self.tool[:40]} (unknown tool)"


class _PathCheck(BaseModel):
    """Checks a read_file path with the same rule as a finding's file."""

    path: RepoPath


class LookupResult(BaseModel):
    """The answer to one lookup.

    Attributes:
        lookup (Lookup): The lookup.
        ok (bool): Whether it found something.
        text (str): The answer, or why there's none.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lookup: Lookup
    ok: bool
    text: str


def lookup_source() -> str:
    """Give lookup_tool.py's source, to copy into sandboxes.

    Returns:
        str: The text of lookup_tool.py.
    """
    return resources.files("pr_reviewer.context").joinpath("lookup_tool.py").read_text("utf-8")


def parse_lookups(raw: Sequence[Any], limit: int) -> list[Lookup]:
    """Read the lookups of a reply, keeping the first few.

    Args:
        raw (Sequence[Any]): The reply's "lookups" list.
        limit (int): Most lookups kept.

    Returns:
        list[Lookup]: The lookups. An item that isn't even an object with a tool becomes a
        lookup of an unknown tool, refused later.
    """
    found = []
    for item in list(raw)[:limit]:
        try:
            found.append(Lookup.model_validate(item))
        except ValidationError:
            found.append(Lookup(tool="(not a lookup)"))
    return found


def answer_in_sandbox(
    box: Sandbox, root: str, lookups: Sequence[Lookup], limits: LookupLimits, timeout: float
) -> list[dict[str, Any]]:
    """Answer lookups that checked out, with lookup_tool.py in a sandbox.

    Args:
        box (Sandbox): The sandbox, network off, with the working copy at the PR's state.
        root (str): The working copy.
        lookups (Sequence[Lookup]): The lookups, each already checked.
        limits (LookupLimits): How much each answer may show.
        timeout (float): Seconds the tool may run.

    Returns:
        list[dict[str, Any]]: One {"ok", "text"} per lookup, in order.

    Raises:
        ContextError: If the tool fails or its answer doesn't match the lookups.
        SandboxError: If Docker or the sandbox fails.
    """
    made = box.run(["mkdir", "-p", LOOKUP_DIR], timeout=60, label="mkdir")
    if not made.ok:
        raise SandboxError(f"Can't make {LOOKUP_DIR}: {made.output.strip()[-300:]}")
    request = {
        "root": root,
        "lookups": [lookup.model_dump() for lookup in lookups],
        **asdict(limits),
    }
    box.write_files(
        LOOKUP_DIR,
        {
            "probe.py": probe_source(),
            "lookup_tool.py": lookup_source(),
            "request.json": json.dumps(request),
        },
    )
    result = box.run(
        [
            "python3",
            f"{LOOKUP_DIR}/lookup_tool.py",
            f"{LOOKUP_DIR}/request.json",
            f"{LOOKUP_DIR}/answer.json",
        ],
        timeout=timeout,
        label="lookup",
    )
    if not result.ok:
        raise ContextError(
            f"The lookup tool failed: {result.describe()}: {result.output.strip()[-800:]}"
        )
    try:
        answer = json.loads(box.read_file(f"{LOOKUP_DIR}/answer.json").decode("utf-8"))
        answers = list(answer["results"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ContextError(f"The lookup tool's answer isn't valid: {exc}") from None
    if len(answers) != len(lookups) or not all(isinstance(a, dict) for a in answers):
        raise ContextError("The lookup tool's answer doesn't match the lookups")
    return answers


def run_lookups(
    client: DockerClient,
    config: SandboxConfig,
    image: str,
    pr: Instance,
    lookups: Sequence[Lookup],
    limits: LookupLimits,
    log_dir: Path | None = None,
) -> list[LookupResult]:
    """Answer lookups in a fresh indexer sandbox, at the PR's last commit.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.
        image (str): The indexer image's name and tag.
        pr (Instance): The PR.
        lookups (Sequence[Lookup]): The lookups, in the model's order.
        limits (LookupLimits): How much each answer may show.
        log_dir (Path | None): Folder for the sandbox's command logs.

    Returns:
        list[LookupResult]: One answer per lookup, in order. Refused lookups get their reason
        without a sandbox.

    Raises:
        ContextError: If the lookup tool fails or its answer doesn't check out.
        RepoError, CheckoutError: If the PR's commit can't be checked out.
        SandboxError: If Docker or the sandbox fails.
    """
    problems = [lookup.check() for lookup in lookups]
    runnable = [lookup for lookup, problem in zip(lookups, problems, strict=True) if not problem]
    answers: list[dict[str, Any]] = []
    if runnable:
        timeout = config.timeouts.test_minutes * 60
        with Sandbox(client, config, image, purpose="lookup", log_dir=log_dir) as box:
            root, _ = checkout(box, pr, "pr")
            answers = answer_in_sandbox(box, root, runnable, limits, timeout)
    results: list[LookupResult] = []
    found = iter(answers)
    for lookup, problem in zip(lookups, problems, strict=True):
        if problem:
            results.append(
                LookupResult(lookup=lookup, ok=False, text=f"{lookup.describe()}: {problem}")
            )
            continue
        item = next(found)
        text = str(item.get("text", ""))[: limits.max_chars + 100]
        results.append(LookupResult(lookup=lookup, ok=bool(item.get("ok")), text=text))
    return results
