"""Static tool warnings: reading each tool's JSON output, and keeping those on changed lines.

Each tool writes its own JSON shape. The parsers here turn all four into one record,
ToolWarning: the tool, its rule (such as "F821" or "attr-defined"), the message, and the file
and lines it points at, relative to the repo root. The output describes untrusted code, so it
is checked like any other input: paths must stay inside the repo, lines must be numbers, and
messages are cut to a fixed length.

A warning is kept for review when its lines touch a line the PR added or changed. Warnings on
untouched lines are about older code, which the PR didn't introduce.

Imported by other modules; the file isn't run directly.
"""

import json
import posixpath
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from pr_reviewer.core.schemas import RepoPath, ToolEvidence

# The static tools, in the order they run.
Tool = Literal["ruff", "mypy", "bandit", "semgrep"]
TOOLS: tuple[Tool, ...] = ("ruff", "mypy", "bandit", "semgrep")

# Longest message kept, in characters.
MESSAGE_CHARS = 500

# A tool's rule ID as it may appear: letters, digits and a few separators.
RULE = re.compile(r"^[A-Za-z0-9_.:/-]{1,200}$")


class ToolWarning(BaseModel):
    """One warning a static tool raised.

    Attributes:
        tool (Tool): The tool.
        rule (str): Its rule or error code, such as "B006" or "attr-defined".
        message (str): Its message, on one line.
        file (RepoPath): The file, relative to the repo root.
        line (int): The first line it points at, from 1.
        end_line (int): The last line it points at.
        severity (str): The tool's own level, in lower case, such as "error" or "medium".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: Tool
    rule: str = Field(pattern=RULE.pattern)
    message: str = Field(max_length=MESSAGE_CHARS)
    file: RepoPath
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    severity: str = Field(default="", max_length=20)

    @model_validator(mode="before")
    @classmethod
    def _fit(cls, data: Any) -> Any:
        """Put a message on one line and cut it, and fix a missing or backwards end line.

        Args:
            data (Any): The raw fields.

        Returns:
            Any: The fields, ready to check.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        message = " ".join(str(data.get("message", "")).split())
        if len(message) > MESSAGE_CHARS:
            message = message[: MESSAGE_CHARS - 3] + "..."
        data["message"] = message
        line = data.get("line")
        end = data.get("end_line")
        if isinstance(line, int) and (not isinstance(end, int) or end < line):
            data["end_line"] = line
        data["severity"] = str(data.get("severity") or "").lower()[:20]
        return data

    def touches(self, lines: Iterable[int]) -> bool:
        """Tell whether the warning's lines include any of the given lines.

        Args:
            lines (Iterable[int]): Lines of the same file.

        Returns:
            bool: True when one of them is between line and end_line.
        """
        return any(self.line <= line <= self.end_line for line in lines)

    def evidence(self, confirmed: bool = False) -> ToolEvidence:
        """Give the warning as tier-2 evidence.

        Args:
            confirmed (bool): Whether the verifier judged the warning to describe the claimed
                problem; only then does the evidence back the claim.

        Returns:
            ToolEvidence: The same tool, rule, message, file and first line.
        """
        return ToolEvidence(
            tool=self.tool,
            rule=self.rule,
            message=self.message,
            file=self.file,
            line=self.line,
            confirmed=confirmed,
        )

    def text(self) -> str:
        """Write the warning on one line, as the reviewer model sees it.

        Returns:
            str: Such as "pkg/io.py:12 mypy [attr-defined] "Foo" has no attribute "bar"".
        """
        where = f"{self.file}:{self.line}"
        if self.end_line != self.line:
            where += f"-{self.end_line}"
        return f"{where} {self.tool} [{self.rule}] {self.message}"


def repo_relative(path: str, root: str) -> str | None:
    """Turn a path a tool printed into a path relative to the repo root.

    Args:
        path (str): The path, absolute inside the sandbox or relative to the root.
        root (str): The repo root inside the sandbox, such as "/work/psf__requests".

    Returns:
        str | None: The relative path, or None when it isn't inside the repo.
    """
    path = path.replace("\\", "/")
    if path.startswith("/"):
        prefix = root.rstrip("/") + "/"
        if not path.startswith(prefix):
            return None
        path = path[len(prefix) :]
    path = posixpath.normpath(path)
    if path in (".", "") or path.startswith("../") or path == "..":
        return None
    return path


def _make(raw: dict[str, Any], root: str) -> ToolWarning | None:
    """Build one warning, or None when its fields don't check out.

    Args:
        raw (dict[str, Any]): The fields, with "file" as the tool printed it.
        root (str): The repo root inside the sandbox.

    Returns:
        ToolWarning | None: The warning, or None.
    """
    file = repo_relative(str(raw.get("file", "")), root)
    if file is None:
        return None
    try:
        return ToolWarning.model_validate({**raw, "file": file})
    except ValidationError:
        return None


def _number(value: Any) -> int | None:
    """Read a line number.

    Args:
        value (Any): The value.

    Returns:
        int | None: It, when it's a whole number (not a bool), else None.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_ruff(text: str, root: str) -> list[ToolWarning]:
    """Read ruff's JSON output (--output-format json).

    Args:
        text (str): The output: a list of diagnostics.
        root (str): The repo root inside the sandbox.

    Returns:
        list[ToolWarning]: The warnings. A syntax error has no code; it's kept as
        "syntax-error".

    Raises:
        ValueError: If the output isn't the JSON list ruff writes.
    """
    items = json.loads(text) if text.strip() else []
    if not isinstance(items, list):
        raise ValueError("ruff's output isn't a JSON list")
    found = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = item.get("location") or {}
        end = item.get("end_location") or {}
        warning = _make(
            {
                "tool": "ruff",
                "rule": item.get("code") or "syntax-error",
                "message": item.get("message", ""),
                "file": item.get("filename", ""),
                "line": _number(start.get("row")),
                "end_line": _number(end.get("row")),
                "severity": "error" if item.get("code") is None else "warning",
            },
            root,
        )
        if warning is not None:
            found.append(warning)
    return found


def parse_mypy(text: str, root: str) -> list[ToolWarning]:
    """Read mypy's JSON output (--output json): one JSON object per line.

    Notes (severity "note") explain the error before them, so they're left out.

    Args:
        text (str): The output.
        root (str): The repo root inside the sandbox.

    Returns:
        list[ToolWarning]: The errors.

    Raises:
        ValueError: If a line isn't a JSON object.
    """
    found = []
    for line in text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError("a line of mypy's output isn't a JSON object")
        if item.get("severity") == "note":
            continue
        message = str(item.get("message", ""))
        if item.get("hint"):
            message += f" ({item['hint']})"
        warning = _make(
            {
                "tool": "mypy",
                "rule": item.get("code") or "misc",
                "message": message,
                "file": item.get("file", ""),
                "line": _number(item.get("line")),
                "severity": item.get("severity", "error"),
            },
            root,
        )
        if warning is not None:
            found.append(warning)
    return found


def parse_bandit(text: str, root: str) -> list[ToolWarning]:
    """Read bandit's JSON output (--format json).

    Args:
        text (str): The output: an object with "results".
        root (str): The repo root inside the sandbox.

    Returns:
        list[ToolWarning]: The warnings.

    Raises:
        ValueError: If the output isn't the JSON object bandit writes.
    """
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise ValueError("bandit's output isn't the expected JSON object")
    found = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        lines = [n for n in item.get("line_range") or [] if _number(n) is not None]
        warning = _make(
            {
                "tool": "bandit",
                "rule": item.get("test_id", ""),
                "message": item.get("issue_text", ""),
                "file": item.get("filename", ""),
                "line": _number(item.get("line_number")),
                "end_line": max(lines) if lines else None,
                "severity": item.get("issue_severity", ""),
            },
            root,
        )
        if warning is not None:
            found.append(warning)
    return found


def semgrep_rule(check_id: str) -> str:
    """Shorten a semgrep rule ID to the rule's own name.

    Rules read from a folder get its path in their ID, such as
    "opt.tools.semgrep-rules.lang.correctness.useless-eqeq.useless-eqeq".

    Args:
        check_id (str): The ID semgrep reported.

    Returns:
        str: The last part, such as "useless-eqeq".
    """
    return check_id.rsplit(".", 1)[-1]


def parse_semgrep(text: str, root: str) -> list[ToolWarning]:
    """Read semgrep's JSON output (--json).

    Args:
        text (str): The output: an object with "results".
        root (str): The repo root inside the sandbox.

    Returns:
        list[ToolWarning]: The warnings.

    Raises:
        ValueError: If the output isn't the JSON object semgrep writes.
    """
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise ValueError("semgrep's output isn't the expected JSON object")
    found = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        extra = item.get("extra") or {}
        warning = _make(
            {
                "tool": "semgrep",
                "rule": semgrep_rule(str(item.get("check_id", ""))),
                "message": extra.get("message", ""),
                "file": item.get("path", ""),
                "line": _number((item.get("start") or {}).get("line")),
                "end_line": _number((item.get("end") or {}).get("line")),
                "severity": extra.get("severity", ""),
            },
            root,
        )
        if warning is not None:
            found.append(warning)
    return found


# Each tool's parser.
PARSERS: dict[Tool, Callable[[str, str], list[ToolWarning]]] = {
    "ruff": parse_ruff,
    "mypy": parse_mypy,
    "bandit": parse_bandit,
    "semgrep": parse_semgrep,
}


def on_changed_lines(
    warnings: Iterable[ToolWarning], changed: Mapping[str, Iterable[int]]
) -> list[ToolWarning]:
    """Keep the warnings that touch a line the PR added or changed, each once.

    Args:
        warnings (Iterable[ToolWarning]): Every warning.
        changed (Mapping[str, Iterable[int]]): The changed lines, by file.

    Returns:
        list[ToolWarning]: The warnings kept, sorted by file, line, tool and rule.
    """
    lines = {file: set(numbers) for file, numbers in changed.items()}
    kept = {
        warning
        for warning in warnings
        if warning.file in lines and warning.touches(lines[warning.file])
    }
    return sorted(kept, key=lambda w: (w.file, w.line, TOOLS.index(w.tool), w.rule, w.message))
