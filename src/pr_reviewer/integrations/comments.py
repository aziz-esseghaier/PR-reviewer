"""Write the gate's proven findings as GitHub review comments.

Each comment says what the problem is, how the generated test proved it (passes on the base
commit, fails on the PR's commit with the claimed error), and shows the test in a collapsed
block. The claim's text comes from the model, which read the PR's untrusted text, so it's
escaped before it reaches GitHub's Markdown:
- HTML and Markdown characters are escaped, so it can't add links, images or hidden HTML;
- "@", "#" and "://" get a zero-width space, so it can't mention people, link issues or
  make a bare address clickable;
- newlines become spaces, and each part has a length cap.
The test code goes in a fenced block longer than any run of backticks in it, where GitHub
shows text as is. Our code writes every other part of the comment.

A hidden marker with a fingerprint of the claim lets a later run skip findings it already
posted. Comments sit on a line of the PR's diff, because GitHub only accepts those; a finding
whose lines aren't in the diff goes into the review's summary instead.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pr_reviewer.core.schemas import CATEGORY_NAMES, Claim

# Longest parts of a comment, in characters.
SUMMARY_CHARS = 300
DETAILS_CHARS = 700
TEST_CHARS = 6_000

# How far from the claim's lines a comment may sit, when none of them is in the diff.
NEAR_LINES = 10

# The hidden marker that names a posted finding.
MARKER = "pr-reviewer:"

ZERO_WIDTH = "​"

# Characters that start Markdown constructs; each gets a backslash.
_MARKDOWN = re.compile(r"([\\`*_\[\]()|~!])")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class ProvenFinding:
    """A finding the gate kept, with the test that proved it.

    Attributes:
        finding_id (str): The finding, such as "f2".
        claim (Claim): What the reviewer said.
        proof (str): "executed" (passes on the base, fails on the PR) or "new_code" (fails
            on the PR from a line it adds; the code doesn't exist on the base).
        test_file (str): Where the test was placed in the repo.
        test_code (str): The test.
        failure (str): The error the test hit on the PR's commit.
    """

    finding_id: str
    claim: Claim
    proof: str
    test_file: str
    test_code: str
    failure: str


def proven_finding(
    claim: Claim, finding_id: str, record: Mapping[str, Any]
) -> ProvenFinding | None:
    """Read the proving test out of a candidate's evidence record.

    Args:
        claim (Claim): The candidate's claim.
        finding_id (str): The candidate.
        record (Mapping[str, Any]): Its saved evidence record (evidence/store.py).

    Returns:
        ProvenFinding | None: The finding and its proof, or None when no executed test proves
        it (or the PR shows the change was deliberate).
    """
    for item in record.get("evidence") or []:
        if item.get("kind") != "execution" or not item.get("proof") or item.get("deliberate"):
            continue
        return ProvenFinding(
            finding_id=finding_id,
            claim=claim,
            proof=str(item["proof"]),
            test_file=str(item["test_file"]),
            test_code=str(item["test_code"]),
            failure=str(item.get("failure") or claim.expected_failure or "an error"),
        )
    return None


def shorten(text: str, limit: int) -> str:
    """Cut a text to a length, at a word if one ends near the cut.

    Args:
        text (str): The text.
        limit (int): Most characters kept, the "…" included.

    Returns:
        str: The text, or its start and "…".
    """
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit * 0.7:
        cut = cut[:space]
    return cut.rstrip() + "…"


def escape_markdown(text: str, limit: int) -> str:
    """Make untrusted text safe to show in a GitHub comment, as one plain line.

    Args:
        text (str): The text, such as a claim's summary written by the model.
        limit (int): Most characters kept before escaping.

    Returns:
        str: The text on one line, with HTML and Markdown escaped, and mentions, issue
        references and addresses broken by a zero-width space.
    """
    plain = _CONTROL.sub("", text.replace("\r", " ").replace("\n", " ").replace("\t", " "))
    plain = shorten(re.sub(r" {2,}", " ", plain).strip(), limit)
    plain = plain.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    plain = _MARKDOWN.sub(r"\\\1", plain)
    plain = plain.replace("@", "@" + ZERO_WIDTH).replace("#", "#" + ZERO_WIDTH)
    plain = re.sub(r"://", ":" + ZERO_WIDTH + "//", plain)
    return re.sub(r"(?i)\bwww\.", lambda m: m.group(0)[:-1] + ZERO_WIDTH + ".", plain)


def code_span(text: str) -> str:
    """Show a short value, such as a file path, as inline code.

    Args:
        text (str): The value.

    Returns:
        str: The value between backticks, with backticks and line breaks taken out.
    """
    clean = _CONTROL.sub("", text.replace("`", "").replace("\n", " ").replace("\r", " "))
    return f"`{shorten(clean, 200)}`"


def fenced(code: str, language: str = "python") -> str:
    """Put code in a fenced block that its own backticks can't close.

    Args:
        code (str): The code.
        language (str): The language name for highlighting.

    Returns:
        str: The fenced block.
    """
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    body = code if len(code) <= TEST_CHARS else code[:TEST_CHARS] + "\n# ... (cut)"
    return f"{fence}{language}\n{body.rstrip()}\n{fence}"


def fingerprint(claim: Claim) -> str:
    """Name a finding so a later run of the same PR can tell it was already posted.

    Args:
        claim (Claim): The claim.

    Returns:
        str: 16 hex characters from its file, category and summary (not its lines, which
        move when the PR changes).
    """
    text = f"{claim.file}\n{claim.category}\n{claim.summary}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def marker(claim: Claim) -> str:
    """Give the hidden marker a posted finding carries.

    Args:
        claim (Claim): The claim.

    Returns:
        str: Such as "<!-- pr-reviewer:0123456789abcdef -->".
    """
    return f"<!-- {MARKER}{fingerprint(claim)} -->"


def posted_fingerprints(bodies: Iterable[str]) -> set[str]:
    """Find the findings already posted, from the bodies of a PR's review comments.

    Args:
        bodies (Iterable[str]): The comments' texts.

    Returns:
        set[str]: The fingerprints their markers carry.
    """
    found: set[str] = set()
    for body in bodies:
        found.update(re.findall(r"<!-- " + re.escape(MARKER) + r"([0-9a-f]{16}) -->", body or ""))
    return found


def commentable_lines(patch: str) -> set[int]:
    """Find the lines of a file's new version that a patch shows, where comments may go.

    Args:
        patch (str): The file's patch against the PR's base, as GitHub gives it.

    Returns:
        set[int]: The new version's line numbers shown in the patch (added or unchanged).
    """
    lines: set[int] = set()
    number = 0
    for line in patch.splitlines():
        hunk = _HUNK.match(line)
        if hunk:
            number = int(hunk.group(1))
            continue
        if number == 0 or line.startswith("\\"):
            continue
        if line.startswith("-"):
            continue
        lines.add(number)
        number += 1
    return lines


def anchor(claim: Claim, diff_lines: Mapping[str, set[int]]) -> int | None:
    """Pick the diff line a finding's comment goes on.

    Args:
        claim (Claim): The claim, with its file and lines.
        diff_lines (Mapping[str, set[int]]): The commentable lines of each changed file.

    Returns:
        int | None: The last of the claim's lines that's in the diff, else the nearest diff
        line within NEAR_LINES; None when there's none (the finding goes in the summary).
    """
    lines = diff_lines.get(claim.file) or set()
    inside = [n for n in lines if claim.start_line <= n <= claim.end_line]
    if inside:
        return max(inside)

    def distance(n: int) -> int:
        return min(abs(n - claim.start_line), abs(n - claim.end_line))

    near = sorted((n for n in lines if distance(n) <= NEAR_LINES), key=lambda n: (distance(n), n))
    return near[0] if near else None


def proof_sentence(item: ProvenFinding, base: str, head: str) -> str:
    """Say how the test proved the finding.

    Args:
        item (ProvenFinding): The finding.
        base (str): The base commit's ID.
        head (str): The reviewed commit's ID.

    Returns:
        str: One sentence, written by our code from checked fields.
    """
    error = code_span(item.failure)
    if item.proof == "new_code":
        return (
            f"this test fails on this pull request's commit {code_span(head[:7])} with {error}, "
            "raised from a line the pull request adds. It can't run on the base commit "
            f"{code_span(base[:7])}, where that code doesn't exist yet."
        )
    return (
        f"this test passes on the base commit {code_span(base[:7])} and fails on this pull "
        f"request's commit {code_span(head[:7])} with {error}."
    )


def comment_body(item: ProvenFinding, base: str, head: str) -> str:
    """Write one finding's comment.

    Args:
        item (ProvenFinding): The finding and its proof.
        base (str): The base commit's ID.
        head (str): The reviewed commit's ID.

    Returns:
        str: The comment's Markdown.
    """
    claim = item.claim
    category = CATEGORY_NAMES.get(claim.category, claim.category)
    lines = [
        f"**Bug proven by a test** · {category} · reviewer confidence {claim.confidence:.2f}",
        "",
        f"**Problem:** {escape_markdown(claim.summary, SUMMARY_CHARS)}",
    ]
    if claim.details:
        lines += ["", f"**Why:** {escape_markdown(claim.details, DETAILS_CHARS)}"]
    test_file = item.test_file.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    test_file = _CONTROL.sub("", test_file).replace("@", "@" + ZERO_WIDTH)
    lines += [
        "",
        f"**Proof:** {proof_sentence(item, base, head)}",
        "",
        f"<details><summary>Generated test ({shorten(test_file, 200)})</summary>",
        "",
        fenced(item.test_code),
        "",
        "</details>",
        "",
        "<sub>Posted by pr-reviewer, which comments only when a generated test proves the "
        f"problem. It never approves or blocks a pull request.</sub> {marker(claim)}",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ReviewPlan:
    """What to post: the inline comments, and the review's summary.

    Attributes:
        comments (list[dict[str, Any]]): GitHub's inline comment objects (path, line, side,
            body).
        body (str): The review's summary, with any finding that couldn't sit on a diff line.
        posted (list[ProvenFinding]): The findings in this review.
        skipped (int): Findings already posted by an earlier run.
    """

    comments: list[dict[str, Any]]
    body: str
    posted: list[ProvenFinding]
    skipped: int


def plan_review(
    proven: Sequence[ProvenFinding],
    diff_lines: Mapping[str, set[int]],
    already: set[str],
    base: str,
    head: str,
    checked: int,
) -> ReviewPlan:
    """Lay out the review: one inline comment per new proven finding, and a summary.

    Args:
        proven (Sequence[ProvenFinding]): The findings the gate kept, best first.
        diff_lines (Mapping[str, set[int]]): The commentable lines of each changed file.
        already (set[str]): Fingerprints of findings posted before.
        base (str): The base commit's ID.
        head (str): The reviewed commit's ID.
        checked (int): How many candidate findings the evidence agent checked.

    Returns:
        ReviewPlan: The comments and the summary. It has no comment when nothing is new.
    """
    comments: list[dict[str, Any]] = []
    loose: list[str] = []
    posted: list[ProvenFinding] = []
    skipped = 0
    for item in proven:
        if fingerprint(item.claim) in already:
            skipped += 1
            continue
        posted.append(item)
        body = comment_body(item, base, head)
        line = anchor(item.claim, diff_lines)
        if line is None:
            where = (
                f"{code_span(item.claim.file)}, lines {item.claim.start_line}-{item.claim.end_line}"
            )
            loose += ["", "---", "", f"In {where}:", "", body]
            continue
        comments.append({"path": item.claim.file, "line": line, "side": "RIGHT", "body": body})
    count = len(posted)
    noun = "problem" if count == 1 else "problems"
    summary = [
        f"pr-reviewer found {count} {noun} proven by an executed test, out of {checked} "
        "candidate problems it checked. The others had no executed proof, so they aren't "
        "posted."
    ]
    return ReviewPlan(comments, "\n".join(summary + loose), posted, skipped)
