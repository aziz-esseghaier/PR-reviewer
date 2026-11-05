"""Checking the differential runner on real tests whose outcome is already known.

Three kinds of cases, from the dev PRs whose environments passed setup:
- pr_tests: the tests a PR itself added or edited, in their PR-state version, run at the base
  and at the PR state. The PR's authors made them pass at the PR state; at the base they fail
  (the feature isn't there yet) or pass. Expected: fail_pass or pass_pass, never pass_fail;
  a pass_fail here would be a false proof.
- fix_tests: a known bug's fix commit (later in the same PR) that added or edited tests. Its
  tests, in their fix version, run at the base, the PR state and the fix commit. Expected:
  they fail at the PR state and pass at the fix ("caught"). The base result shows whether the
  bug could be proven the reviewer's way: passes at the base, fails at the PR state.
- related: for repos with no PR of the first kind, one PR's smallest related test file, run
  as it is at both sides. Expected: unchanged (pass_pass, or fail_fail for a test already
  failing at the base), and no flaky or not_run.
A test skipped at both sides (by the repo's own skip marks, such as a missing optional package
or tool) says nothing about the runner, so it counts as neither expected nor unexpected; the
summary shows how many there were.
All git work (diffs, file versions) happens inside a sandbox; the host only parses the text.

Imported by other modules; the file isn't run directly.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import psycopg
from psycopg.rows import TupleRow
from pydantic import BaseModel, ConfigDict

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.core.schemas import EnvRecipe
from pr_reviewer.difftest.environments import PreparedSide
from pr_reviewer.difftest.outcomes import LABELS, OUTCOMES, DiffOutcome, Outcome, Runner
from pr_reviewer.difftest.runner import DiffResult, FileSpec, SideSpec, runner_kind
from pr_reviewer.difftest.store import diff_with_store
from pr_reviewer.difftest.targets import changed_lines, changed_tests, find_classes, node_ids
from pr_reviewer.sandbox.checkout import base_of
from pr_reviewer.sandbox.repos import git, repo_dir
from pr_reviewer.sandbox.runner import OUTPUT_LIMIT, Sandbox
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.setup_agent.selection import django_labels, is_test_file

# The kinds of validation case.
Kind = Literal["pr_tests", "fix_tests", "related"]
KINDS: tuple[Kind, ...] = ("pr_tests", "fix_tests", "related")

# Outcomes that count as expected, per kind, for the base -> PR state run.
EXPECTED: dict[Kind, frozenset[Outcome]] = {
    "pr_tests": frozenset({"fail_pass", "pass_pass"}),
    "fix_tests": frozenset(OUTCOMES),  # judged by the PR state -> fix run instead
    "related": frozenset({"pass_pass", "fail_fail"}),
}

# Seconds one git command may take while building cases.
GIT_SECONDS = 300

# How each count is written in the summary: the outcomes, then tests skipped at both sides.
_COLUMNS: dict[str, str] = dict(LABELS.items(), skipped="skipped")


@dataclass(frozen=True)
class Case:
    """One validation case: tests to run on one PR, with a known expected outcome.

    Attributes:
        kind (Kind): "pr_tests", "fix_tests" or "related".
        instance_id (str): The PR.
        files (tuple[FileSpec, ...]): The test files to write at every side.
        targets (tuple[str, ...]): The tests to run.
        fix_commit (str | None): For fix_tests, the fix commit.
        issue (str): For fix_tests, the known issue's category, such as "F.2".
    """

    kind: Kind
    instance_id: str
    files: tuple[FileSpec, ...]
    targets: tuple[str, ...]
    fix_commit: str | None = None
    issue: str = ""


class CaseResult(BaseModel):
    """How a validation case went.

    Attributes:
        kind (Kind): The case's kind.
        instance_id (str): The PR.
        repo (str): Its repo.
        issue (str): For fix_tests, the known issue's category.
        fix_commit (str | None): For fix_tests, the fix commit.
        base_pr (DiffResult): The run from the base to the PR state.
        pr_fix (DiffResult | None): For fix_tests, the run from the PR state to the fix.
        expected (int): Tests whose outcome was as expected.
        unexpected (list[str]): The others, as "test: outcome", leaving out skipped ones.
        caught (bool | None): For fix_tests, whether a test failed at the PR state and passed
            at the fix.
        provable (int): Tests that pass at the base and fail at the PR state (pass_fail).
        skipped (int): Tests skipped at both sides of the judged run.
    """

    model_config = ConfigDict(frozen=True)

    kind: Kind
    instance_id: str
    repo: str
    issue: str
    fix_commit: str | None
    base_pr: DiffResult
    pr_fix: DiffResult | None
    expected: int
    unexpected: list[str]
    caught: bool | None
    provable: int
    skipped: int = 0


@dataclass
class CaseSource:
    """What building a PR's cases found, besides the cases.

    Attributes:
        notes (list[str]): Why some files or fixes gave no case.
    """

    notes: list[str] = field(default_factory=list)


def _file_at(box: Sandbox, git_dir: str, sha: str, path: str) -> str | None:
    """Read a file as it is at a commit.

    Args:
        box (Sandbox): The sandbox.
        git_dir (str): The repo's shared copy.
        sha (str): The commit.
        path (str): The file, relative to the repo root.

    Returns:
        str | None: Its text, or None when it doesn't exist there or is too big to read whole.
    """
    result = box.run(
        ["git", "-C", git_dir, "show", f"{sha}:{path}"], timeout=GIT_SECONDS, label="show"
    )
    if not result.ok or len(result.output) >= OUTPUT_LIMIT:
        return None
    return result.output


def _changed_test_case(
    box: Sandbox,
    git_dir: str,
    old: str,
    new: str,
    runner: Runner,
    source: CaseSource,
) -> tuple[tuple[FileSpec, ...], tuple[str, ...]]:
    """Find the test files and tests changed from one commit to another.

    Args:
        box (Sandbox): The sandbox.
        git_dir (str): The repo's shared copy.
        old (str): The older commit.
        new (str): The newer commit; the files are taken as they are there.
        runner (Runner): "pytest" or "django".
        source (CaseSource): Collects notes on files that gave no tests.

    Returns:
        tuple[tuple[FileSpec, ...], tuple[str, ...]]: The test files (newer version) and the
        changed tests' IDs.
    """
    names = git(
        box,
        ["-C", git_dir, "diff", "--name-only", "--diff-filter=AM", old, new],
        timeout=GIT_SECONDS,
        label="names",
    ).splitlines()
    files: list[FileSpec] = []
    targets: list[str] = []
    for path in (name for name in names if is_test_file(name)):
        content = _file_at(box, git_dir, new, path)
        if content is None:
            source.notes.append(f"{path}: can't read it at {new[:10]}")
            continue
        diff = git(
            box,
            ["-C", git_dir, "diff", "-U0", old, new, "--", path],
            timeout=GIT_SECONDS,
            label="diff",
        )
        if runner == "django" and not django_labels([path]):
            source.notes.append(f"{path}: not under Django's tests/ folder")
            continue
        found = changed_tests(content, changed_lines(diff).get(path, set()))
        classes = find_classes(content)
        ids: list[str] = []
        for test in found:
            under = node_ids(path, test, classes, runner)
            if not under:
                source.notes.append(
                    f"{path}: {test.cls}.{test.name} runs only through classes in other files"
                )
            ids.extend(test_id for test_id in under if test_id not in ids)
        if not ids:
            if not found:
                source.notes.append(f"{path}: no test function changed")
            continue
        files.append(FileSpec(path, content))
        targets.extend(ids)
    return tuple(files), tuple(targets)


def build_cases(
    ctx: TryContext,
    pr: Instance,
    runner: Runner,
    related: Sequence[str] = (),
) -> tuple[list[Case], CaseSource]:
    """Build a PR's validation cases, reading its commits inside a sandbox.

    Args:
        ctx (TryContext): Docker and the sandbox settings.
        pr (Instance): The PR.
        runner (Runner): "pytest" or "django".
        related (Sequence[str]): Its related test files, for a related case; empty for none.

    Returns:
        tuple[list[Case], CaseSource]: The cases, and notes on what gave none.
    """
    source = CaseSource()
    cases: list[Case] = []
    git_dir = repo_dir(pr.repo)
    with Sandbox(ctx.client, ctx.config, ctx.base_image, purpose="cases") as box:
        base = base_of(box, git_dir, pr)
        head = pr.head_commit
        files, targets = _changed_test_case(box, git_dir, base, head, runner, source)
        if targets:
            cases.append(Case("pr_tests", pr.instance_id, files, targets))
        commits = {commit.sha for commit in pr.all_commits}
        fixes: dict[str, str] = {}
        for issue in pr.changes:
            fix = issue.change_resolve_info
            if issue.is_functional and fix and fix.commit_sha in commits:
                fixes.setdefault(fix.commit_sha, issue.category)
        for sha, category in fixes.items():
            files, targets = _changed_test_case(box, git_dir, head, sha, runner, source)
            if targets:
                cases.append(Case("fix_tests", pr.instance_id, files, targets, sha, category))
            else:
                source.notes.append(f"fix {sha[:10]} ({category}): changes no test")
        if related:
            sizes = {}
            for path in related:
                size = box.run(
                    ["git", "-C", git_dir, "cat-file", "-s", f"{base}:{path}"],
                    timeout=GIT_SECONDS,
                    label="size",
                )
                if size.ok and size.output.strip().isdigit():
                    sizes[path] = int(size.output.strip())
            if sizes:
                path = min(sizes, key=lambda name: (sizes[name], name))
                content = _file_at(box, git_dir, base, path)
                target = django_labels([path]) if runner == "django" else [path]
                if content is not None and target:
                    cases.append(
                        Case("related", pr.instance_id, (FileSpec(path, content),), (target[0],))
                    )
    return cases, source


def skipped_both(test: DiffOutcome) -> bool:
    """Tell whether a test was skipped at both sides, so its outcome says nothing.

    Args:
        test (DiffOutcome): The test's outcome.

    Returns:
        bool: True when it didn't run at either side because it was skipped.
    """
    return test.outcome == "not_run" and test.old.reason == test.new.reason == "skipped"


def judge_case(case: Case, repo: str, base_pr: DiffResult, pr_fix: DiffResult | None) -> CaseResult:
    """Compare a case's outcomes with what's expected of its kind.

    Args:
        case (Case): The case.
        repo (str): Its PR's repo.
        base_pr (DiffResult): The run from the base to the PR state.
        pr_fix (DiffResult | None): For fix_tests, the run from the PR state to the fix.

    Returns:
        CaseResult: The case's verdict.
    """
    if case.kind == "fix_tests" and pr_fix is not None:
        judged = [test for test in pr_fix.tests if not skipped_both(test)]
        skipped = len(pr_fix.tests) - len(judged)
        good = [test for test in judged if test.outcome == "fail_pass"]
        caught: bool | None = bool(good)
        # Tests the fix didn't affect may legitimately stay as they were.
        bad = [test for test in judged if test.outcome in ("pass_fail", "flaky", "not_run")]
    else:
        judged = [test for test in base_pr.tests if not skipped_both(test)]
        skipped = len(base_pr.tests) - len(judged)
        allowed = EXPECTED[case.kind]
        good = [test for test in judged if test.outcome in allowed]
        bad = [test for test in judged if test.outcome not in allowed]
        caught = None
    return CaseResult(
        kind=case.kind,
        instance_id=case.instance_id,
        repo=repo,
        issue=case.issue,
        fix_commit=case.fix_commit,
        base_pr=base_pr,
        pr_fix=pr_fix,
        expected=len(good),
        unexpected=[f"{test.test}: {test.outcome}" for test in bad],
        caught=caught,
        provable=sum(test.outcome == "pass_fail" for test in base_pr.tests),
        skipped=skipped,
    )


def run_case(
    conn: psycopg.Connection[TupleRow],
    ctx: TryContext,
    pr: Instance,
    recipe: EnvRecipe,
    images: Mapping[PreparedSide, str],
    case: Case,
    again: bool = False,
) -> CaseResult:
    """Run one validation case and judge it.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection, for stored runs.
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        pr (Instance): The PR.
        recipe (EnvRecipe): Its recipe.
        images (Mapping[PreparedSide, str]): Its prepared images.
        case (Case): The case.
        again (bool): Run even when results are stored.

    Returns:
        CaseResult: The case's verdict.
    """
    base = SideSpec("base", images["base"])
    head = SideSpec("pr", images["pr"])
    base_pr = diff_with_store(
        conn, ctx, pr, recipe, base, head, case.files, case.targets, again=again
    )
    pr_fix = None
    if case.kind == "fix_tests" and case.fix_commit:
        fix = SideSpec("fix", images["pr"], case.fix_commit)
        pr_fix = diff_with_store(
            conn, ctx, pr, recipe, head, fix, case.files, case.targets, again=again
        )
    return judge_case(case, pr.repo, base_pr, pr_fix)


def _counts_row(name: str, results: Sequence[CaseResult]) -> str:
    """Write one table row of outcome counts for some cases' base -> PR state runs.

    Args:
        name (str): The row's name.
        results (Sequence[CaseResult]): The cases.

    Returns:
        str: The Markdown row.
    """
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(_split_counts(result.base_pr))
    tests = sum(len(result.base_pr.tests) for result in results)
    expected = sum(result.expected for result in results)
    cells = " | ".join(str(counts[outcome]) for outcome in (*OUTCOMES, "skipped"))
    return f"| {name} | {len(results)} | {tests} | {cells} | {expected} |"


def _split_counts(run: DiffResult) -> dict[str, int]:
    """Count a run's tests per outcome, with tests skipped at both sides apart from not_run.

    Args:
        run (DiffResult): The run.

    Returns:
        dict[str, int]: Every outcome, then "skipped".
    """
    counts: dict[str, int] = dict(run.counts().items())
    skipped = sum(skipped_both(test) for test in run.tests)
    counts["not_run"] -= skipped
    counts["skipped"] = skipped
    return counts


def summary_markdown(
    results: Sequence[CaseResult], notes: Mapping[str, Sequence[str]], config: Mapping[str, Any]
) -> str:
    """Write the validation summary as readable tables.

    Args:
        results (Sequence[CaseResult]): Every case's verdict.
        notes (Mapping[str, Sequence[str]]): Per PR, why some files or fixes gave no case.
        config (Mapping[str, Any]): The settings, from config.json.

    Returns:
        str: The Markdown text of summary.md.
    """
    prs = {result.instance_id for result in results}
    header = " | ".join(LABELS[outcome] for outcome in OUTCOMES)
    lines = [
        "# Differential test runner: validation on known tests",
        "",
        f"- {len(results)} cases on {len(prs)} dev PRs; each side's tests ran "
        f"{config['runs_per_side']} times, at most {config['difftest_minutes']:g} minutes a run",
        "- Outcomes are from the base to the PR state. Expected: the PR's own tests "
        "fail→pass or pass→pass; related tests unchanged (pass→pass or fail→fail); a bug "
        "fix's tests fail at the PR state and pass at the fix.",
        "- Skipped: tests the repo's own skip marks skipped at both sides (such as a missing "
        'optional package). They\'re not in "not run" and count as neither expected nor not.',
        "",
        "## Outcomes by kind",
        "",
        f"| Kind | Cases | Tests | {header} | skipped | As expected |",
        "|---|---|---|" + "---|" * (len(OUTCOMES) + 1) + "---|",
    ]
    for kind in KINDS:
        chosen = [result for result in results if result.kind == kind]
        if chosen:
            lines.append(_counts_row(kind, chosen))
    fixes = [result for result in results if result.kind == "fix_tests"]
    if fixes:
        lines += [
            "",
            "## Known bugs: the fix's tests",
            "",
            "Caught: a test fails at the PR state and passes at the fix. Provable: a test also "
            "passes at the base, so the bug shows as pass→fail.",
            "",
            "| PR | Issue | Fix | Tests | Caught | PR→fix outcomes | Provable from base |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in fixes:
            assert result.pr_fix is not None
            outcomes = ", ".join(
                f"{_COLUMNS[outcome]} {count}"
                for outcome, count in _split_counts(result.pr_fix).items()
                if count
            )
            lines.append(
                f"| {result.instance_id} | {result.issue} | {(result.fix_commit or '')[:10]} | "
                f"{len(result.pr_fix.tests)} | {'yes' if result.caught else 'no'} | "
                f"{outcomes} | {result.provable} |"
            )
        caught = sum(bool(result.caught) for result in fixes)
        provable = sum(result.provable > 0 for result in fixes)
        lines += [
            "",
            f"Caught {caught} of {len(fixes)} known bugs; {provable} provable as pass→fail "
            "from the base.",
        ]
    odd = [(result, line) for result in results for line in result.unexpected]
    lines += ["", "## Not as expected", ""]
    if not odd:
        lines.append("Every test's outcome was as expected.")
    for result, line in odd:
        lines.append(f"- {result.kind}, {result.instance_id}: {line}")
    noted = [(pr, note) for pr, found in sorted(notes.items()) for note in found]
    if noted:
        lines += ["", "## Files or fixes that gave no case", ""]
        lines += [f"- {pr}: {note}" for pr, note in noted]
    return "\n".join(lines) + "\n"


def write_validation(
    folder: Path,
    results: Sequence[CaseResult],
    notes: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
) -> None:
    """Write results/difftest-validation/: summary, cases, config and raw outcomes.

    Args:
        folder (Path): The folder. Created if missing; earlier files are replaced.
        results (Sequence[CaseResult]): Every case's verdict.
        notes (Mapping[str, Sequence[str]]): Per PR, why some files or fixes gave no case.
        config (Mapping[str, Any]): The settings.
    """
    raw = folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    def write(name: str, data: Any) -> None:
        (folder / name).write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    write("config.json", dict(config))
    write(
        "cases.json",
        [
            {
                "kind": result.kind,
                "instance_id": result.instance_id,
                "issue": result.issue,
                "fix_commit": result.fix_commit,
                "targets": list(result.base_pr.targets),
                "base_pr": result.base_pr.counts(),
                "pr_fix": result.pr_fix.counts() if result.pr_fix else None,
                "expected": result.expected,
                "unexpected": result.unexpected,
                "caught": result.caught,
                "provable": result.provable,
                "skipped": result.skipped,
            }
            for result in results
        ],
    )
    (folder / "summary.md").write_text(
        summary_markdown(results, notes, config), encoding="utf-8", newline="\n"
    )
    with (raw / "results.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(result.model_dump_json() + "\n")


def run_validation(
    conn: psycopg.Connection[TupleRow],
    ctx: TryContext,
    plan: Sequence[tuple[Instance, EnvRecipe, Mapping[PreparedSide, str], Sequence[str]]],
    on_line: Callable[[str], None],
    again: bool = False,
) -> tuple[list[CaseResult], dict[str, list[str]]]:
    """Build and run every PR's validation cases.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection, for stored runs.
        ctx (TryContext): Docker, the sandbox settings and the log folder.
        plan (Sequence[tuple[Instance, EnvRecipe, Mapping[PreparedSide, str], Sequence[str]]]):
            Per PR: the PR, its recipe, its prepared images and its related test files (empty
            unless it gets a related case).
        on_line (Callable[[str], None]): Called with a progress line.
        again (bool): Run even when results are stored.

    Returns:
        tuple[list[CaseResult], dict[str, list[str]]]: Every case's verdict, and per PR the
        notes on what gave no case.
    """
    results: list[CaseResult] = []
    notes: dict[str, list[str]] = {}
    for pr, recipe, images, related in plan:
        cases, source = build_cases(ctx, pr, runner_kind(recipe), related)
        if source.notes:
            notes[pr.instance_id] = source.notes
        if not cases:
            on_line(f"{pr.instance_id}: no case ({'; '.join(source.notes) or 'nothing changed'})")
        for case in cases:
            result = run_case(conn, ctx, pr, recipe, images, case, again)
            took = result.base_pr.minutes + (result.pr_fix.minutes if result.pr_fix else 0)
            stored = " (stored)" if result.base_pr.reused else f" ({took:.1f} min)"
            verdict = (
                f"caught={result.caught}"
                if result.caught is not None
                else f"{result.expected}/{len(result.base_pr.tests)} as expected"
            )
            on_line(f"{pr.instance_id} {case.kind}: {verdict}{stored}")
            results.append(result)
    return results, notes
