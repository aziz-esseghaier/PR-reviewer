"""The setup report: success rates by repo and year, failure causes, and the usable subset.

It reads the most recent setup of each dev PR in scope (those from a given year onward) from
setup_runs, and writes results/setup-dev/:
- summary.md: success by repo and by year, what got environments working (a cached recipe,
  the standard recipe or the agent), the failure categories, and model calls and minutes;
- usable_prs.json: the fixed usable subset, the dev PRs whose environment passed at both
  sides, which later experiments run on;
- config.json: how the setups ran (agent model, budget, pass rate, image);
- raw/runs.jsonl: every PR's latest setup, with its attempts (not in git).

Imported by other modules; the file isn't run directly.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pr_reviewer.benchmark.dataset import Instance


def _rate(done: int, total: int) -> str:
    """Write a success rate as "done/total (percent)".

    Args:
        done (int): Successes.
        total (int): Tries.

    Returns:
        str: Such as "7/9 (78%)", or "-" when nothing was tried.
    """
    return f"{done}/{total} ({done / total:.0%})" if total else "-"


def summary_markdown(
    prs: Sequence[Instance], runs: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> str:
    """Write the setup report as readable tables.

    Args:
        prs (Sequence[Instance]): The dev PRs in scope, from the config's `since` year onward.
        runs (Mapping[str, Mapping[str, Any]]): Each PR's latest setup_runs row, by ID.
        config (Mapping[str, Any]): The report's config.json contents.

    Returns:
        str: The Markdown text of summary.md.
    """
    tried = [pr for pr in prs if pr.instance_id in runs]
    good = [pr for pr in tried if runs[pr.instance_id]["verified"]]
    lines = [
        "# Environment setup: dev set",
        "",
        f"- {len(tried)} of {len(prs)} dev PRs from {config['since']} onward set up so far; "
        "agent model "
        f"{config['agent_model']}, at most {config['budget']['model_calls']} model calls and "
        f"{config['budget']['minutes']:g} minutes each",
        f"- Verified: the related tests (at most 5 files) passed, at least "
        f"{config['min_pass_rate']:.0%} of them, at the PR's base and again at its PR state, "
        "with the network off",
        "",
        "## Overall",
        "",
        f"Verified: {_rate(len(good), len(tried))}",
        "",
    ]
    for title, key in (("By repo", lambda pr: pr.repo), ("By year", lambda pr: str(pr.year))):
        groups: dict[str, list[Instance]] = defaultdict(list)
        for pr in tried:
            groups[key(pr)].append(pr)
        lines += [f"## {title}", "", "| | Verified |", "|---|---|"]
        for name in sorted(groups):
            members = groups[name]
            done = sum(bool(runs[pr.instance_id]["verified"]) for pr in members)
            lines.append(f"| {name} | {_rate(done, len(members))} |")
        lines.append("")
    sources = Counter(runs[pr.instance_id]["source"] for pr in good)
    lines += ["## What got environments working", "", "| Source | PRs |", "|---|---|"]
    for source, text in (
        ("cache", "a recipe kept from an earlier PR"),
        ("standard", "the standard recipe"),
        ("agent", "a recipe from the setup agent's model"),
    ):
        lines.append(f"| {text} | {sources.get(source, 0)} |")
    fallback = sum(
        bool((runs[pr.instance_id].get("details") or {}).get("tests_fallback")) for pr in good
    )
    lines += [
        "",
        f"{fallback} of the verified PRs change no Python code, so they were checked with a few "
        "of the repo's own tests instead of related ones.",
    ]
    failures = Counter(
        runs[pr.instance_id]["failure_category"]
        for pr in tried
        if not runs[pr.instance_id]["verified"]
    )
    lines += ["", "## Why setups failed", "", "| Failure category | PRs |", "|---|---|"]
    for category, count in failures.most_common():
        lines.append(f"| {category} | {count} |")
    if not failures:
        lines.append("| (none) | 0 |")
    calls = sum(int(runs[pr.instance_id]["model_calls"]) for pr in tried)
    minutes = sum(float(runs[pr.instance_id]["minutes"]) for pr in tried)
    lines += [
        "",
        f"Model calls in all: {calls}. Minutes in sandboxes: {minutes:,.0f}.",
        "",
        f"The usable subset (usable_prs.json) holds the {len(good)} verified PRs.",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    folder: Path,
    prs: Sequence[Instance],
    runs: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[str]:
    """Write results/setup-dev/: summary, usable subset, config and raw runs.

    Args:
        folder (Path): The folder. Created if missing; earlier files are replaced.
        prs (Sequence[Instance]): The dev PRs in scope. Runs of other PRs are left out.
        runs (Mapping[str, Mapping[str, Any]]): Each PR's latest setup_runs row, by ID.
        config (Mapping[str, Any]): How the setups ran.

    Returns:
        list[str]: The usable subset: verified PR IDs, sorted.
    """
    raw = folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    usable = sorted(pr.instance_id for pr in prs if runs.get(pr.instance_id, {}).get("verified"))

    def write(name: str, data: Any) -> None:
        (folder / name).write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )

    write("config.json", dict(config))
    write("usable_prs.json", usable)
    (folder / "summary.md").write_text(
        summary_markdown(prs, runs, config), encoding="utf-8", newline="\n"
    )
    with (raw / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for pr in prs:
            if pr.instance_id in runs:
                file.write(json.dumps(dict(runs[pr.instance_id]), default=str) + "\n")
    return usable
