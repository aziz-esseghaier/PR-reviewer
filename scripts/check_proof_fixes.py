"""Check the proof fixes on the dev run's executed proofs (tier 1), before the test-set run.

Two fixes followed the dev evidence run:
- the test check refuses guards (an early return, hasattr, getattr with a default, a skip, an
  except that returns or passes), which let the base commit pass without running the code;
- the intent check finds more signs that a change was deliberate (notes such as "changed to"
  or versionchanged, removed deprecation warnings, tests calling the class's public API).
For each tier 1 record of the dev run, this script runs the new test check on its proving
test, then (for a test that still passes the check) surveys the PR in Docker to rebuild the
changed function, looks for intent signs and, when it finds some, asks the intent check. It
prints what each proof becomes and the judge's label, so you can compare tier 1 before and
after. Uses Docker, and a model call per proof with signs (cached, so a rerun costs nothing).

How to run (from the project folder, with the database and Docker Desktop up):
    python scripts/check_proof_fixes.py
    python scripts/check_proof_fixes.py --run evidence-tools-gpt-oss-120b
"""

import argparse
from collections import Counter

from pr_reviewer.benchmark.dataset import load_dataset
from pr_reviewer.benchmark.source import DATASET, default_dataset_path
from pr_reviewer.benchmark.split import SPLIT_SHA256, load_split, split_path, split_prs
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import load_env, load_models_config
from pr_reviewer.core.limits import RetryPolicy
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.connection import connect
from pr_reviewer.difftest.environments import pr_environment, ready_prepared
from pr_reviewer.evidence.commands import EVIDENCE_PROFILE, MAX_WAIT_S, SOURCE_RUN
from pr_reviewer.evidence.graph import SerialClient, ask
from pr_reviewer.evidence.inputs import load_candidates
from pr_reviewer.evidence.intent import signs
from pr_reviewer.evidence.prompts import (
    INTENT_FORMAT,
    INTENT_PROMPT,
    IntentReply,
    budget_chars,
    check_test,
    intent_request,
)
from pr_reviewer.evidence.report import judge_labels
from pr_reviewer.evidence.store import load_records
from pr_reviewer.evidence.survey import short_names, survey_pr
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.runner import docker_client
from pr_reviewer.setup_agent.attempt import TryContext


def main() -> int:
    """Check each tier 1 proof of a dev evidence run against the fixed checks.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="evidence-tools-gpt-oss-120b", help="the dev run")
    args = parser.parse_args()
    load_env()
    instances = load_dataset(default_dataset_path(), sha256=DATASET.sha256)
    split = load_split(split_path(), sha256=SPLIT_SHA256)
    dev = {pr.instance_id: pr for pr in split_prs(instances, split, "dev")}
    models = load_models_config()
    profile = models.profile(EVIDENCE_PROFILE)
    sandbox_config = load_sandbox_config()
    client = docker_client()
    ctx = TryContext(client, sandbox_config, ready_image(client, sandbox_config), logs_dir())
    outcome: Counter[tuple[str, str]] = Counter()
    with connect() as conn:
        candidates = load_candidates(conn, SOURCE_RUN)
        labels = judge_labels(conn, SOURCE_RUN, candidates, dev)
        proofs = [r for r in load_records(conn, args.run) if r["tier"] == 1]
        llm = SerialClient(
            LLMClient(
                EVIDENCE_PROFILE,
                models,
                cache=PostgresCache(conn),
                usage=PostgresUsageLog(conn),
                retry_policy=RetryPolicy(max_wait_s=MAX_WAIT_S),
            )
        )
        for record in proofs:
            pr = dev[record["instance_id"]]
            fid = record["finding_id"]
            label = labels[(pr.instance_id, fid)]
            candidate = next(c for c in candidates[pr.instance_id] if c.finding_id == fid)
            proving = next(a for a in reversed(record["details"]["attempts"]) if a.get("proof"))
            print(f"{pr.instance_id} {fid} ({label}): {candidate.claim.summary}")
            problem = check_test(proving["code"], candidate.expected or "")
            if problem:
                print(f"  refused by the test check: {problem}")
                outcome[("refused (guard)", label)] += 1
                continue
            env = pr_environment(conn, pr)
            images = ready_prepared(ctx, pr, env, on_step=lambda line: print(f"    {line}"))
            survey = survey_pr(ctx, pr, images["pr"], [candidate], {})
            materials = survey.materials[fid]
            found = signs(
                pr,
                candidate.claim.file,
                short_names(materials.function),
                candidate.expected,
                materials.code,
            )
            print(f"  function {materials.function!r}: {len(found)} signs")
            for sign in found:
                print(f"    {sign}")
            if not found:
                outcome[("kept (no signs)", label)] += 1
                continue
            messages, _ = intent_request(
                candidate.claim,
                candidate.expected,
                proving["code"],
                f"It {proving['base']}. It {proving['pr']}",
                found,
                materials.code,
                budget_chars(profile, INTENT_PROMPT),
            )
            asked = ask(llm, messages, INTENT_FORMAT, IntentReply, record.get("sample", 0))
            if asked.reply is None:
                print(f"  intent check: no usable reply ({asked.note})")
                outcome[("intent failed", label)] += 1
                continue
            word = "deliberate" if asked.reply.deliberate else "not deliberate"
            print(f"  intent check: {word}: {asked.reply.reason[:300]}")
            outcome[("dropped (deliberate)" if asked.reply.deliberate else "kept", label)] += 1
    print()
    for (what, label), count in sorted(outcome.items()):
        print(f"{what:<22} {label:<12} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
