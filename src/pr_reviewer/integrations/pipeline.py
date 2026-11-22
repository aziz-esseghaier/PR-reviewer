"""Run the whole reviewer on one live pull request, the way the test-set run does.

Each stage reuses the benchmark's own code, with the settings of the measured run:
1. download: the repo and the PR's commits go into the repo volume, from a sandbox whose only
   way out is the allow-list proxy;
2. setup: the setup agent builds the PR's environment, trying cached recipes and the standard
   recipe before asking the model, and checks it with the repo's own tests;
3. context: the indexer collects the code around the changed functions;
4. static: ruff, mypy, bandit and semgrep run on the changed lines;
5. candidates: the candidate reviewer lists plausible problems;
6. evidence: the test writer tries to prove each one with a test that passes on the base and
   fails on the PR, and the intent check drops proofs the PR shows were deliberate. The
   verifier is off, as in the test-set run, because the gate keeps only tier 1;
7. gate: configs/gate.yaml picks the comments (tier 1, one per place, at most 3).
Every model call goes through the response cache, so a rerun of the same PR costs no quota.

The run names hold the PR and its head commit, so a new push gets fresh runs while a rerun of
the same commit continues where it stopped.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import TupleRow

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.benchmark.judge import check_reviewer
from pr_reviewer.context.candidates import Piece
from pr_reviewer.context.lookups import LookupLimits
from pr_reviewer.context.retrieve import (
    ContextError,
    candidates_path,
    ready_indexer,
    retrieval_setup,
    retrieve_candidates,
)
from pr_reviewer.context.store import save_context
from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import load_models_config
from pr_reviewer.core.limits import RetryPolicy
from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.checkpoints import open_checkpointer
from pr_reviewer.db.connection import connect
from pr_reviewer.evidence.commands import (
    DEFAULT_PARALLEL,
    EVIDENCE_PROFILE,
    MAX_WAIT_S,
    check_one,
)
from pr_reviewer.evidence.graph import SerialClient
from pr_reviewer.evidence.inputs import Candidate, load_candidates
from pr_reviewer.evidence.prompts import prompts_fingerprint
from pr_reviewer.evidence.store import load_records, start_evidence_run
from pr_reviewer.gate.policy import Checked, gate, load_policy
from pr_reviewer.integrations.comments import ProvenFinding, proven_finding
from pr_reviewer.reviewer import candidates as candidate_reviewer
from pr_reviewer.reviewer.commands import (
    DEFAULT_VARIANT,
    REVIEWER_PROFILE,
    LookupRunner,
    candidate_settings,
    static_input,
)
from pr_reviewer.reviewer.store import save_review, start_run
from pr_reviewer.sandbox.commands import logs_dir, ready_image
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.repos import update_repo
from pr_reviewer.sandbox.runner import Sandbox, docker_client
from pr_reviewer.setup_agent.agent import SetupDeps, run_setup
from pr_reviewer.setup_agent.attempt import TryContext
from pr_reviewer.setup_agent.commands import SETUP_PROFILE
from pr_reviewer.static.commands import run_one
from pr_reviewer.static.store import stored_static
from pr_reviewer.static.tools import ready_tools, static_setup

# Reviews asked for before giving up on a PR whose reviewer replies don't check out; each
# retry asks for fresh replies, as a rerun of the benchmark does.
REVIEW_TRIES = 2

# Rounds of the evidence agent: a candidate whose check failed gets fresh replies once.
EVIDENCE_ROUNDS = 2

Show = Callable[[str], None]


class LiveReviewError(Exception):
    """Raised when a stage can't run at all, such as when the PR's commits can't be found."""


@dataclass
class LiveReview:
    """How the review of a live PR went.

    Attributes:
        pr (Instance): The PR.
        stopped (str): Why the review ended early, such as "no working environment"; "" when
            every stage ran.
        candidates (int): Candidate problems the reviewer listed.
        tiers (dict[int, int]): How many candidates ended at each evidence tier.
        proven (list[ProvenFinding]): The findings the gate kept, best first, with their tests.
        notes (list[str]): One line per stage.
    """

    pr: Instance
    stopped: str = ""
    candidates: int = 0
    tiers: dict[int, int] = field(default_factory=dict)
    proven: list[ProvenFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_names(pr: Instance) -> tuple[str, str]:
    """Name the PR's review run and evidence run.

    Args:
        pr (Instance): The PR.

    Returns:
        tuple[str, str]: Such as ("live-candidates-octo__demo-7-0123456789ab", and the
        evidence run's name with "live-evidence").
    """
    suffix = f"{pr.instance_id}-{pr.head_commit[:12]}"
    return f"live-candidates-{suffix}", f"live-evidence-{suffix}"


def save_live_pr(conn: psycopg.Connection[TupleRow], pr: Instance) -> None:
    """Record a live PR in bench_prs, which the stages' tables point to.

    The tables only know dev and test PRs, and a live PR has no known issues, so it's stored
    like a clean dev PR. The Action's database is its own and lasts one job.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        pr (Instance): The PR.
    """
    conn.execute(
        """
        INSERT INTO bench_prs (instance_id, repo, year, kind, split, functional_issues,
                               evolvability_issues)
        VALUES (%s, %s, %s, 'clean', 'dev', 0, 0)
        ON CONFLICT (instance_id) DO NOTHING
        """,
        (pr.instance_id, pr.repo, pr.year),
    )


def _download(pr: Instance, show: Show) -> None:
    """Put the repo and the PR's commits in the repo volume.

    Args:
        pr (Instance): The PR.
        show (Show): Prints a progress line.

    Raises:
        LiveReviewError: If a reviewed commit is still missing afterwards.
    """
    client = docker_client()
    config = load_sandbox_config()
    with Sandbox(
        client,
        config,
        ready_image(client, config),
        purpose="repos",
        network=True,
        repos_writable=True,
        log_dir=logs_dir(),
    ) as sandbox:
        update = update_repo(sandbox, pr.repo, [pr], timeout=config.timeouts.download_minutes * 60)
    if update.missing:
        raise LiveReviewError(
            f"{len(update.missing)} of the PR's commits can't be downloaded, such as "
            f"{update.missing[0][:10]}"
        )
    show(f"download: {'cloned' if update.cloned else 'updated'} {pr.repo}")


def _setup(pr: Instance, ctx: TryContext, show: Show) -> dict[str, Any]:
    """Build and check the PR's environment with the setup agent.

    Args:
        pr (Instance): The PR.
        ctx (TryContext): Docker, the sandbox settings, the sandbox image and the log folder.
        show (Show): Prints a progress line.

    Returns:
        dict[str, Any]: The setup's outcome: verified or not, what worked, or why it failed.
    """
    models = load_models_config()
    with connect() as conn, open_checkpointer() as checkpointer:
        llm = LLMClient(
            SETUP_PROFILE, models, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        deps = SetupDeps(ctx=ctx, llm=llm, conn=conn, hosts=ctx.config.proxy.allowed_hosts)
        thread = f"setup-{pr.instance_id}-{pr.head_commit[:12]}"
        result = run_setup(deps, pr, checkpointer, thread, on_step=lambda line: show(f"  {line}"))
    return result.outcome


def _context(pr: Instance, show: Show) -> tuple[list[Piece], dict[str, Path]]:
    """Collect the PR's context with the indexer, as the benchmark's context build does.

    Args:
        pr (Instance): The PR.
        show (Show): Prints a progress line.

    Returns:
        tuple[list[Piece], dict[str, Path]]: The context pieces, and the stored file by PR
        (empty when the probe failed: the review then goes on without context).
    """
    client = docker_client()
    config = load_sandbox_config()
    profile = load_models_config().profile(REVIEWER_PROFILE)
    settings, _, key, _ = retrieval_setup(config, profile)
    image = ready_indexer(client, config)
    try:
        found = retrieve_candidates(client, config, image, pr, settings, logs_dir())
    except ContextError as exc:
        show(f"context: none ({str(exc)[:200]})")
        return [], {}
    path = candidates_path(key, pr.instance_id)
    with connect() as conn:
        save_context(conn, key, path, found)
    show(f"context: {len(found.pieces)} pieces around {len(found.changed)} changed functions")
    return list(found.pieces), {pr.instance_id: path}


def _static(pr: Instance, ctx: TryContext, show: Show) -> dict[str, Any]:
    """Run the static tools on the PR's changed lines.

    Args:
        pr (Instance): The PR.
        ctx (TryContext): Docker, the sandbox settings, the sandbox image and the log folder.
        show (Show): Prints a progress line.

    Returns:
        dict[str, Any]: The stored static run, by PR.
    """
    tools = ready_tools(ctx.client, ctx.config)
    settings, _, key = static_setup(ctx.config)
    with connect() as conn:
        line = run_one(ctx, tools, pr, settings, conn, key)
        stored = stored_static(conn, key)
    show(f"static: {line}")
    return {pr.instance_id: stored[pr.instance_id]}


def _candidates(
    pr: Instance, run: str, pieces: list[Piece], statics: dict[str, Any], show: Show
) -> list[Candidate]:
    """List the PR's candidate problems with the candidate reviewer.

    Args:
        pr (Instance): The PR.
        run (str): The review run's name.
        pieces (list[Piece]): The PR's context.
        statics (dict[str, Any]): The stored static run, by PR.
        show (Show): Prints a progress line.

    Returns:
        list[Candidate]: The candidates, in the reviewer's order; empty when it found none.

    Raises:
        LiveReviewError: If no review came out after REVIEW_TRIES tries.
    """
    models = load_models_config()
    profile = models.profile(REVIEWER_PROFILE)
    config = load_sandbox_config()
    extra, budget, _, _ = candidate_settings(DEFAULT_VARIANT, profile, config)
    settings = json.loads(
        json.dumps(
            {
                "name": run,
                "reviewer": "candidates",
                "live": True,
                "profile": REVIEWER_PROFILE,
                "model": check_reviewer(REVIEWER_PROFILE, models),
                **extra,
            }
        )
    )
    lookups = LookupRunner(config, LookupLimits())
    with connect() as conn, open_checkpointer() as checkpointer:
        start_run(
            conn,
            run,
            reviewer="candidates",
            profile=REVIEWER_PROFILE,
            model=settings["model"],
            split="dev",
            settings=settings,
        )
        llm = LLMClient(
            REVIEWER_PROFILE, models, cache=PostgresCache(conn), usage=PostgresUsageLog(conn)
        )
        for failures in range(REVIEW_TRIES):
            result = candidate_reviewer.review_candidates(
                llm,
                checkpointer,
                pr,
                f"{run}:{pr.instance_id}:{failures}",
                sample=failures,
                budget_chars=budget,
                variant=DEFAULT_VARIANT,
                static=static_input(statics[pr.instance_id]),
                pieces=pieces,
                explorer=lookups.for_pr(pr),
                on_step=lambda line: show(f"  {line}"),
            )
            save_review(conn, run, result, failures + (0 if result.ok else 1))
            if result.ok:
                break
        else:
            raise LiveReviewError("The reviewer gave no usable reply")
        found = load_candidates(conn, run).get(pr.instance_id, [])
    show(f"candidates: {len(found)}")
    return found


def _evidence(
    pr: Instance,
    run: str,
    source: str,
    found: list[Candidate],
    contexts: dict[str, Path],
    statics: dict[str, Any],
    ctx: TryContext,
    show: Show,
) -> dict[str, dict[str, Any]]:
    """Check each candidate with the evidence agent, verifier off.

    Args:
        pr (Instance): The PR.
        run (str): The evidence run's name.
        source (str): The review run the candidates come from.
        found (list[Candidate]): The candidates.
        contexts (dict[str, Path]): The stored context file, by PR.
        statics (dict[str, Any]): The stored static run, by PR.
        ctx (TryContext): Docker, the sandbox settings, the sandbox image and the log folder.
        show (Show): Prints a progress line.

    Returns:
        dict[str, dict[str, Any]]: Each candidate's evidence record, by finding ID.
    """
    models = load_models_config()
    profile = models.profile(EVIDENCE_PROFILE)
    model = check_reviewer(EVIDENCE_PROFILE, models)
    settings: dict[str, Any] = {
        "name": run,
        "source_run": source,
        "live": True,
        "profile": EVIDENCE_PROFILE,
        "model": model,
        "prompts_sha256": prompts_fingerprint(),
        "verifier": False,
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    lookups = LookupRunner(ctx.config, LookupLimits())
    with connect() as conn, open_checkpointer() as checkpointer:
        start_evidence_run(
            conn,
            run,
            source_run=source,
            profile=EVIDENCE_PROFILE,
            model=model,
            settings=settings,
        )
        llm = SerialClient(
            LLMClient(
                EVIDENCE_PROFILE,
                models,
                cache=PostgresCache(conn),
                usage=PostgresUsageLog(conn),
                retry_policy=RetryPolicy(max_wait_s=MAX_WAIT_S),
            )
        )
        records: dict[str, dict[str, Any]] = {}
        for _ in range(EVIDENCE_ROUNDS):
            records = {r["finding_id"]: r for r in load_records(conn, run)}
            left = [
                (
                    c.finding_id,
                    int(records[c.finding_id]["failures"]) if c.finding_id in records else 0,
                )
                for c in found
                if records.get(c.finding_id, {}).get("status") != "done"
            ]
            if not left:
                break
            check_one(
                run,
                DEFAULT_PARALLEL,
                False,
                checkpointer,
                ctx,
                profile,
                llm,
                lookups,
                pr,
                left,
                found,
                contexts,
                statics,
                key,
            )
        records = {r["finding_id"]: r for r in load_records(conn, run)}
    return records


def review_live(pr: Instance, show: Show = print) -> LiveReview:
    """Review a live PR with every stage, and pick the comments the gate keeps.

    Args:
        pr (Instance): The PR (live_pr.build_instance()).
        show (Show): Prints a progress line.

    Returns:
        LiveReview: What each stage did, and the proven findings to post.

    Raises:
        LiveReviewError: If the PR's commits can't be downloaded, or the reviewer never
            gives a usable reply.
        LimitError, SandboxError: If the model's quota runs out or Docker fails.
    """
    review = LiveReview(pr)

    def note(line: str) -> None:
        review.notes.append(line)
        show(line)

    with connect() as conn:
        save_live_pr(conn, pr)
    _download(pr, note)
    client = docker_client()
    config = load_sandbox_config()
    ctx = TryContext(client, config, ready_image(client, config), logs_dir())
    outcome = _setup(pr, ctx, show)
    if not outcome.get("verified"):
        review.stopped = f"no working environment ({outcome.get('failure_category')})"
        note(f"setup: {review.stopped}")
        return review
    note(f"setup: verified ({outcome.get('source')})")
    pieces, contexts = _context(pr, note)
    statics = _static(pr, ctx, note)
    candidates_run, evidence_run = run_names(pr)
    found = _candidates(pr, candidates_run, pieces, statics, note)
    review.candidates = len(found)
    if not found:
        return review
    records = _evidence(pr, evidence_run, candidates_run, found, contexts, statics, ctx, show)
    checked = []
    for order, candidate in enumerate(found):
        record = records.get(candidate.finding_id)
        tier = int(record["tier"]) if record and record["status"] == "done" else 4
        checked.append(Checked(pr.instance_id, candidate.finding_id, order, candidate.claim, tier))
        review.tiers[tier] = review.tiers.get(tier, 0) + 1
    tiers = ", ".join(f"tier {t}: {n}" for t, n in sorted(review.tiers.items()))
    note(f"evidence: {tiers}")
    for item in gate(checked, load_policy()):
        proven = proven_finding(item.claim, item.finding_id, records[item.finding_id])
        if proven is not None:
            review.proven.append(proven)
    note(f"gate: {len(review.proven)} comments")
    return review
