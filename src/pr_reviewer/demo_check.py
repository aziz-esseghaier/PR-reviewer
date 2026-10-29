"""End-to-end check of the demo graph with one model: a first run asks the model, and a second
identical run is answered entirely from the cache.

Both runs use the same sample number (see LLMClient.complete). The caller picks one not used
before, so replies cached earlier can't answer the first run. Requests and cache hits are
counted from the usage entries the client records, the same ones that go to the llm_usage
table. scripts/check_foundations.py runs this check for each model profile.

Imported by that script and the tests; the file isn't run directly.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from pr_reviewer.core.llm import LLMClient
from pr_reviewer.core.usage import UsageEntry, UsageLog
from pr_reviewer.demo_graph import run_demo


class CountingUsageLog:
    """A usage log that keeps a copy of every entry, so a run's requests can be counted.

    Entries are passed on to another log (the PostgreSQL one in real runs), so the daily
    limits and reports keep working.

    Attributes:
        entries (list[UsageEntry]): Every entry recorded through this log, oldest first.
    """

    def __init__(self, inner: UsageLog | None = None) -> None:
        """Wrap a log.

        Args:
            inner (UsageLog | None): The log to pass entries on to. None keeps them only here.
        """
        self.entries: list[UsageEntry] = []
        self._inner = inner

    def record(self, entry: UsageEntry) -> None:
        """Keep one entry and pass it on.

        Args:
            entry (UsageEntry): The request or cache hit to save.
        """
        self.entries.append(entry)
        if self._inner is not None:
            self._inner.record(entry)

    def recent(self, provider: str, model: str, since: datetime) -> list[tuple[datetime, int]]:
        """List a model's requests since a moment, from the wrapped log.

        Args:
            provider (str): Provider name.
            model (str): Model ID.
            since (datetime): Only requests after this moment count.

        Returns:
            list[tuple[datetime, int]]: (time, tokens) of each request, oldest first. Empty
            without a wrapped log.
        """
        return self._inner.recent(provider, model, since) if self._inner is not None else []


@dataclass(frozen=True)
class RunCount:
    """What one demo run cost and produced.

    Attributes:
        sent (int): Requests sent to the provider, retries included.
        cached (int): Replies served from the cache.
        calls (int): Model calls that got a reply, from the provider or the cache.
        tokens (int): Tokens of the requests sent. Cached replies count none.
        seconds (float): How long the run took.
        review (str): The review text it produced.
    """

    sent: int
    cached: int
    calls: int
    tokens: int
    seconds: float
    review: str


@dataclass(frozen=True)
class CheckResult:
    """The outcome of the check for one model profile.

    Attributes:
        profile (str): The model profile checked.
        first (RunCount): The run that should reach the model.
        second (RunCount): The run that should be answered from the cache.
        problems (tuple[str, ...]): What went wrong, in plain words. Empty when it passed.
    """

    profile: str
    first: RunCount
    second: RunCount
    problems: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the check found no problem."""
        return not self.problems


def count_run(entries: Sequence[UsageEntry], seconds: float, review: str) -> RunCount:
    """Count the requests and cache hits of one run.

    Args:
        entries (Sequence[UsageEntry]): The usage entries recorded during the run.
        seconds (float): How long the run took.
        review (str): The review text it produced.

    Returns:
        RunCount: The counts.
    """
    sent = [entry for entry in entries if entry.outcome != "cache_hit"]
    return RunCount(
        sent=len(sent),
        cached=sum(entry.outcome == "cache_hit" for entry in entries),
        # Each call that got a reply ends with exactly one "ok" or "cache_hit" entry; the
        # failed tries before an "ok" are retries of the same call.
        calls=sum(entry.outcome in ("ok", "cache_hit") for entry in entries),
        tokens=sum(entry.input_tokens + entry.output_tokens for entry in sent),
        seconds=seconds,
        review=review,
    )


def find_problems(first: RunCount, second: RunCount) -> tuple[str, ...]:
    """Compare the two runs against what the check expects.

    Args:
        first (RunCount): The run that should reach the model.
        second (RunCount): The run that should be answered from the cache.

    Returns:
        tuple[str, ...]: What went wrong. Empty when both runs behaved as expected.
    """
    problems: list[str] = []
    if first.sent == 0:
        problems.append("run 1 sent no request, so it never reached the model")
    if second.sent > 0:
        problems.append(f"run 2 sent {second.sent} request(s) instead of using the cache")
    if second.cached != first.calls:
        problems.append(f"run 2 took {second.cached} of {first.calls} replies from the cache")
    if second.review != first.review:
        problems.append("run 2 wrote a different review than run 1")
    return tuple(problems)


def check_demo(
    client: LLMClient,
    usage: CountingUsageLog,
    checkpointer: BaseCheckpointSaver[Any],
    sample: int,
) -> CheckResult:
    """Run the demo graph twice with one model and check that the second run is fully cached.

    Each run gets its own new thread, so the second one asks every question again instead
    of being resumed from the checkpointer.

    Args:
        client (LLMClient): The model to check. It must record its usage to `usage`.
        usage (CountingUsageLog): The log the client records to, used to count requests.
        checkpointer (BaseCheckpointSaver[Any]): Where the runs' progress is saved.
        sample (int): Which answer both runs ask for. Pick one never used before, so run 1
            has to reach the model.

    Returns:
        CheckResult: The counts of both runs and any problems.

    Raises:
        openai.APIError, LimitError, LLMError: If a model call fails.
    """
    counts: list[RunCount] = []
    for run in (1, 2):
        start = len(usage.entries)
        started = time.perf_counter()
        thread_id = f"check-{client.name}-s{sample}-run{run}"
        result = run_demo(client, checkpointer, thread_id, sample=sample)
        seconds = time.perf_counter() - started
        counts.append(count_run(usage.entries[start:], seconds, result.review))
    first, second = counts
    return CheckResult(client.name, first, second, find_problems(first, second))
