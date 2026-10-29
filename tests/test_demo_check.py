"""Tests for the end-to-end demo check in demo_check.py.

Runs the check against scripted fake models, so no real provider is called. Checks that it
passes when the second run is fully cached, and that it fails when it should: with no cache,
or when the first run was already cached and so never reached the model. Also checks that
retries count as requests sent but not as extra calls.

How to run (from the project folder):
    pytest tests/test_demo_check.py
"""

from langgraph.checkpoint.memory import InMemorySaver

from pr_reviewer.core.llm import ResponseCache
from pr_reviewer.db.checkpoints import checkpoint_serializer
from pr_reviewer.demo_check import CheckResult, CountingUsageLog, check_demo
from pr_reviewer.demo_graph import STATE_TYPES
from tests.fakes import MemoryCache, ScriptedProvider, fake_client
from tests.test_demo_graph import CLAIM_JSON, UPHELD


def check(
    provider: ScriptedProvider,
    cache: ResponseCache | None,
    saver: InMemorySaver | None = None,
    sample: int = 7,
) -> CheckResult:
    """Run the check once with a fake model.

    Args:
        provider (ScriptedProvider): The fake model's replies.
        cache (ResponseCache | None): Response cache. None turns caching off.
        saver (InMemorySaver | None): Checkpointer. None makes a new one.
        sample (int): Sample number for both runs.

    Returns:
        CheckResult: The check's outcome.
    """
    usage = CountingUsageLog()
    client = fake_client(provider, cache=cache, usage=usage)
    checkpointer = saver or InMemorySaver(serde=checkpoint_serializer(STATE_TYPES))
    return check_demo(client, usage, checkpointer, sample)


def test_passes_when_the_second_run_is_cached() -> None:
    """Check the normal case: run 1 asks twice, run 2 gets both replies from the cache."""
    provider = ScriptedProvider([CLAIM_JSON, UPHELD])
    result = check(provider, MemoryCache())
    assert result.passed, result.problems
    assert (result.first.sent, result.first.cached, result.first.calls) == (2, 0, 2)
    assert (result.second.sent, result.second.cached) == (0, 2)
    assert result.first.tokens == 30  # the fake reports 12 + 3 tokens per request
    assert result.second.tokens == 0
    assert len(provider.requests) == 2


def test_fails_without_a_cache() -> None:
    """Check that the check catches a second run that asks the model again."""
    result = check(ScriptedProvider([CLAIM_JSON, UPHELD, CLAIM_JSON, UPHELD]), cache=None)
    assert not result.passed
    assert "run 2 sent 2 request(s) instead of using the cache" in result.problems
    assert "run 2 took 0 of 2 replies from the cache" in result.problems


def test_fails_when_run_1_was_already_cached() -> None:
    """Check that the check catches a first run that never reached the model."""
    cache = MemoryCache()
    assert check(ScriptedProvider([CLAIM_JSON, UPHELD]), cache, sample=3).passed
    # The same sample again, with a new checkpointer so the threads are new: run 1 now finds
    # every reply in the cache.
    again = check(ScriptedProvider([]), cache, sample=3)
    assert "run 1 sent no request, so it never reached the model" in again.problems


def test_retries_count_as_requests_not_calls() -> None:
    """Check that a retried request adds to the requests sent but is still one call."""
    result = check(ScriptedProvider([503, CLAIM_JSON, UPHELD]), MemoryCache())
    assert result.passed, result.problems
    assert (result.first.sent, result.first.calls) == (3, 2)
    assert result.second.cached == 2
