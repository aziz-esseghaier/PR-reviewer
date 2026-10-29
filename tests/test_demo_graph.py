"""Tests for the demo LangGraph graph in demo_graph.py and the checkpointer in db/checkpoints.py.

Runs the graph with a scripted fake model, so no real provider is called. Checks the normal
path, the retry loop on bad JSON, giving up, a refuted claim, resuming a stopped run without
asking the finished steps again, refusing to switch models on a thread, and a second run on a
new thread answered from the cache. Most tests keep checkpoints in memory; the ones marked
`integration` use PostgreSQL in a throwaway database and are skipped when it isn't running.

How to run (from the project folder):
    pytest tests/test_demo_graph.py
"""

import json
from collections.abc import Iterator

import openai
import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.conninfo import conninfo_to_dict

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.llm import ResponseCache
from pr_reviewer.core.schemas import NO_ISSUES, Claim, Finding, Tier
from pr_reviewer.db.checkpoints import checkpoint_serializer, open_checkpointer, setup_checkpoints
from pr_reviewer.db.migrate import migrate
from pr_reviewer.demo_graph import (
    STATE_TYPES,
    VERIFY_PROMPT,
    DemoResult,
    Verdict,
    parse_json_reply,
    run_demo,
)
from tests.fakes import MemoryCache, ScriptedProvider, fake_client

CLAIM_JSON = json.dumps(
    {
        "file": "src/shop/cart.py",
        "start_line": 6,
        "end_line": 6,
        "category": "F.4",
        "summary": "average_price divides by zero when the cart is empty.",
        "details": "len(items) is 0 for an empty cart.",
        "expected_failure": "ZeroDivisionError",
        "confidence": 0.9,
    }
)
UPHELD = '{"verdict": "upheld", "argument": "Nothing checks for an empty list."}'
REFUTED = '{"verdict": "refuted", "argument": "Callers never pass an empty cart."}'


def memory_saver() -> InMemorySaver:
    """Make an in-memory checkpointer that stores state the same way the PostgreSQL one does.

    Returns:
        InMemorySaver: The checkpointer.
    """
    return InMemorySaver(serde=checkpoint_serializer(STATE_TYPES))


def run(
    provider: ScriptedProvider,
    saver: InMemorySaver | None = None,
    thread: str = "t1",
    cache: ResponseCache | None = None,
    profile: str = "fake-small",
) -> DemoResult:
    """Run the demo graph against a scripted fake model.

    Args:
        provider (ScriptedProvider): The fake model's replies.
        saver (InMemorySaver | None): Checkpointer to use. None makes a new one.
        thread (str): Thread ID.
        cache (ResponseCache | None): Response cache. None turns caching off.
        profile (str): Fake model profile.

    Returns:
        DemoResult: How the run ended.
    """
    client = fake_client(provider, cache=cache, profile=profile)
    return run_demo(client, saver or memory_saver(), thread)


def test_valid_claim_is_verified_and_reported() -> None:
    """Check the normal path: a claim (in a code fence), an upheld verdict, one comment."""
    provider = ScriptedProvider([f"```json\n{CLAIM_JSON}\n```", UPHELD])
    result = run(provider)
    assert isinstance(result.finding, Finding)  # rebuilt as an object from the checkpoint
    assert result.finding.tier is Tier.VERIFIED
    assert result.review.startswith("1. src/shop/cart.py, line 6: average_price divides")
    assert len(provider.requests) == 2
    assert [line.split(":")[0] for line in result.log] == ["find_problem", "verify", "report"]


def test_bad_reply_is_retried_with_the_reason() -> None:
    """Check the loop: a reply without JSON goes back to find_problem with the rejection."""
    provider = ScriptedProvider(["The bug is a division by zero.", CLAIM_JSON, UPHELD])
    result = run(provider)
    assert result.finding is not None
    assert len(provider.requests) == 3
    retry = provider.messages(1)
    assert retry[-2] == {"role": "assistant", "content": "The bug is a division by zero."}
    assert "rejected (the reply has no JSON object)" in retry[-1]["content"]


def test_gives_up_after_three_bad_replies() -> None:
    """Check that the run ends with no comment after MAX_TRIES invalid replies."""
    provider = ScriptedProvider(["no", "still no", '{"file": "/etc/passwd"}'])
    result = run(provider)
    assert result.finding is None
    assert result.review == NO_ISSUES
    assert len(provider.requests) == 3
    assert result.log[-1] == "give_up: no valid claim in 3 tries"


def test_refuted_claim_is_not_reported() -> None:
    """Check that a claim the verifier refutes has no evidence and makes no comment."""
    result = run(ScriptedProvider([CLAIM_JSON, REFUTED]))
    assert result.finding is not None
    assert result.finding.tier is Tier.NONE
    assert result.review == NO_ISSUES


def test_unreadable_verdict_leaves_the_claim_unproven() -> None:
    """Check that a verifier reply that isn't valid JSON adds no evidence."""
    result = run(ScriptedProvider([CLAIM_JSON, "I agree with the reviewer."]))
    assert result.finding is not None
    assert result.finding.evidence == ()
    assert "unreadable reply" in result.log[1]


def test_stopped_run_resumes_after_the_last_finished_step() -> None:
    """Check that a run stopped in verify continues there, without asking find_problem again.

    The cache is off, so only the checkpointer can explain why the claim isn't asked again.
    """
    saver = memory_saver()
    with pytest.raises(openai.AuthenticationError):
        run(ScriptedProvider([CLAIM_JSON, 401]), saver)
    second = ScriptedProvider([UPHELD])
    result = run(second, saver)
    assert result.resumed
    assert len(second.requests) == 1
    assert second.messages(0)[0]["content"] == VERIFY_PROMPT
    assert result.finding is not None
    assert result.finding.tier is Tier.VERIFIED


def test_finished_thread_is_not_run_again() -> None:
    """Check that running a finished thread again sends nothing and shows the saved result."""
    saver = memory_saver()
    first = run(ScriptedProvider([CLAIM_JSON, UPHELD]), saver)
    again = ScriptedProvider([])
    result = run(again, saver)
    assert result.already_finished
    assert again.requests == []
    assert result.review == first.review


def test_thread_keeps_its_model() -> None:
    """Check that a thread can't be continued with a different model profile."""
    saver = memory_saver()
    with pytest.raises(openai.AuthenticationError):
        run(ScriptedProvider([401]), saver)
    with pytest.raises(ConfigError, match="was run with fake-small"):
        run(ScriptedProvider([]), saver, profile="fake-limited")


def test_new_thread_is_answered_from_the_cache() -> None:
    """Check that a second run on a new thread asks the same questions and gets cached replies."""
    cache = MemoryCache()
    first = run(ScriptedProvider([CLAIM_JSON, UPHELD]), thread="t1", cache=cache)
    offline = ScriptedProvider([])
    second = run(offline, thread="t2", cache=cache)
    assert offline.requests == []
    assert second.review == first.review
    assert all("from cache" in line for line in second.log[:2])


@pytest.mark.parametrize(
    "text",
    [
        CLAIM_JSON,
        f"```json\n{CLAIM_JSON}\n```",
        f"Here is the problem I found:\n{CLAIM_JSON}\nHope this helps.",
    ],
)
def test_parse_json_reply_finds_the_object(text: str) -> None:
    """Check that JSON is read whether it's bare, fenced or surrounded by text."""
    assert parse_json_reply(text, Claim).start_line == 6


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("No problems found.", "no JSON object"),
        ('{"file": "src/shop/cart.py", "start_line": 6', "no JSON object"),
        ('{"verdict": "maybe", "argument": "unsure"}', "verdict: Input should be"),
    ],
)
def test_parse_json_reply_explains_rejections(text: str, message: str) -> None:
    """Check that a rejected reply gets a short reason the model can act on."""
    with pytest.raises(ValueError, match=message):
        parse_json_reply(text, Verdict)


def test_verdict_accepts_any_capitalization() -> None:
    """Check that "Upheld" counts as "upheld"."""
    verdict = parse_json_reply('{"verdict": " Upheld ", "argument": "ok"}', Verdict)
    assert verdict.verdict == "upheld"


@pytest.fixture
def checkpoint_db(test_db_url: str) -> Iterator[str]:
    """Prepare a throwaway database with the project tables and the checkpoint tables.

    Args:
        test_db_url (str): Connection string of the empty test database.

    Yields:
        str: The test database's name, to pass to open_checkpointer.
    """
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        migrate(conn)
    dbname = str(conninfo_to_dict(test_db_url)["dbname"])
    setup_checkpoints(dbname)
    yield dbname


@pytest.mark.integration
def test_postgres_checkpointer_resumes_across_connections(checkpoint_db: str) -> None:
    """Check resuming with checkpoints in PostgreSQL, each run on its own connection."""
    with open_checkpointer(STATE_TYPES, dbname=checkpoint_db) as saver:
        client = fake_client(ScriptedProvider([CLAIM_JSON, 401]))
        with pytest.raises(openai.AuthenticationError):
            run_demo(client, saver, "pg-thread")
    second = ScriptedProvider([UPHELD])
    with open_checkpointer(STATE_TYPES, dbname=checkpoint_db) as saver:
        result = run_demo(fake_client(second), saver, "pg-thread")
    assert result.resumed
    assert len(second.requests) == 1
    assert isinstance(result.finding, Finding)
    assert result.finding.tier is Tier.VERIFIED


@pytest.mark.integration
def test_checkpointer_needs_the_migration(test_db_url: str) -> None:
    """Check that using checkpoints before `db migrate` explains what to run."""
    dbname = str(conninfo_to_dict(test_db_url)["dbname"])
    with pytest.raises(ConfigError, match="db migrate"), open_checkpointer(dbname=dbname):
        pass
