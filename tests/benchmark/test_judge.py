"""
Tests for loading SWR-Bench's scorer (benchmark/official.py) and the judge's run_chat
(benchmark/judge.py).

Loads the official evaluation_struct.py unchanged and runs its evaluate_one() with a scripted
fake judge model, so no real provider is called. Checks that our run_chat is the one the
script uses, that an edited script is refused, that a retry of the same request asks for a new
sample instead of the cached reply, and that failures and wrong settings are handled.

How to run (from the project folder):
    pytest tests/benchmark/test_judge.py
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from pr_reviewer.benchmark.judge import (
    JUDGE_MODEL,
    JUDGE_PROFILE,
    JUDGE_PROVIDER,
    JudgeChat,
    make_judge,
    scorer_settings,
)
from pr_reviewer.benchmark.official import load_evaluation
from pr_reviewer.benchmark.source import OFFICIAL_FILES, DatasetError
from pr_reviewer.core.config import ConfigError, load_models_config
from pr_reviewer.core.llm import ChatMessage
from tests.benchmark.test_dataset import make_row
from tests.fakes import MemoryCache, ScriptedProvider, fake_client

OFFICIAL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "swrbench"
LOG = logging.getLogger("swr-test")

GOOD_CLEAN = json.dumps({"identified_as_good": "YES", "pred_points": []})
GOOD_CHANGE = json.dumps(
    {
        "identified_as_good": "NO",
        "pred_points": [
            {
                "id": "PRED-POINT-1",
                "description": "Divides by zero for an empty cart.",
                "change_category": "F.4 Check",
                "severity_score": 7,
            }
        ],
        "gt_points": [
            {
                "id": "GT-POINT-1",
                "description": "average_price divides by zero for an empty cart.",
                "change_category": "F.2",
                "hit": "YES",
                "hit_by": "PRED-POINT-1",
            }
        ],
    }
)


def item_for(row: dict[str, Any], review: str) -> dict[str, Any]:
    """Shape a dataset record and a review the way SWR-Bench's scorer expects them.

    Args:
        row (dict[str, Any]): The raw dataset record.
        review (str): The review to score.

    Returns:
        dict[str, Any]: {"instance": ..., "pred": {"review": ...}}.
    """
    return {"instance": json.loads(json.dumps(row)), "pred": {"review": review}}


def with_timeline(row: dict[str, Any]) -> dict[str, Any]:
    """Give a built record a review discussion, which the judge prompt for issues needs.

    Args:
        row (dict[str, Any]): A record from make_row().

    Returns:
        dict[str, Any]: The same record with one reviewer comment in its timeline.
    """
    row["pr_timeline"].append(
        {"type": "comment", "user": "reviewer", "body": "Empty cart?", "created_at": "2020-03-02"}
    )
    return row


def test_scorer_uses_our_run_chat_and_scores_a_pr() -> None:
    """Check that evaluate_one() sends SWR-Bench's prompt through our client and reads the hit."""
    provider = ScriptedProvider([GOOD_CHANGE])
    chat = JudgeChat(fake_client(provider, profile="fake-small"))
    swr = load_evaluation(chat)
    result = swr.evaluate_one(
        scorer_settings(chat.client), item_for(with_timeline(make_row(1)), "x"), LOG
    )
    assert result["hit"] == 1 and result["total"] == 1
    body = json.loads(provider.requests[0].content)
    assert "<GroundTruthChanges>" in body["messages"][0]["content"]
    assert body["response_format"]["json_schema"]["name"] == "evaluation"
    assert "utils" not in sys.modules  # the stand-in was removed after loading


def test_retry_of_the_same_request_asks_for_a_new_sample() -> None:
    """Check that a bad cached reply isn't handed back on the scorer's retry."""
    provider = ScriptedProvider(["not json", GOOD_CLEAN])
    chat = JudgeChat(fake_client(provider, cache=MemoryCache()))
    swr = load_evaluation(chat)
    result = swr.evaluate_one(
        scorer_settings(chat.client), item_for(make_row(1, issues=()), "Fine."), LOG
    )
    assert result["identified_as_good"] == "YES"
    assert chat.calls == 2
    assert len(provider.requests) == 2


def test_new_judge_chat_reuses_the_cached_answers() -> None:
    """Check that scoring the same review in a new run takes the judge's replies from the cache."""
    cache = MemoryCache()
    first = JudgeChat(fake_client(ScriptedProvider(["not json", GOOD_CLEAN]), cache=cache))
    load_evaluation(first).evaluate_one(
        scorer_settings(first.client), item_for(make_row(1, issues=()), "Fine."), LOG
    )
    offline = ScriptedProvider([])
    second = JudgeChat(fake_client(offline, cache=cache))
    result = load_evaluation(second).evaluate_one(
        scorer_settings(second.client), item_for(make_row(1, issues=()), "Fine."), LOG
    )
    assert result["identified_as_good"] == "YES"
    assert offline.requests == []
    assert [reply.cached for reply in second.responses] == [True, True]


def test_failed_call_returns_none_so_the_scorer_retries() -> None:
    """Check that a provider error becomes None, SWR-Bench's signal to try again."""
    chat = JudgeChat(fake_client(ScriptedProvider([400, GOOD_CLEAN])))
    swr = load_evaluation(chat)
    result = swr.evaluate_one(
        scorer_settings(chat.client), item_for(make_row(1, issues=()), "Ok."), LOG
    )
    assert result["identified_as_good"] == "YES"
    assert chat.calls == 2


def test_other_settings_than_the_profile_are_refused() -> None:
    """Check that the scorer can't call the judge with another model or temperature."""
    chat = JudgeChat(fake_client(ScriptedProvider([GOOD_CLEAN])))
    messages: list[ChatMessage] = [{"role": "user", "content": "Hi"}]
    settings = scorer_settings(chat.client)
    with pytest.raises(ValueError, match="not the judge profile"):
        chat("other-model", messages, settings.temperature, settings.max_tokens)
    with pytest.raises(ValueError, match="not the judge profile"):
        chat(settings.model, messages, 0.6, settings.max_tokens)
    assert chat.calls == 0


def test_project_config_has_the_fixed_judge() -> None:
    """Check that configs/models.yaml holds the judge profile with the pinned model and settings."""
    config = load_models_config(OFFICIAL_DIR.parents[1] / "configs" / "models.yaml")
    profile = config.profile(JUDGE_PROFILE)
    assert (profile.provider, profile.model) == (JUDGE_PROVIDER, JUDGE_MODEL)
    assert (profile.temperature, profile.max_tokens) == (0.0, 8192)


@pytest.mark.parametrize(
    "change", [{"model": "gemini-3.1-flash-lite"}, {"temperature": 0.6}, {"max_tokens": 4096}]
)
def test_changed_judge_profile_is_refused(change: dict[str, Any]) -> None:
    """Check that make_judge() refuses a judge profile whose model or settings changed."""
    config = load_models_config(OFFICIAL_DIR.parents[1] / "configs" / "models.yaml")
    changed = config.profile(JUDGE_PROFILE).model_copy(update=change)
    edited = config.model_copy(update={"models": {**config.models, JUDGE_PROFILE: changed}})
    with pytest.raises(ConfigError, match="stays the same for the whole project"):
        make_judge(edited)


def test_edited_scorer_is_never_loaded(tmp_path: Path) -> None:
    """Check that a changed evaluation_struct.py is refused before it runs."""
    for file in OFFICIAL_FILES:
        shutil.copyfile(OFFICIAL_DIR / file.name, tmp_path / file.name)
    with (tmp_path / "evaluation_struct.py").open("a", encoding="utf-8") as script:
        script.write("\nprint('edited')\n")
    with pytest.raises(DatasetError, match="evaluation_struct.py differs"):
        load_evaluation(folder=tmp_path)


def test_scorer_without_a_model_refuses_to_call_one() -> None:
    """Check that the default load builds prompts but can't reach a model."""
    swr = load_evaluation()
    prompt, schema = swr.create_clean_pr_prompt(item_for(make_row(1, issues=()), "Fine."))
    assert "Fine." in prompt
    with pytest.raises(RuntimeError, match="no model to call"):
        swr.run_chat("m", [], 0.0, 1, None)
