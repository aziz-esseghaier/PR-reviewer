"""Check the project's foundations end to end: the demo LangGraph graph runs with each free model
in configs/models.yaml, and a second identical run is answered entirely from the cache.

For each model profile, runs the demo graph twice on new threads with a fresh sample number
(the current time, so no reply is cached yet). Run 1 must reach the model. Run 2 must send no
request, take every reply from the cache, and write the same review. Prints the counts of each
run and PASS or FAIL per model. Uses a few requests of each model's free daily quota. With
LangSmith tracing on, both runs show there too.

How to run (from the project folder, with the database up and migrated):
    python scripts/check_foundations.py                      every profile in the config
    python scripts/check_foundations.py groq-gpt-oss-120b    only the profiles named
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import openai
import psycopg

from pr_reviewer.core.cache import PostgresCache
from pr_reviewer.core.config import ConfigError, load_env, load_models_config, read_api_key
from pr_reviewer.core.limits import LimitError
from pr_reviewer.core.llm import LLMClient, LLMError, explain_api_error
from pr_reviewer.core.tracing import describe_tracing, flush_traces, start_tracing
from pr_reviewer.core.usage import PostgresUsageLog
from pr_reviewer.db.checkpoints import open_checkpointer
from pr_reviewer.db.connection import connect
from pr_reviewer.demo_check import CheckResult, CountingUsageLog, RunCount, check_demo
from pr_reviewer.demo_graph import STATE_TYPES


def describe_run(number: int, run: RunCount) -> str:
    """Write one line about a run's requests, cache hits, tokens and time.

    Args:
        number (int): 1 or 2.
        run (RunCount): The run's counts.

    Returns:
        str: Such as "run 1: 2 requests sent, 0 from cache, 1,234 tokens, 3.1 s".
    """
    return (
        f"run {number}: {run.sent} request(s) sent, {run.cached} from cache, "
        f"{run.tokens:,} tokens, {run.seconds:.1f} s"
    )


def print_result(result: CheckResult) -> None:
    """Print the counts of both runs and the verdict for one model.

    Args:
        result (CheckResult): The check's outcome.
    """
    print(f"  {describe_run(1, result.first)}")
    print(f"  {describe_run(2, result.second)}")
    if result.passed:
        print("  PASS: run 1 asked the model, run 2 came entirely from the cache")
    for problem in result.problems:
        print(f"  FAIL: {problem}")


def check_all(profiles: Sequence[str], config_path: Path | None) -> int:
    """Run the check for each model profile and print the results.

    Args:
        profiles (Sequence[str]): Profiles to check. Empty means every profile in the config.
        config_path (Path | None): Model config file. None finds configs/models.yaml.

    Returns:
        int: Exit code, 0 when every model passed and 1 otherwise.

    Raises:
        ConfigError: If a profile doesn't exist or its API key isn't set.
    """
    config = load_models_config(config_path)
    names = list(profiles) or list(config.models)
    # Check every name and key first, so a typo doesn't stop the check halfway.
    for name in names:
        read_api_key(config.provider(config.profile(name).provider))
    sample = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    print(f"Checking {len(names)} model(s) with sample {sample} (new, so nothing is cached yet)")
    print(f"Tracing: {describe_tracing()}")
    passed = 0
    with connect() as conn, open_checkpointer(STATE_TYPES) as checkpointer:
        cache = PostgresCache(conn)
        for name in names:
            print(f"\n{name}")
            usage = CountingUsageLog(PostgresUsageLog(conn))
            client = LLMClient(name, config, cache=cache, usage=usage)
            try:
                result = check_demo(client, usage, checkpointer, sample)
            except (openai.APIError, LimitError, LLMError) as exc:
                reason = explain_api_error(exc) if isinstance(exc, openai.APIError) else str(exc)
                print(f"  FAIL: a run stopped: {reason}")
                continue
            print_result(result)
            if result.passed:
                passed += 1
    print(f"\n{passed} of {len(names)} model(s) passed.")
    return 0 if passed == len(names) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, run the check and turn setup problems into short messages.

    Args:
        argv (Sequence[str] | None): Arguments to parse. None means the real command line.

    Returns:
        int: Exit code, 0 when every model passed.
    """
    parser = argparse.ArgumentParser(
        description="Run the demo graph twice per model; the second run must be fully cached."
    )
    parser.add_argument(
        "profiles", nargs="*", help="model profiles to check (default: all in the config)"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="model config file (default: configs/models.yaml)"
    )
    args = parser.parse_args(argv)
    load_env()
    # Show the client's waits and retries as they happen.
    logging.basicConfig(format="%(message)s")
    logging.getLogger("pr_reviewer").setLevel(logging.INFO)
    start_tracing()
    try:
        return check_all(args.profiles, args.config)
    except ConfigError as exc:
        print(f"Setup problem: {exc}", file=sys.stderr)
    except psycopg.OperationalError as exc:
        print(
            f"Can't reach PostgreSQL: {exc}\nStart it with: docker compose up -d",
            file=sys.stderr,
        )
    finally:
        flush_traces()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
