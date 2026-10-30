"""LangSmith tracing: each run as a timeline of graph steps and model calls, on smith.langchain.com.

Tracing is on when .env sets LANGSMITH_TRACING=true and LANGSMITH_API_KEY. LANGSMITH_PROJECT
names the LangSmith project the traces go to. LangGraph then reports each graph run and node
by itself. Our model calls go through the OpenAI SDK directly, which LangSmith can't see, so
LLMClient wraps each call in `trace_model_call`. That step holds the messages, the reply, the
model, the token counts and whether the reply came from the cache, never an API key.

Traces upload in a background thread. If LangSmith is down, the run goes on and only its trace
is lost. Imported by other modules; the file isn't run directly.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import langsmith
from langsmith.run_helpers import get_tracing_context
from langsmith.run_trees import RunTree, get_cached_client
from langsmith.utils import get_tracer_project, tracing_is_enabled

if TYPE_CHECKING:
    # Only for type hints: llm.py imports this file, so a normal import would be circular.
    from pr_reviewer.core.llm import LLMRequest, LLMResponse

log = logging.getLogger(__name__)

# Longest wait, in seconds, for the last traces to upload when a command ends.
FLUSH_TIMEOUT_S = 10.0


def tracing_on() -> bool:
    """Tell whether steps are being traced to LangSmith right now.

    Returns:
        bool: True when tracing is on, from .env or from a tracing_context block in tests.
    """
    return tracing_is_enabled() is True


def start_tracing() -> None:
    """Check the tracing settings once, when a command starts.

    Tracing turned on without an API key would only produce upload errors, so it is turned
    off with a warning.
    """
    if not tracing_on():
        return
    key = os.environ.get("LANGSMITH_API_KEY", "") or os.environ.get("LANGCHAIN_API_KEY", "")
    if not key.strip():
        langsmith.configure(enabled=False)
        log.warning("Tracing is off: LANGSMITH_TRACING is true, but LANGSMITH_API_KEY is empty.")


def describe_tracing() -> str:
    """Describe the tracing state in a few words, for the start of a run.

    Returns:
        str: Such as "on (LangSmith project pr-reviewer)", or "off".
    """
    return f"on (LangSmith project {get_tracer_project()})" if tracing_on() else "off"


def flush_traces() -> None:
    """Wait until this command's traces are uploaded, for at most FLUSH_TIMEOUT_S.

    The upload thread could otherwise be cut off when the command ends. A failed upload is
    reported as a warning and never fails the command.
    """
    if not tracing_on():
        return
    client = get_tracing_context()["client"] or get_cached_client()
    try:
        client.flush(timeout=FLUSH_TIMEOUT_S)
    except Exception as exc:
        log.warning("Couldn't upload all traces to LangSmith: %s", exc)


class ModelCallTrace:
    """The traced step of one model call, filled in once the reply is known."""

    def __init__(self, run: RunTree | None) -> None:
        """Wrap the step.

        Args:
            run (RunTree | None): The LangSmith step. None when tracing is off.
        """
        self._run = run

    def record(self, response: "LLMResponse") -> None:
        """Add the reply, its token counts and where it came from to the step.

        Args:
            response (LLMResponse): The model's reply.
        """
        if self._run is None:
            return
        # The OpenAI reply shape, which LangSmith shows as a chat message.
        outputs: dict[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": response.finish_reason,
                }
            ]
        }
        if not response.cached:
            # A reply from the cache used no tokens this time, so it reports none.
            outputs["usage_metadata"] = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
            }
        self._run.metadata.update(
            {
                "cached": response.cached,
                "attempts": response.attempts,
                "waited_s": round(response.waited_s, 1),
            }
        )
        self._run.end(outputs=outputs)


@contextmanager
def trace_task(
    name: str, inputs: dict[str, Any], metadata: dict[str, Any] | None = None
) -> Iterator[None]:
    """Group the model calls made inside the block under one step, so a long task is one trace.

    For work that isn't a LangGraph run, such as scoring a set of reviews: without it, each of
    its model calls would be a separate trace. With tracing off this does nothing.

    Args:
        name (str): The step's name, such as "score no-comments-baseline".
        inputs (dict[str, Any]): What the task works on, shown as the step's input.
        metadata (dict[str, Any] | None): Extra facts to show with the step.

    Yields:
        None: Run the task inside the block.
    """
    if not tracing_on():
        yield
        return
    with langsmith.trace(name, run_type="chain", inputs=inputs, metadata=metadata or {}):
        yield


@contextmanager
def trace_model_call(profile: str, request: "LLMRequest") -> Iterator[ModelCallTrace]:
    """Record one model call as a step of type "llm", inside the current trace if there is one.

    With tracing off this does nothing. A call that raises is recorded with its error.

    Args:
        profile (str): Model profile name, used as the step's name.
        request (LLMRequest): What the call asks: the messages and the sampling settings.

    Yields:
        ModelCallTrace: Call its record(response) once the reply is known.
    """
    if not tracing_on():
        yield ModelCallTrace(None)
        return
    # The ls_ keys are the names LangSmith reads to show the provider, model and settings.
    metadata = {
        "ls_provider": request.provider,
        "ls_model_name": request.model,
        "ls_temperature": request.temperature,
        "ls_max_tokens": request.max_tokens,
        "profile": profile,
        "sample": request.sample,
    }
    inputs = {"messages": list(request.messages)}
    with langsmith.trace(profile, run_type="llm", inputs=inputs, metadata=metadata) as run:
        yield ModelCallTrace(run)
