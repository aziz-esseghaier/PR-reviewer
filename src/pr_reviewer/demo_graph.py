"""Demo LangGraph graph: a tiny reviewer that looks for one problem in a built-in function.

It uses the LangGraph pieces the real reviewer will need:
- typed state (DemoState) that flows from node to node, saved after each one;
- nodes: find_problem asks the model for one Claim as JSON, verify asks the same model to
  refute it, report writes the review text, give_up ends a run with no usable claim;
- a conditional edge: a reply that isn't a valid Claim loops back to find_problem, up to
  MAX_TRIES times, with the reason added to the conversation;
- a checkpointer, so a run that stops (daily limit, crash, Ctrl+C) resumes where it stopped.

Every model call goes through LLMClient, so the cache, retries, limits and usage log all apply.
A run uses one model profile from start to end.

How to run (from the project folder, with the database up and migrated):
    pr-reviewer demo <profile>                  start a new run
    pr-reviewer demo <profile> --thread <id>    resume a stopped run, or show a finished one
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.llm import ChatMessage, LLMClient, LLMResponse
from pr_reviewer.core.schemas import (
    CATEGORY_NAMES,
    NO_ISSUES,
    Claim,
    Finding,
    Tier,
    VerifierEvidence,
    render_review,
)

# Replies asked of find_problem before the run gives up.
MAX_TRIES = 3

# The pull request the demo reviews: one new file. An empty cart divides by zero on line 6.
DEMO_CODE = '''The pull request adds this file. Line numbers are on the left.

src/shop/cart.py
1  def average_price(items):
2      """Return the average price of the items in the cart."""
3      total = 0
4      for item in items:
5          total += item["price"]
6      return total / len(items)
'''

_CATEGORY_LIST = ", ".join(f"{code} {name}" for code, name in CATEGORY_NAMES.items())

FIND_PROMPT = (
    "You review Python pull requests. Report the single most serious problem the change "
    "introduces. Reply with only a JSON object with these keys: "
    '"file", "start_line", "end_line", '
    f'"category" (the code only, one of: {_CATEGORY_LIST}), '
    '"summary" (one sentence), "details" (one or two sentences), '
    '"expected_failure" (the Python exception a test would raise, such as "KeyError", or null), '
    '"confidence" (a number from 0 to 1).'
)

VERIFY_PROMPT = (
    "You check claims made by a code reviewer. Try hard to show the claim is wrong: the code "
    "may already handle the case, or the case may be impossible. Reply with only a JSON "
    'object with these keys: "verdict" ("upheld" if the claim survives, "refuted" if you '
    'showed it is wrong) and "argument" (one or two sentences).'
)


class Verdict(BaseModel):
    """The verifier's answer, as the model writes it.

    Attributes:
        verdict (Literal["upheld", "refuted"]): Whether the claim survived.
        argument (str): The verifier's reason.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    verdict: Literal["upheld", "refuted"]
    argument: str = Field(min_length=1)

    @field_validator("verdict", mode="before")
    @classmethod
    def _lowercase(cls, value: Any) -> Any:
        """Accept "Upheld" or " REFUTED " as well.

        Args:
            value (Any): The verdict as written.

        Returns:
            Any: The verdict in lowercase, if it's text.
        """
        return value.strip().lower() if isinstance(value, str) else value


def parse_json_reply[M: BaseModel](text: str, model: type[M]) -> M:
    """Read the JSON object in a model reply and check it against a pydantic model.

    Models often wrap JSON in a ```json fence or add a sentence around it, so this reads from
    the first "{" to the last "}".

    Args:
        text (str): The reply text.
        model (type[M]): The pydantic model the JSON must match, such as Claim.

    Returns:
        M: The checked object.

    Raises:
        ValueError: If there's no JSON object, or it doesn't match. The message is short
            enough to show the model on the next try.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("the reply has no JSON object")
    try:
        return model.model_validate_json(text[start : end + 1])
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or 'reply'}: {error['msg']}"
            for error in exc.errors()
        ]
        raise ValueError("; ".join(problems)) from None


class DemoState(TypedDict):
    """Everything the demo graph knows during a run. The checkpointer saves it after each node.

    Attributes:
        profile (str): Model profile of the run. A thread is never resumed with another one.
        code (str): The pull request to review.
        messages (list[ChatMessage]): The conversation with the reviewer so far. A rejected
            reply stays in it with the reason, so the next try is a new request and not a
            cache hit on the same bad reply.
        tries (int): Replies asked of find_problem so far.
        finding (Finding | None): The claim and its evidence, once a valid claim came back.
        review (str): The review text, set by the last node.
        log (list[str]): One line per node run. Nodes return new lines and LangGraph appends
            them to the list (operator.add) instead of replacing it.
    """

    profile: str
    code: str
    messages: list[ChatMessage]
    tries: int
    finding: Finding | None
    review: str
    log: Annotated[list[str], operator.add]


# The pydantic classes kept in DemoState. The checkpointer rebuilds only these from the database.
STATE_TYPES: tuple[type[BaseModel], ...] = (Finding,)

DemoGraph = CompiledStateGraph[DemoState, None, DemoState, DemoState]


def _source(response: LLMResponse) -> str:
    """Say where a reply came from, for the log.

    Args:
        response (LLMResponse): The reply.

    Returns:
        str: "from cache" or "asked the model".
    """
    return "from cache" if response.cached else "asked the model"


def _describe_tier(tier: Tier) -> str:
    """Name a tier for the log, such as "tier 3 (verified)".

    Args:
        tier (Tier): The tier.

    Returns:
        str: Its number and name.
    """
    return f"tier {tier.value} ({tier.name.lower().replace('_', ' ')})"


def after_find_problem(state: DemoState) -> Literal["verify", "find_problem", "give_up"]:
    """Pick the node after find_problem: the conditional edge.

    Args:
        state (DemoState): The state after find_problem ran.

    Returns:
        Literal["verify", "find_problem", "give_up"]: verify when a valid claim came back,
        find_problem to try again, or give_up after MAX_TRIES bad replies.
    """
    if state["finding"] is not None:
        return "verify"
    return "find_problem" if state["tries"] < MAX_TRIES else "give_up"


def report(state: DemoState) -> dict[str, Any]:
    """Write the review text. Only a claim with evidence becomes a comment.

    Args:
        state (DemoState): The state after verify.

    Returns:
        dict[str, Any]: The review text and a log line.
    """
    finding = state["finding"]
    tier = finding.tier if finding is not None else Tier.NONE
    published = [finding] if finding is not None and tier < Tier.NONE else []
    return {
        "review": render_review(published),
        "log": [f"report: {_describe_tier(tier)}, {len(published)} comment(s)"],
    }


def give_up(state: DemoState) -> dict[str, Any]:
    """End a run in which the model never gave a valid claim.

    Args:
        state (DemoState): The state after the last try.

    Returns:
        dict[str, Any]: An empty review and a log line.
    """
    return {"review": NO_ISSUES, "log": [f"give_up: no valid claim in {state['tries']} tries"]}


def build_demo_graph(
    client: LLMClient, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> DemoGraph:
    """Wire the demo nodes and edges into a runnable graph.

    Args:
        client (LLMClient): The model every node asks. One client means one model per run.
        checkpointer (BaseCheckpointSaver[Any] | None): Where to save progress after each
            node. None saves nothing, so a stopped run starts over.

    Returns:
        DemoGraph: The compiled graph.
    """

    def find_problem(state: DemoState) -> dict[str, Any]:
        """Ask the model for one claim, and check the reply against the Claim schema.

        Args:
            state (DemoState): The current state.

        Returns:
            dict[str, Any]: The fields to change: the conversation, the try count, a log
            line, and the finding when the reply was valid.
        """
        messages = state["messages"] or [
            ChatMessage(role="system", content=FIND_PROMPT),
            ChatMessage(role="user", content=state["code"]),
        ]
        tries = state["tries"] + 1
        response = client.complete(messages)
        try:
            claim = parse_json_reply(response.text, Claim)
        except ValueError as exc:
            retry = [
                ChatMessage(role="assistant", content=response.text),
                ChatMessage(
                    role="user",
                    content=f"That reply was rejected ({exc}). Reply again with only the JSON.",
                ),
            ]
            return {
                "messages": [*messages, *retry],
                "tries": tries,
                "log": [f"find_problem: try {tries}, {_source(response)}, rejected: {exc}"],
            }
        return {
            "messages": messages,
            "tries": tries,
            "finding": Finding(id="c1", claim=claim),
            "log": [f"find_problem: try {tries}, {_source(response)}, claim accepted"],
        }

    def verify(state: DemoState) -> dict[str, Any]:
        """Ask the same model to refute the claim, and add its verdict as evidence.

        Args:
            state (DemoState): The state with a valid claim.

        Returns:
            dict[str, Any]: The finding with the verdict added, and a log line. An unreadable
            verdict adds no evidence, so the claim stays unproven.
        """
        finding = state["finding"]
        if finding is None:
            raise ValueError("verify needs a claim; the edge after find_problem should ensure it")
        claim = finding.claim
        question = (
            f"{state['code']}\nClaim about {claim.file}, lines {claim.start_line}-"
            f"{claim.end_line}: {claim.summary} {claim.details}"
        )
        response = client.complete(
            [
                ChatMessage(role="system", content=VERIFY_PROMPT),
                ChatMessage(role="user", content=question),
            ]
        )
        try:
            verdict = parse_json_reply(response.text, Verdict)
        except ValueError as exc:
            return {"log": [f"verify: {_source(response)}, unreadable reply ({exc})"]}
        evidence = VerifierEvidence(
            reasoning=claim.details or claim.summary,
            verdict=verdict.verdict,
            argument=verdict.argument,
        )
        return {
            "finding": finding.with_evidence(evidence),
            "log": [f"verify: {_source(response)}, verdict {verdict.verdict}"],
        }

    builder = StateGraph(DemoState)
    builder.add_node("find_problem", find_problem)
    builder.add_node("verify", verify)
    builder.add_node("report", report)
    builder.add_node("give_up", give_up)
    builder.add_edge(START, "find_problem")
    builder.add_conditional_edges(
        "find_problem", after_find_problem, ["verify", "find_problem", "give_up"]
    )
    builder.add_edge("verify", "report")
    builder.add_edge("report", END)
    builder.add_edge("give_up", END)
    return builder.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class DemoResult:
    """How a demo run ended.

    Attributes:
        thread_id (str): Thread the run's progress is saved under.
        finding (Finding | None): The claim and its evidence. None if no valid claim came back.
        review (str): The review text SWR-Bench would score.
        log (list[str]): One line per node run, over all runs of the thread.
        resumed (bool): True if this run continued a stopped one.
        already_finished (bool): True if the thread had finished before, so nothing ran.
    """

    thread_id: str
    finding: Finding | None
    review: str
    log: list[str]
    resumed: bool
    already_finished: bool


def new_thread_id(profile: str) -> str:
    """Make a thread ID for a new run, from the profile and the current time.

    Args:
        profile (str): The run's model profile.

    Returns:
        str: Such as "demo-groq-gpt-oss-120b-20261029-171500".
    """
    return f"demo-{profile}-{datetime.now():%Y%m%d-%H%M%S}"


def run_demo(
    client: LLMClient,
    checkpointer: BaseCheckpointSaver[Any],
    thread_id: str,
    on_step: Callable[[str], None] | None = None,
) -> DemoResult:
    """Run the demo graph on a thread: start it, resume it, or show how it ended.

    Args:
        client (LLMClient): The model to use.
        checkpointer (BaseCheckpointSaver[Any]): Where the thread's progress is saved.
        thread_id (str): The thread. A new ID starts a new run.
        on_step (Callable[[str], None] | None): Called with each log line as nodes finish,
            for live progress. None prints nothing.

    Returns:
        DemoResult: The finding, the review text and the log.

    Raises:
        ConfigError: If the thread was started with another model profile.
        openai.APIError, LimitError, LLMError: If a model call fails. The steps that
            finished stay saved, so running the same thread again resumes after them.
    """
    show = on_step or (lambda line: None)
    graph = build_demo_graph(client, checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    saved = graph.get_state(config)
    start: DemoState | None
    if saved.values:
        if saved.values["profile"] != client.name:
            raise ConfigError(
                f"Thread {thread_id} was run with {saved.values['profile']}. Resume it with "
                "that profile, or start a new thread."
            )
        if not saved.next:
            return _result(thread_id, saved.values, resumed=False, already_finished=True)
        show(f"(resuming at {', '.join(saved.next)})")
        start = None  # None tells LangGraph to continue from the saved state
    else:
        start = DemoState(
            profile=client.name,
            code=DEMO_CODE,
            messages=[],
            tries=0,
            finding=None,
            review="",
            log=[],
        )
    for update in graph.stream(start, config, stream_mode="updates"):
        for changes in update.values():
            for line in (changes or {}).get("log", []):
                show(line)
    final = graph.get_state(config).values
    return _result(thread_id, final, resumed=start is None, already_finished=False)


def _result(
    thread_id: str, values: dict[str, Any], *, resumed: bool, already_finished: bool
) -> DemoResult:
    """Build a DemoResult from a thread's saved state.

    Args:
        thread_id (str): The thread.
        values (dict[str, Any]): The thread's state.
        resumed (bool): Whether this run continued a stopped one.
        already_finished (bool): Whether the thread had finished before this call.

    Returns:
        DemoResult: The result.
    """
    return DemoResult(
        thread_id=thread_id,
        finding=values.get("finding"),
        review=values.get("review", ""),
        log=list(values.get("log", [])),
        resumed=resumed,
        already_finished=already_finished,
    )
