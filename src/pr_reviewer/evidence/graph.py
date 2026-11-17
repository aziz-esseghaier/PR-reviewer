"""The evidence agent's graphs: one subgraph per candidate, run a few at a time for each PR.

The PR graph:
1. survey: one sandbox reads what every candidate needs (evidence/survey.py);
2. fan out: LangGraph's Send starts the candidate subgraph once per candidate, each with its
   own state; the run's max_concurrency decides how many run at the same time;
3. collect: once all are done, a summary line.

The candidate subgraph, for one claim:
- start: a functional claim that names an exception, in a Python file of a repo whose tests
  run with pytest, goes to the test writer; any other goes straight to the verifier.
- write: the model writes a test (or, after a broken run, fixes it), or once calls the lookup
  tools instead (explore_write answers them). Our code checks the test before it runs.
- run: the differential runner runs the test 3 times at the base and 3 times at the PR, in
  sandboxes with the network off, watching the lines the PR changed; the matcher
  (evidence/matching.py) reads the outcome. A proof goes to the intent check when the PR's diff
  shows signs the change was deliberate (evidence/intent.py), else it's done. A broken test
  goes back to write while the budget lasts. Anything else goes to the verifier.
- intent: the model decides whether the proven change was deliberate; a deliberate one doesn't
  count, and the claim is dropped.
- verify: the model tries to refute the claim, and judges the static warnings on its lines; it
  may call the lookup tools once first (explore_verify).
- finish: the evidence becomes a Finding, whose tier is the strongest supporting evidence, and
  is saved.

The budget per candidate: at most MAX_WRITER_STEPS test writer replies (write, fix or lookup;
one lookup round at most), DOCKER_MINUTES of test runs, and one verifier decision plus one
optional lookup round. A reply that isn't valid JSON, or that the provider refused (see
refusal_note), is asked for again, up to MAX_TRIES times per request, like every other reviewer;
after a refusal or an empty reply, the next tries close the lookups (see ask). A refused reply
that called a tool the request doesn't offer is read anyway when it holds a lookup or a valid
answer (refused_calls).

Model calls go through one lock (SerialClient), so they're made one at a time while another
candidate's tests run in Docker. Every reply and every test run is cached, so a stopped run
that starts a candidate again costs no quota and no Docker time.

Imported by other modules; the file isn't run directly.
"""

import json
import logging
import operator
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from posixpath import basename
from typing import Annotated, Any, Literal, Protocol, TypedDict

import openai
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.candidates import Piece
from pr_reviewer.context.lookups import Lookup, LookupResult, parse_lookups
from pr_reviewer.context.pack import format_piece
from pr_reviewer.core.llm import ChatMessage, LLMResponse, ToolCall
from pr_reviewer.core.schemas import (
    ExecutionEvidence,
    Finding,
    Tier,
    ToolEvidence,
    VerifierEvidence,
)
from pr_reviewer.demo_graph import parse_json_reply
from pr_reviewer.difftest.outcomes import DiffOutcome, SideResult
from pr_reviewer.evidence.inputs import Candidate
from pr_reviewer.evidence.intent import signs
from pr_reviewer.evidence.matching import describe_failure, judge
from pr_reviewer.evidence.prompts import (
    INTENT_FORMAT,
    LOOKUP_TOOLS,
    MAX_LOOKUPS,
    TEST_FORMAT,
    VERIFY_FORMAT,
    IntentReply,
    VerifyInputs,
    VerifyReply,
    WriterInputs,
    WriterReply,
    call_lookups,
    check_test,
    intent_request,
    parse_reply,
    verify_request,
    writer_request,
)
from pr_reviewer.evidence.survey import Materials, PrSurvey, short_names
from pr_reviewer.reviewer.naive import FEEDBACK_CHARS, MAX_TRIES, _format_refused, retry_note
from pr_reviewer.setup_agent.selection import is_test_file
from pr_reviewer.static.warnings import ToolWarning

log = logging.getLogger(__name__)

# Most test writer replies per candidate: a first test, plus fixes or one lookup round.
MAX_WRITER_STEPS = 3

# Characters of a refused reply shown in the log: its start and its end.
REFUSED_HEAD = 900
REFUSED_TAIL = 600

# The note after a refused or empty reply, when the next try closes the lookups (see ask()).
CLOSED_NOTE = "it held no JSON object. Lookups are off now: answer with the JSON"

# The lookup tools' names.
LOOKUP_NAMES = frozenset(tool["function"]["name"] for tool in LOOKUP_TOOLS)

# Most minutes of test runs per candidate; a fix isn't run once they're spent.
DOCKER_MINUTES = 20.0

# Seconds one pytest run of a generated test may take.
RUN_SECONDS = 120

# An exception name ExecutionEvidence accepts.
_EXCEPTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

# How the test writer ended for a candidate.
WriterResult = Literal["not_tried", "proof", "no_proof", "broken", "invalid", "no_reply", "budget"]


class Completer(Protocol):
    """Anything that answers chat requests like LLMClient.

    Attributes:
        name (str): The model profile's name.
    """

    name: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        sample: int = 0,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Get the model's reply.

        Args:
            messages (Sequence[ChatMessage]): The conversation.
            sample (int): Which answer to this exact request.
            response_format (dict[str, Any] | None): The reply format.
            tools (list[dict[str, Any]] | None): Tools the model may call instead.

        Returns:
            LLMResponse: The reply.
        """
        ...


class SerialClient:
    """An LLM client shared by parallel candidates that makes their calls one at a time.

    Groq's tokens-per-minute limit fits about one request a minute, so parallel calls would
    only wait for each other; the lock also keeps the client's database connection to one
    thread at a time.

    Args:
        client (Completer): The client to share.
    """

    def __init__(self, client: Completer) -> None:
        self._client = client
        self._lock = threading.Lock()
        self.name = client.name

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        sample: int = 0,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Get the model's reply, waiting for any other candidate's call to finish first.

        Args:
            messages (Sequence[ChatMessage]): The conversation.
            sample (int): Which answer to this exact request.
            response_format (dict[str, Any] | None): The reply format.
            tools (list[dict[str, Any]] | None): Tools the model may call instead.

        Returns:
            LLMResponse: The reply.
        """
        with self._lock:
            return self._client.complete(
                messages, sample=sample, response_format=response_format, tools=tools
            )


@dataclass(frozen=True)
class GeneratedRun:
    """One run of a generated test at both sides.

    Attributes:
        outcomes (tuple[DiffOutcome, ...]): Each test's outcome (one for a good test).
        minutes (float): Docker time it took; 0 when the result was stored already.
        problems (str): The runs that couldn't run and why, in short; "" for none.
        file (str): Where the test's code was saved, relative to the project.
    """

    outcomes: tuple[DiffOutcome, ...]
    minutes: float
    problems: str
    file: str


# Runs a test: (finding ID, path in the repo, code, attempt number, lines to watch).
RunTest = Callable[[str, str, str, int, Mapping[str, Sequence[int]]], GeneratedRun]

# Answers lookups in the repo at the PR's last commit.
Explorer = Callable[[list[Lookup]], list[LookupResult]]


class EvidenceRecord(BaseModel):
    """How the evidence agent ended for one candidate, as it's saved.

    Attributes:
        instance_id (str): The PR.
        finding_id (str): The candidate.
        sample (int): Which answers its model calls asked for (its earlier failures).
        status (Literal["done", "failed"]): "failed" when a needed reply never came.
        functional (bool): Whether the claim is functional.
        expected_failure (str | None): Its expected exception, cleaned up.
        tier (int): The strongest supporting evidence's tier, 4 for none.
        proof (str | None): "executed" or "new_code" for a tier-1 proof.
        test_result (str): How the test writer ended.
        verdict (str | None): "upheld", "refuted" or "deliberate"; None when not asked.
        tool_confirmed (bool): Whether a static warning was confirmed as the same problem.
        test_path (str | None): The proving test's file, relative to the project.
        calls (int): Model replies asked for.
        input_tokens (int): Prompt tokens.
        output_tokens (int): Reply tokens.
        docker_minutes (float): Time spent running tests.
        evidence (list[dict[str, Any]]): The Finding's evidence, as JSON.
        details (dict[str, Any]): The attempts, the verifier's and intent check's replies,
            the log.
    """

    model_config = ConfigDict(frozen=True)

    instance_id: str
    finding_id: str
    sample: int
    status: Literal["done", "failed"]
    functional: bool
    expected_failure: str | None
    tier: int
    proof: str | None
    test_result: str
    verdict: str | None
    tool_confirmed: bool
    test_path: str | None
    calls: int
    input_tokens: int
    output_tokens: int
    docker_minutes: float
    evidence: list[dict[str, Any]]
    details: dict[str, Any]


@dataclass(frozen=True)
class PrInputs:
    """What the graphs know about one PR before they start.

    Attributes:
        pr (Instance): The PR.
        candidates (Mapping[str, Candidate]): Its candidates, by finding ID.
        warnings (Sequence[ToolWarning]): The static warnings on its changed lines.
        pieces (Sequence[Piece]): Its retrieved context pieces, for callers.
        pytest (bool): Whether its tests run with pytest (the test writer needs it).
        writer_budget (int): Most characters of a test writer request's user message.
        verify_budget (int): The same for the verifier.
        intent_budget (int): The same for the intent check.
    """

    pr: Instance
    candidates: Mapping[str, Candidate]
    warnings: Sequence[ToolWarning]
    pieces: Sequence[Piece]
    pytest: bool
    writer_budget: int
    verify_budget: int
    intent_budget: int


@dataclass(frozen=True)
class Services:
    """What the graphs use to reach the model, Docker and the database.

    Attributes:
        llm (Completer): The model, shared by the PR's candidates (a SerialClient).
        survey (Callable[[], PrSurvey]): Surveys the PR in a sandbox.
        run_test (RunTest): Runs a generated test at both sides.
        explore (Explorer): Answers lookups.
        save (Callable[[EvidenceRecord], None]): Saves a candidate's result.
        show (Callable[[str], None]): Shows a candidate's log line as its node finishes.
    """

    llm: Completer
    survey: Callable[[], PrSurvey]
    run_test: RunTest
    explore: Explorer
    save: Callable[[EvidenceRecord], None]
    show: Callable[[str], None] = lambda line: None


class CandidateState(TypedDict):
    """Everything the candidate subgraph knows about one candidate. Saved after each node.

    Attributes:
        finding_id (str): The candidate.
        sample (int): Which answers its model calls ask for.
        materials (dict[str, Any] | None): Its Materials, from the survey; None without one.
        changed (dict[str, list[int]]): The PR's changed source lines (test files left out).
        new_names (list[str]): The names on the PR's added lines.
        route (str): The next node.
        writer_steps (int): Test writer replies used.
        looked (bool): Whether the test writer used its lookup round.
        pending (list[dict[str, Any]]): Lookups waiting for answers.
        writer_answers (list[dict[str, Any]]): The test writer's lookup answers.
        attempts (list[dict[str, Any]]): Each test it wrote: code, path, and how it ran.
        test_result (str): How the test writer ended (WriterResult).
        docker_minutes (float): Time spent running tests.
        verify_looked (bool): Whether the verifier used its lookup round.
        verify_answers (list[dict[str, Any]]): The verifier's lookup answers.
        verdict (dict[str, Any] | None): The verifier's reply; None until it answers.
        intent (dict[str, Any] | None): The signs, and the intent check's reply.
        failed (str): Why the candidate failed; "" when it didn't.
        calls (int): Model replies asked for.
        input_tokens (int): Prompt tokens.
        output_tokens (int): Reply tokens.
        steps (list[str]): One line per node run; LangGraph appends new lines.
        results (list[dict[str, Any]]): The saved record, handed to the PR graph.
    """

    finding_id: str
    sample: int
    materials: dict[str, Any] | None
    changed: dict[str, list[int]]
    new_names: list[str]
    route: str
    writer_steps: int
    looked: bool
    pending: list[dict[str, Any]]
    writer_answers: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    test_result: str
    docker_minutes: float
    verify_looked: bool
    verify_answers: list[dict[str, Any]]
    verdict: dict[str, Any] | None
    intent: dict[str, Any] | None
    failed: str
    calls: int
    input_tokens: int
    output_tokens: int
    steps: Annotated[list[str], operator.add]
    results: list[dict[str, Any]]


class PrState(TypedDict):
    """Everything the PR graph knows. Saved after each node.

    Attributes:
        pr_id (str): The PR.
        todo (list[list[Any]]): The candidates to check, each [finding ID, sample].
        survey (dict[str, Any] | None): The survey's result; None until made, or on failure.
        problem (str): Why the survey failed; "" when it didn't.
        results (list[dict[str, Any]]): Each checked candidate's record.
        notes (list[str]): Log lines of the PR graph.
    """

    pr_id: str
    todo: list[list[Any]]
    survey: dict[str, Any] | None
    problem: str
    results: Annotated[list[dict[str, Any]], operator.add]
    notes: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class Asked[M: BaseModel]:
    """The result of asking the model one request, with retries for bad replies.

    Attributes:
        reply (M | None): The checked reply; None when every try was rejected, or when the
            model called the lookup tools instead.
        calls (int): Replies asked for.
        input_tokens (int): Prompt tokens.
        output_tokens (int): Reply tokens.
        note (str): What happened, for the log.
        lookups (list[dict[str, Any]] | None): The lookups the model asked for by calling the
            tools, not checked yet; None when it didn't.
    """

    reply: M | None
    calls: int
    input_tokens: int
    output_tokens: int
    note: str
    lookups: list[dict[str, Any]] | None = None


def refusal_note(exc: openai.BadRequestError) -> str:
    """Give the note for a reply the provider refused, or "" when the request itself was bad.

    Groq answers HTTP 400 for two kinds of bad replies. "json_validate_failed" means the output
    broke the JSON schema; with no output at all, gpt-oss ended its turn without the JSON,
    mostly after "asking" for a lookup in its thinking as if it were a tool call.
    "tool_use_failed" means it called a tool for real, though the request offers none (it was
    trained with built-in tools). Both are a bad reply to ask for again, not a bad request.

    Args:
        exc (openai.BadRequestError): The error.

    Returns:
        str: The note for the model's next try, or "" for any other 400.
    """
    if _format_refused(exc):
        body = exc.body if isinstance(exc.body, dict) else {}
        if not body.get("failed_generation"):
            return "it held no JSON object; answer with the JSON"
        return "the reply didn't match the JSON format"
    text = f"{exc.body} {exc}"
    if "tool_use_failed" in text or "model called a tool" in text:
        return "the reply called a tool, but there are none here; answer with the JSON only"
    return ""


def describe_refusal(exc: openai.BadRequestError) -> str:
    """Describe a refused reply for the log: Groq's code and what the model wrote.

    Groq sends the refused text as "failed_generation". Only its start and end are kept, on
    one line, so the log shows why it was refused (a tool call, broken JSON, or thinking that
    ran out of room) without filling up.

    Args:
        exc (openai.BadRequestError): The refusal.

    Returns:
        str: Such as 'tool_use_failed, 120 characters: {"name": "python", ...}'.
    """
    body = exc.body if isinstance(exc.body, dict) else {}
    code = str(body.get("code") or "unknown")
    text = str(body.get("failed_generation") or "")
    shown = text
    if len(text) > REFUSED_HEAD + REFUSED_TAIL:
        shown = f"{text[:REFUSED_HEAD]} [...] {text[-REFUSED_TAIL:]}"
    return f"{code}, {len(text)} characters: {shown!r}"


def refused_calls(exc: openai.BadRequestError) -> list[ToolCall]:
    """Read the tool calls in a reply Groq refused as "tool_use_failed".

    gpt-oss sometimes calls a tool the request doesn't offer: a lookup under another name (such
    as "repo_browser.search_symbol"), or its whole answer as a call to a tool named "json".
    Groq refuses the reply but sends it back as "failed_generation", so it can still be used.

    Args:
        exc (openai.BadRequestError): The refusal.

    Returns:
        list[ToolCall]: The calls; empty for any other refusal, or text that isn't one call or
        a list of calls.
    """
    body = exc.body if isinstance(exc.body, dict) else {}
    if body.get("code") != "tool_use_failed":
        return []
    try:
        found = json.loads(str(body.get("failed_generation") or ""))
    except ValueError:
        return []
    calls = []
    for item in found if isinstance(found, list) else [found]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return []
        arguments = item.get("arguments", {})
        text = arguments if isinstance(arguments, str) else json.dumps(arguments)
        calls.append(ToolCall(name=item["name"], arguments=text))
    return calls


def _answer_in[M: BaseModel](
    calls: Sequence[ToolCall], model: type[M], check: Callable[[M], str] | None
) -> M | None:
    """Give the reply held in a refused call to a tool that isn't a lookup, if it checks out.

    Args:
        calls (Sequence[ToolCall]): The refused reply's calls (refused_calls).
        model (type[M]): The reply's shape.
        check (Callable[[M], str] | None): Further checks, as in ask().

    Returns:
        M | None: The reply; None unless there is one such call and its arguments are a
        reply that passes every check.
    """
    if len(calls) != 1 or _tool(calls[0]) in LOOKUP_NAMES:
        return None
    try:
        reply = parse_json_reply(calls[0].arguments, model)
    except ValueError:
        return None
    return None if check is not None and check(reply) else reply


def _tool(call: ToolCall) -> str:
    """Give a tool call's name without the prefix gpt-oss sometimes adds.

    Args:
        call (ToolCall): The call.

    Returns:
        str: Such as "search_symbol" for "repo_browser.search_symbol".
    """
    return call.name.rsplit(".", 1)[-1]


def ask[M: BaseModel](
    llm: Completer,
    messages: list[ChatMessage],
    response_format: dict[str, Any],
    model: type[M],
    sample: int,
    check: Callable[[M], str] | None = None,
    closed: list[ChatMessage] | None = None,
) -> Asked[M]:
    """Ask the model, and ask again with the reason when its reply doesn't check out.

    When the caller gives a closed request, this request offers the lookups as tools
    (LOOKUP_TOOLS), and a reply that calls them comes back as lookups. Groq takes tools or a
    JSON format, not both, so such a reply's JSON is checked by our code alone. A refused
    reply that called a tool the request doesn't offer is still read (refused_calls): a
    lookup under another name counts as a lookup (only while the tools are offered), and an
    answer sent as a tool's arguments counts as the reply if it checks out. If the provider
    refuses any other reply, or the model ends its turn with neither an answer nor a call,
    the next tries use the closed request: no tools, and the JSON format.

    Args:
        llm (Completer): The model.
        messages (list[ChatMessage]): The request.
        response_format (dict[str, Any]): The reply format.
        model (type[M]): The reply's shape.
        sample (int): Which answer to ask for.
        check (Callable[[M], str] | None): Further checks on a parsed reply, giving the
            problem ("" when fine).
        closed (list[ChatMessage] | None): The same request with lookups closed; None when
            the request offers no lookups.

    Returns:
        Asked[M]: The reply (or None after MAX_TRIES rejected ones, or lookups) and what it
        cost.
    """
    feedback = ""
    calls = tokens_in = tokens_out = 0
    sources = []
    offer = closed is not None
    for _ in range(MAX_TRIES):
        request = list(messages)
        if feedback:
            request.append(ChatMessage(role="user", content=retry_note(feedback)))
        calls += 1
        try:
            if offer:
                response = llm.complete(request, sample=sample, tools=LOOKUP_TOOLS)
            else:
                response = llm.complete(request, sample=sample, response_format=response_format)
        except openai.BadRequestError as exc:
            feedback = refusal_note(exc)
            if not feedback:
                raise
            log.warning("reply refused: %s", describe_refusal(exc))
            sources.append("refused")
            found = refused_calls(exc)
            if offer and found and all(_tool(call) in LOOKUP_NAMES for call in found):
                sources[-1] = "refused: tool call read"
                note = f"{calls} tries ({', '.join(sources)})"
                return Asked(None, calls, tokens_in, tokens_out, note, call_lookups(found))
            answer = _answer_in(found, model, check)
            if answer is not None:
                sources[-1] = "refused: answer read"
                note = f"{calls} tries ({', '.join(sources)})"
                return Asked(answer, calls, tokens_in, tokens_out, note)
            if offer and closed is not None:
                messages, offer, feedback = closed, False, CLOSED_NOTE
                sources[-1] = "refused: lookups closed"
            continue
        tokens_in += response.input_tokens
        tokens_out += response.output_tokens
        sources.append("cache" if response.cached else "model")
        if offer and response.tool_calls:
            note = f"{calls} tries ({', '.join(sources)})"
            lookups = call_lookups(response.tool_calls)
            return Asked(None, calls, tokens_in, tokens_out, note, lookups)
        if offer and closed is not None and not response.text.strip():
            messages, offer, feedback = closed, False, CLOSED_NOTE
            sources[-1] += ": empty, lookups closed"
            continue
        try:
            reply = parse_reply(response, model)
            problem = check(reply) if check is not None else ""
        except ValueError as exc:
            problem = str(exc)
        if not problem:
            return Asked(
                reply, calls, tokens_in, tokens_out, f"{calls} tries ({', '.join(sources)})"
            )
        feedback = problem[:FEEDBACK_CHARS]
    note = f"{calls} tries, all rejected (last: {feedback[:120]})"
    return Asked(None, calls, tokens_in, tokens_out, note)


def _used[M: BaseModel](state: CandidateState, asked: Asked[M]) -> dict[str, Any]:
    """Give the state keys a model request adds to.

    Args:
        state (CandidateState): The state before.
        asked (Asked[M]): The request's result.

    Returns:
        dict[str, Any]: The new totals of calls and tokens.
    """
    return {
        "calls": state["calls"] + asked.calls,
        "input_tokens": state["input_tokens"] + asked.input_tokens,
        "output_tokens": state["output_tokens"] + asked.output_tokens,
    }


def source_changes(changed: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    """Keep the changed lines of source files: test files and conftest.py files are left out.

    A test that runs a line the PR changed in a test helper shows nothing about the code.

    Args:
        changed (Mapping[str, Sequence[int]]): The PR's changed lines, by file.

    Returns:
        dict[str, list[int]]: The same, for source files only.
    """
    return {
        path: list(lines)
        for path, lines in changed.items()
        if not is_test_file(path) and basename(path) != "conftest.py"
    }


def overlapping(warnings: Sequence[ToolWarning], candidate: Candidate) -> list[ToolWarning]:
    """Give the static warnings on a claim's lines.

    Args:
        warnings (Sequence[ToolWarning]): The PR's warnings on changed lines.
        candidate (Candidate): The candidate.

    Returns:
        list[ToolWarning]: Those in the claim's file whose lines touch its range.
    """
    claim = candidate.claim
    lines = range(claim.start_line, claim.end_line + 1)
    return [w for w in warnings if w.file == claim.file and w.touches(lines)]


def callers_text(pieces: Sequence[Piece], function: str) -> str:
    """Write the retrieved callers of a function, then the definitions it calls.

    Args:
        pieces (Sequence[Piece]): The PR's context pieces.
        function (str): The changed function's qualified name; "" for none.

    Returns:
        str: The pieces, formatted as the reviewers saw them; "" when none relate to it.
    """
    if not function:
        return ""
    related = [p for p in pieces if p.kind == "caller" and p.target == function]
    related += [p for p in pieces if p.kind == "called" and p.target == function]
    return "".join(format_piece(piece) for piece in related)


def side_text(side: SideResult, name: str) -> str:
    """Say in one line how a test ended at one side.

    Args:
        side (SideResult): Its verdict there.
        name (str): The side, such as "the base commit".

    Returns:
        str: Such as "passes on the base commit".
    """
    if side.verdict == "pass":
        return f"passes on {name}"
    if side.verdict == "fail":
        return describe_failure(side.failure, name)
    return f"{side.verdict.replace('_', ' ')} on {name} ({side.reason or ', '.join(side.statuses)})"


def _name(value: str | None) -> str | None:
    """Keep an exception name only when ExecutionEvidence accepts it.

    Args:
        value (str | None): The name.

    Returns:
        str | None: It, or None.
    """
    return value if value and _EXCEPTION_NAME.match(value) else None


def build_record(inputs: PrInputs, state: CandidateState, candidate: Candidate) -> EvidenceRecord:
    """Turn a finished candidate's state into its Finding and saved record.

    Args:
        inputs (PrInputs): The PR's inputs.
        state (CandidateState): The candidate's final state.
        candidate (Candidate): The candidate.

    Returns:
        EvidenceRecord: The record, with the Finding's evidence and tier.
    """
    intent = state["intent"] or {}
    deliberate = bool(intent.get("deliberate"))
    evidence: list[Any] = []
    proving: dict[str, Any] | None = None
    for attempt in state["attempts"]:
        if "base_result" not in attempt:
            continue
        is_proof = attempt.get("proof") is not None
        if is_proof:
            proving = attempt
        evidence.append(
            ExecutionEvidence(
                test_file=attempt["path"],
                test_code=attempt["code"],
                base_result=attempt["base_result"],
                pr_result=attempt["pr_result"],
                failure=_name(attempt.get("failure")),
                failure_classes=tuple(c for c in attempt.get("failure_classes", []) if _name(c)),
                caught_error=_name(attempt.get("caught")),
                proof=attempt.get("proof"),
                reason=attempt.get("reason", ""),
                deliberate=deliberate and is_proof,
                log_path=attempt.get("file") or None,
            )
        )
    verdict = state["verdict"]
    judged = {
        int(item["id"]): bool(item["same_problem"]) for item in (verdict or {}).get("warnings", [])
    }
    warnings = overlapping(inputs.warnings, candidate)
    tools: list[ToolEvidence] = [
        warning.evidence(confirmed=judged.get(n, False))
        for n, warning in enumerate(warnings, start=1)
    ]
    evidence.extend(tools)
    if verdict is not None:
        evidence.append(
            VerifierEvidence(
                reasoning=verdict.get("trace") or verdict.get("argument") or "(none given)",
                verdict=verdict["verdict"],
                argument=verdict.get("argument", ""),
            )
        )
    finding = Finding(id=candidate.finding_id, claim=candidate.claim, evidence=tuple(evidence))
    tier = finding.tier
    status: Literal["done", "failed"] = "failed" if state["failed"] else "done"
    verdict_name = "deliberate" if deliberate else (verdict or {}).get("verdict")
    return EvidenceRecord(
        instance_id=inputs.pr.instance_id,
        finding_id=candidate.finding_id,
        sample=state["sample"],
        status=status,
        functional=candidate.claim.is_functional,
        expected_failure=candidate.expected,
        tier=int(tier),
        proof=proving.get("proof") if proving and tier is Tier.EXECUTED else None,
        test_result=state["test_result"],
        verdict=verdict_name,
        tool_confirmed=any(tool.supports(candidate.claim) for tool in tools),
        test_path=proving.get("file") if proving and tier is Tier.EXECUTED else None,
        calls=state["calls"],
        input_tokens=state["input_tokens"],
        output_tokens=state["output_tokens"],
        docker_minutes=round(state["docker_minutes"], 2),
        evidence=[item.model_dump(mode="json") for item in finding.evidence],
        details={
            "attempts": state["attempts"],
            "verdict": verdict,
            "intent": state["intent"],
            "failed": state["failed"],
            "log": state["steps"],
        },
    )


def _answers(items: Sequence[dict[str, Any]]) -> list[LookupResult]:
    """Read saved lookup answers back.

    Args:
        items (Sequence[dict[str, Any]]): The answers, as saved in the state.

    Returns:
        list[LookupResult]: The answers.
    """
    return [LookupResult.model_validate(item) for item in items]


def build_candidate_graph(inputs: PrInputs, services: Services) -> Any:
    """Wire the candidate subgraph's nodes and edges.

    Args:
        inputs (PrInputs): The PR's inputs.
        services (Services): The model, Docker and the database.

    Returns:
        Any: The compiled subgraph, to add to the PR graph as a node. It saves its progress
        with the PR graph's checkpointer.
    """

    def candidate_of(state: CandidateState) -> Candidate:
        return inputs.candidates[state["finding_id"]]

    def materials_of(state: CandidateState) -> Materials | None:
        raw = state["materials"]
        return Materials.model_validate(raw) if raw is not None else None

    def start(state: CandidateState) -> dict[str, Any]:
        """Send a testable claim to the test writer, any other to the verifier."""
        candidate = candidate_of(state)
        materials = materials_of(state)
        claim = candidate.claim
        why = ""
        if not candidate.testable:
            why = "not a functional claim with an expected exception"
        elif not claim.file.endswith(".py"):
            why = "the claim isn't in a Python file"
        elif not inputs.pytest:
            why = "the repo's tests don't run with pytest"
        elif materials is None:
            why = "no survey"
        elif materials.test_path is None:
            why = "no test file to put a test next to"
        fid = state["finding_id"]
        if why:
            return {"route": "verify", "steps": [f"{fid} start: no test ({why})"]}
        return {"route": "write", "steps": [f"{fid} start: test writer ({candidate.expected})"]}

    def write(state: CandidateState) -> dict[str, Any]:
        """Ask for a test, a fix of the last one, or lookups; check the test."""
        candidate = candidate_of(state)
        materials = materials_of(state)
        assert materials is not None and materials.test_path is not None
        expected = candidate.expected or ""
        fid, step = state["finding_id"], state["writer_steps"] + 1
        last = state["attempts"][-1] if state["attempts"] else None
        previous = last["code"] if last else ""
        error = last.get("feedback", "") if last else ""
        may_look = not state["looked"] and step < MAX_WRITER_STEPS
        writer = WriterInputs(
            claim=candidate.claim,
            expected=expected,
            test_path=materials.test_path,
            code=materials.code,
            hunks=materials.hunks,
            example=materials.example,
            fixtures=materials.fixtures,
        )

        def request(look: bool) -> tuple[list[ChatMessage], list[str]]:
            return writer_request(
                writer,
                inputs.writer_budget,
                previous=previous,
                error=error,
                lookups=_answers(state["writer_answers"]),
                may_look=look,
            )

        def check(reply: WriterReply) -> str:
            if not reply.test_code.strip():
                return "the reply has no test_code; write the test now"
            return ""

        messages, trimmed = request(may_look)
        closed = request(False)[0] if may_look else None
        asked = ask(
            services.llm, messages, TEST_FORMAT, WriterReply, state["sample"], check, closed
        )
        used = {**_used(state, asked), "writer_steps": step}
        kind = "fix" if last else "write"
        cut = f", parts cut: {', '.join(trimmed)}" if trimmed else ""
        if asked.lookups is not None:
            lookups = parse_lookups(asked.lookups, MAX_LOOKUPS)
            asked_for = "; ".join(lookup.describe() for lookup in lookups)
            return {
                **used,
                "route": "explore_write",
                "looked": True,
                "pending": [lookup.model_dump() for lookup in lookups],
                "steps": [
                    f"{fid} {kind} {step}: {len(lookups)} lookups ({asked.note}): {asked_for}{cut}"
                ],
            }
        if asked.reply is None:
            return {
                **used,
                "route": "verify",
                "test_result": "no_reply",
                "steps": [f"{fid} {kind} {step}: no usable reply ({asked.note}){cut}"],
            }
        reply = asked.reply
        problem = check_test(reply.test_code, expected)
        attempt = {
            "n": len(state["attempts"]) + 1,
            "path": materials.test_path,
            "code": reply.test_code,
        }
        if problem:
            attempt.update(
                {"kind": "invalid", "feedback": f"Our checks refused the test: {problem}."}
            )
            more = step < MAX_WRITER_STEPS
            return {
                **used,
                "attempts": [*state["attempts"], attempt],
                "route": "write" if more else "verify",
                "test_result": state["test_result"] if more else "invalid",
                "steps": [f"{fid} {kind} {step}: refused ({problem}); {asked.note}{cut}"],
            }
        return {
            **used,
            "attempts": [*state["attempts"], attempt],
            "route": "run",
            "steps": [f"{fid} {kind} {step}: test written ({asked.note}){cut}"],
        }

    def explore(state: CandidateState, answers_key: str, back: str) -> dict[str, Any]:
        """Answer the pending lookups, and go back to the node that asked."""
        lookups = [Lookup.model_validate(item) for item in state["pending"]]
        results = services.explore(lookups)
        found = sum(result.ok for result in results)
        return {
            answers_key: [result.model_dump() for result in results],
            "pending": [],
            "route": back,
            "steps": [f"{state['finding_id']} explore: {found} of {len(results)} lookups found"],
        }

    def run(state: CandidateState) -> dict[str, Any]:
        """Run the last test at both sides, and judge the outcome."""
        candidate = candidate_of(state)
        materials = materials_of(state)
        assert materials is not None
        fid = state["finding_id"]
        attempt = dict(state["attempts"][-1])
        changed = state["changed"]
        ran = services.run_test(fid, attempt["path"], attempt["code"], attempt["n"], changed)
        minutes = state["docker_minutes"] + ran.minutes
        attempt["file"] = ran.file
        outcomes = ran.outcomes
        match = None
        if len(outcomes) == 1:
            match = judge(
                outcomes[0],
                candidate.expected or "",
                attempt["path"],
                {path: set(lines) for path, lines in changed.items()},
                set(state["new_names"]),
                attempt["code"],
            )
        if match is None:
            attempt.update(
                {
                    "kind": "broken",
                    "feedback": (
                        f"The file ran as {len(outcomes)} tests; write exactly one test "
                        "function, with no parametrize."
                    ),
                    "reason": f"{len(outcomes)} tests",
                }
            )
            verdict = "broken"
        else:
            outcome = outcomes[0]
            attempt.update(
                {
                    "kind": match.verdict,
                    "reason": match.reason,
                    "feedback": match.feedback,
                    "outcome": outcome.outcome,
                    "base_result": match.base_result,
                    "pr_result": match.pr_result,
                    "failure": match.failure,
                    "failure_classes": list(match.failure_classes),
                    "through": list(match.through),
                    "caught": match.caught,
                    "proof": match.proof,
                    "base": side_text(outcome.old, "the base commit"),
                    "pr": side_text(outcome.new, "the pull request's commit"),
                    "hits": list(outcome.new.hits),
                }
            )
            verdict = match.verdict
        if ran.problems:
            attempt["problems"] = ran.problems
        attempts = [*state["attempts"][:-1], attempt]
        line = f"{fid} run {attempt['n']}: {verdict} ({attempt['reason']}), {ran.minutes:.1f} min"
        update: dict[str, Any] = {"attempts": attempts, "docker_minutes": minutes}
        if verdict == "proof":
            found = signs(
                inputs.pr,
                candidate.claim.file,
                short_names(materials.function),
                candidate.expected,
                materials.code,
            )
            route = "intent" if found else "finish"
            note = f"; {len(found)} signs of intent" if found else ""
            return {
                **update,
                "test_result": "proof",
                "intent": {"signs": found} if found else None,
                "route": route,
                "steps": [line + note],
            }
        if verdict == "broken":
            if state["writer_steps"] >= MAX_WRITER_STEPS:
                return {**update, "test_result": "broken", "route": "verify", "steps": [line]}
            if minutes >= DOCKER_MINUTES:
                return {**update, "test_result": "budget", "route": "verify", "steps": [line]}
            return {**update, "route": "write", "steps": [line]}
        return {**update, "test_result": "no_proof", "route": "verify", "steps": [line]}

    def intent(state: CandidateState) -> dict[str, Any]:
        """Ask whether the proven change was deliberate."""
        candidate = candidate_of(state)
        materials = materials_of(state)
        assert materials is not None and state["intent"] is not None
        proving = next(a for a in reversed(state["attempts"]) if a.get("proof"))
        test_run = f"It {proving['base']}. It {proving['pr']}"
        messages, _ = intent_request(
            candidate.claim,
            candidate.expected,
            proving["code"],
            test_run,
            state["intent"]["signs"],
            materials.code,
            inputs.intent_budget,
        )
        asked = ask(services.llm, messages, INTENT_FORMAT, IntentReply, state["sample"])
        fid = state["finding_id"]
        if asked.reply is None:
            return {
                **_used(state, asked),
                "failed": "no usable intent check reply",
                "route": "finish",
                "steps": [f"{fid} intent: no usable reply ({asked.note})"],
            }
        reply = asked.reply
        word = "deliberate" if reply.deliberate else "not deliberate"
        return {
            **_used(state, asked),
            "intent": {**state["intent"], **reply.model_dump()},
            "route": "finish",
            "steps": [f"{fid} intent: {word} ({asked.note})"],
        }

    def verify(state: CandidateState) -> dict[str, Any]:
        """Ask the verifier to refute the claim, or for lookups first."""
        candidate = candidate_of(state)
        materials = materials_of(state)
        may_look = not state["verify_looked"]
        warnings = overlapping(inputs.warnings, candidate)
        function = materials.function if materials else ""
        verify_inputs = VerifyInputs(
            claim=candidate.claim,
            expected=candidate.expected,
            code=materials.code if materials else "",
            hunks=materials.hunks if materials else "",
            callers=callers_text(inputs.pieces, function),
            warnings=[warning.text() for warning in warnings],
        )

        def request(look: bool) -> tuple[list[ChatMessage], list[str]]:
            return verify_request(
                verify_inputs,
                inputs.verify_budget,
                lookups=_answers(state["verify_answers"]),
                may_look=look,
            )

        messages, trimmed = request(may_look)
        closed = request(False)[0] if may_look else None
        asked = ask(
            services.llm, messages, VERIFY_FORMAT, VerifyReply, state["sample"], None, closed
        )
        fid = state["finding_id"]
        cut = f", parts cut: {', '.join(trimmed)}" if trimmed else ""
        if asked.lookups is not None:
            lookups = parse_lookups(asked.lookups, MAX_LOOKUPS)
            asked_for = "; ".join(lookup.describe() for lookup in lookups)
            return {
                **_used(state, asked),
                "verify_looked": True,
                "pending": [lookup.model_dump() for lookup in lookups],
                "route": "explore_verify",
                "steps": [f"{fid} verify: {len(lookups)} lookups ({asked.note}): {asked_for}{cut}"],
            }
        if asked.reply is None:
            return {
                **_used(state, asked),
                "failed": "no usable verifier reply",
                "route": "finish",
                "steps": [f"{fid} verify: no usable reply ({asked.note}){cut}"],
            }
        reply = asked.reply
        return {
            **_used(state, asked),
            "verdict": reply.model_dump(),
            "route": "finish",
            "steps": [f"{fid} verify: {reply.verdict} ({asked.note}){cut}"],
        }

    def finish(state: CandidateState) -> dict[str, Any]:
        """Build the candidate's Finding and save its record."""
        record = build_record(inputs, state, candidate_of(state))
        services.save(record)
        fid = state["finding_id"]
        line = f"{fid} done: tier {record.tier}" + (f" ({record.proof})" if record.proof else "")
        if record.status == "failed":
            line = f"{fid} failed: {state['failed']}"
        return {"results": [record.model_dump(mode="json")], "steps": [line]}

    def route(state: CandidateState) -> str:
        return state["route"]

    def shown(node: Callable[[CandidateState], dict[str, Any]]) -> Any:
        """Wrap a node so its log lines are shown as soon as it finishes."""

        def run_node(state: CandidateState) -> dict[str, Any]:
            update = node(state)
            for line in update.get("steps", []):
                services.show(line)
            return update

        return run_node

    builder = StateGraph(CandidateState)
    builder.add_node("start", shown(start))
    builder.add_node("write", shown(write))
    builder.add_node(
        "explore_write", shown(lambda state: explore(state, "writer_answers", "write"))
    )
    builder.add_node("run", shown(run))
    builder.add_node("intent", shown(intent))
    builder.add_node("verify", shown(verify))
    builder.add_node(
        "explore_verify", shown(lambda state: explore(state, "verify_answers", "verify"))
    )
    builder.add_node("finish", shown(finish))
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", route, ["write", "verify"])
    builder.add_conditional_edges("write", route, ["explore_write", "run", "write", "verify"])
    builder.add_conditional_edges("explore_write", route, ["write"])
    builder.add_conditional_edges("run", route, ["intent", "finish", "write", "verify"])
    builder.add_conditional_edges("intent", route, ["finish"])
    builder.add_conditional_edges("verify", route, ["explore_verify", "finish"])
    builder.add_conditional_edges("explore_verify", route, ["verify"])
    builder.add_edge("finish", END)
    return builder.compile()


def first_state(
    finding_id: str, sample: int, survey: PrSurvey | None, materials: Materials | None
) -> CandidateState:
    """Give a candidate subgraph's starting state.

    Args:
        finding_id (str): The candidate.
        sample (int): Which answers its model calls ask for.
        survey (PrSurvey | None): The PR's survey, or None when it failed.
        materials (Materials | None): The candidate's materials.

    Returns:
        CandidateState: The state.
    """
    return CandidateState(
        finding_id=finding_id,
        sample=sample,
        materials=materials.model_dump() if materials is not None else None,
        changed=source_changes(survey.changed) if survey is not None else {},
        new_names=list(survey.added_names) if survey is not None else [],
        route="",
        writer_steps=0,
        looked=False,
        pending=[],
        writer_answers=[],
        attempts=[],
        test_result="not_tried",
        docker_minutes=0.0,
        verify_looked=False,
        verify_answers=[],
        verdict=None,
        intent=None,
        failed="",
        calls=0,
        input_tokens=0,
        output_tokens=0,
        steps=[],
        results=[],
    )


# Errors that mean a PR can't be surveyed (its environment or commits are missing), as
# opposed to Docker failing, which stops the run.
SurveyProblem = tuple[type[Exception], ...]


def build_pr_graph(
    inputs: PrInputs,
    services: Services,
    checkpointer: BaseCheckpointSaver[Any] | None,
    survey_problems: SurveyProblem = (),
    fallback: Callable[[Candidate], Materials] | None = None,
) -> Any:
    """Wire the PR graph: survey, one candidate subgraph per candidate, collect.

    Args:
        inputs (PrInputs): The PR's inputs.
        services (Services): The model, Docker and the database.
        checkpointer (BaseCheckpointSaver[Any] | None): Where progress is saved after each
            node, the subgraphs' included. None saves nothing.
        survey_problems (SurveyProblem): Errors that mean the PR can't be surveyed; its
            candidates then go to the verifier with what the PR's own diff shows.
        fallback (Callable[[Candidate], Materials] | None): Builds a candidate's materials
            without a survey.

    Returns:
        Any: The compiled graph.
    """
    candidate_graph = build_candidate_graph(inputs, services)

    def survey(state: PrState) -> dict[str, Any]:
        """Survey the PR in a sandbox."""
        try:
            found = services.survey()
        except survey_problems as exc:
            return {"problem": str(exc)[:500], "notes": [f"survey failed: {str(exc)[:200]}"]}
        notes = (
            f"survey: {len(found.changed)} changed Python files, "
            f"{sum(m.test_path is not None for m in found.materials.values())} candidates "
            "with a place for a test"
        )
        return {"survey": found.model_dump(), "notes": [notes]}

    def fan_out(state: PrState) -> list[Send] | str:
        """Start one candidate subgraph per candidate to check."""
        if not state["todo"]:
            return "collect"
        found = PrSurvey.model_validate(state["survey"]) if state["survey"] else None
        sends = []
        for finding_id, sample in state["todo"]:
            materials = found.materials.get(finding_id) if found is not None else None
            if materials is None and fallback is not None:
                materials = fallback(inputs.candidates[finding_id])
            sends.append(Send("candidate", first_state(finding_id, sample, found, materials)))
        return sends

    def collect(state: PrState) -> dict[str, Any]:
        """Sum up the PR's checked candidates."""
        tiers = sorted(int(record["tier"]) for record in state["results"])
        counts = {tier: tiers.count(tier) for tier in (1, 2, 3, 4)}
        text = ", ".join(f"tier {tier}: {n}" for tier, n in counts.items() if n)
        return {"notes": [f"collect: {len(tiers)} candidates checked ({text or 'none'})"]}

    builder = StateGraph(PrState)
    builder.add_node("survey", survey)
    builder.add_node("candidate", candidate_graph)
    builder.add_node("collect", collect)
    builder.add_edge(START, "survey")
    builder.add_conditional_edges("survey", fan_out, ["candidate", "collect"])
    builder.add_edge("candidate", "collect")
    builder.add_edge("collect", END)
    return builder.compile(checkpointer=checkpointer)


def check_pr(
    graph: Any,
    pr_id: str,
    todo: Sequence[tuple[str, int]],
    thread_id: str,
    *,
    parallel: int,
    on_line: Callable[[str], None] | None = None,
    tags: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Run a PR graph on a thread: start it, or resume it where it stopped.

    Args:
        graph (Any): The compiled PR graph (build_pr_graph()).
        pr_id (str): The PR.
        todo (Sequence[tuple[str, int]]): The candidates to check, with their samples.
        thread_id (str): The thread. A new ID starts over.
        parallel (int): Most candidates checked at the same time.
        on_line (Callable[[str], None] | None): Called with each log line.
        tags (Sequence[str]): Tags for tracing, such as the model profile.

    Returns:
        list[dict[str, Any]]: The records of the candidates checked on this thread.

    Raises:
        openai.APIError, LimitError, LLMError, SandboxError: If a model call or Docker fails.
            What finished stays saved, so the same thread resumes after it.
    """
    show = on_line or (lambda line: None)
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": parallel,
        "run_name": "evidence",
        "tags": list(tags),
        "metadata": {"instance_id": pr_id},
    }
    saved = graph.get_state(config)
    start: PrState | None
    if saved.values:
        if not saved.next:
            return list(saved.values.get("results", []))
        show(f"(resuming at {', '.join(saved.next)})")
        start = None
    else:
        start = PrState(
            pr_id=pr_id,
            todo=[[finding_id, sample] for finding_id, sample in todo],
            survey=None,
            problem="",
            results=[],
            notes=[],
        )
    # Only the PR graph's own updates are streamed: its subgraphs print their lines through
    # Services.show as their nodes finish. (Streaming the subgraphs too, with subgraphs=True,
    # hangs LangGraph 1.2 when max_concurrency is set.)
    for update in graph.stream(start, config, stream_mode="updates"):
        for name, changes in update.items():
            if name != "candidate":
                for line in (changes or {}).get("notes", []):
                    show(line)
    return list(graph.get_state(config).values.get("results", []))
