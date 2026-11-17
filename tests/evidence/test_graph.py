"""Tests for the evidence agent's graphs (evidence/graph.py), with a fake model and fake Docker.

The model is a fake provider that answers by prompt (test writer, verifier, intent check) and
by candidate, so candidates can run in parallel in any order. Tests "run" by returning scripted
outcomes. Checks each path: a test that proves the claim, a broken test fixed, a lookup round,
a run that proves nothing going to the verifier, the test writer's budget, an evolvability
claim with a confirmed static warning, a deliberate change, a verifier that never answers, a
reply the provider refused (no JSON, bad JSON or a call to a tool that doesn't exist) asked
again with lookups closed, or read when it called a tool not offered with a lookup or a valid
answer, two candidates at once, and resuming after Docker failed mid-run.

How to run (from the project folder):
    pytest tests/evidence/test_graph.py
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx2
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from pr_reviewer.context.lookups import Lookup, LookupResult
from pr_reviewer.core.schemas import Claim
from pr_reviewer.difftest.outcomes import DiffOutcome
from pr_reviewer.evidence.graph import (
    EvidenceRecord,
    GeneratedRun,
    PrInputs,
    SerialClient,
    Services,
    build_pr_graph,
    check_pr,
)
from pr_reviewer.evidence.inputs import Candidate
from pr_reviewer.evidence.prompts import TEST_PROMPT, VERIFY_PROMPT
from pr_reviewer.evidence.survey import Materials, PrSurvey
from pr_reviewer.sandbox.runner import SandboxError
from tests.evidence.helpers import failure, outcome, side
from tests.fakes import completion_body, fake_client
from tests.reviewer.helpers import make_pr
from tests.static.test_warnings import warning

TEST = "pkg/tests/test_pr_reviewer_{}.py"

GOOD_TEST = "from pkg.cart import Cart\n\n\ndef test_x():\n    Cart(None).total()\n"


def claim(fid: str, category: str = "F.4", expected: str | None = "AttributeError") -> Claim:
    """Build a claim whose summary names its candidate, so the fake model can tell them apart.

    Args:
        fid (str): The finding ID.
        category (str): Its category.
        expected (str | None): Its expected exception.

    Returns:
        Claim: The claim, on pkg/cart.py lines 10-12.
    """
    return Claim(
        file="pkg/cart.py",
        start_line=10,
        end_line=12,
        category=category,
        summary=f"SUMMARY-{fid}: total breaks on None.",
        expected_failure=expected,
        confidence=0.7,
    )


class RoutedModel:
    """A fake provider that answers by prompt kind and candidate, with queued replies.

    Attributes:
        queues (dict[tuple[str, str], list[str | httpx2.Response]]): Replies by (kind, finding
            ID), where kind is "test", "verify" or "intent". A text is the model's reply; a
            response is sent as it is (such as a provider error).
        asked (list[tuple[str, str, str]]): Every request: kind, finding ID, user message.
        offered (list[bool]): For every request, whether it offered the lookup tools.
    """

    def __init__(self, queues: Mapping[tuple[str, str], Sequence[str | httpx2.Response]]) -> None:
        self.queues = {key: list(replies) for key, replies in queues.items()}
        self.asked: list[tuple[str, str, str]] = []
        self.offered: list[bool] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Answer one request with the next reply queued for its kind and candidate.

        Args:
            request (httpx2.Request): The request.

        Returns:
            httpx2.Response: The reply, or a 400 error when none is queued.
        """
        body = json.loads(request.content)
        messages = body["messages"]
        assert not ("tools" in body and "response_format" in body), "Groq takes one of them"
        self.offered.append("tools" in body)
        system, user = messages[0]["content"], messages[1]["content"]
        kind = (
            "test" if system == TEST_PROMPT else "verify" if system == VERIFY_PROMPT else "intent"
        )
        fid = user.split("SUMMARY-", 1)[1].split(":", 1)[0]
        self.asked.append((kind, fid, "\n".join(m["content"] for m in messages[1:])))
        queue = self.queues.get((kind, fid), [])
        if not queue:
            return httpx2.Response(400, json={"error": {"message": f"no reply for {kind} {fid}"}})
        answer = queue.pop(0)
        if isinstance(answer, httpx2.Response):
            return answer
        return httpx2.Response(200, json=completion_body(answer))

    def count(self, kind: str, fid: str) -> int:
        """Count the requests of one kind for one candidate.

        Args:
            kind (str): "test", "verify" or "intent".
            fid (str): The finding ID.

        Returns:
            int: How many were asked.
        """
        return sum(1 for k, f, _ in self.asked if (k, f) == (kind, fid))


def wrote(code: str = GOOD_TEST) -> str:
    """Write the test writer's reply with a test.

    Args:
        code (str): The test.

    Returns:
        str: The JSON reply.
    """
    return json.dumps({"test_code": code})


def verdict(word: str, warnings: Sequence[dict[str, Any]] = ()) -> str:
    """Write the verifier's reply.

    Args:
        word (str): "upheld" or "refuted".
        warnings (Sequence[dict[str, Any]]): Its judgments of the warnings.

    Returns:
        str: The JSON reply.
    """
    trace = "Cart(None) -> line 11 -> None.items" if word == "upheld" else ""
    return json.dumps(
        {
            "verdict": word,
            "argument": f"{word} because.",
            "trace": trace,
            "warnings": list(warnings),
        }
    )


READ_CART = ("read_file", {"path": "pkg/cart.py", "start_line": 1, "end_line": 9})


def called(*calls: tuple[str, dict[str, Any] | str]) -> httpx2.Response:
    """Build a reply that calls the lookup tools instead of answering.

    Args:
        *calls (tuple[str, dict[str, Any] | str]): Each call's tool name and arguments.

    Returns:
        httpx2.Response: The reply.
    """
    return httpx2.Response(200, json=completion_body(None, tool_calls=calls or (READ_CART,)))


PROOF = outcome(side("pass"), side("fail", failure(frames=("TESTFILE:5", "pkg/cart.py:11"))))
BROKEN = outcome(
    side("fail", failure("ModuleNotFoundError", phase="collect", message="No module named 'x'")),
    side("fail", failure("ModuleNotFoundError", phase="collect", message="No module named 'x'")),
)
PASSES = outcome(side("pass"), side("pass"))


class Harness:
    """Fake services for one PR: survey, test runs, lookups and saved records.

    Attributes:
        runs (dict[str, list[DiffOutcome]]): Scripted outcomes per finding ID, in order.
        ran (list[tuple[str, int, Mapping[str, Sequence[int]]]]): Each run: finding, attempt,
            watched lines.
        saved (list[EvidenceRecord]): Every saved record.
        explored (list[list[Lookup]]): Every lookup round answered.
        fail_runs (int): Runs left that raise SandboxError first.
    """

    def __init__(self, runs: Mapping[str, Sequence[DiffOutcome]], fail_runs: int = 0) -> None:
        self.runs = {fid: list(found) for fid, found in runs.items()}
        self.ran: list[tuple[str, int, Mapping[str, Sequence[int]]]] = []
        self.saved: list[EvidenceRecord] = []
        self.explored: list[list[Lookup]] = []
        self.fail_runs = fail_runs

    def run_test(
        self, fid: str, path: str, code: str, attempt: int, watch: Mapping[str, Sequence[int]]
    ) -> GeneratedRun:
        """Give the next scripted outcome of a candidate's test.

        Args:
            fid (str): The finding ID.
            path (str): The test's path.
            code (str): Its code.
            attempt (int): Its attempt number.
            watch (Mapping[str, Sequence[int]]): The lines to watch.

        Returns:
            GeneratedRun: The run.

        Raises:
            SandboxError: While fail_runs is above 0.
        """
        if self.fail_runs:
            self.fail_runs -= 1
            raise SandboxError("Docker went away")
        self.ran.append((fid, attempt, watch))
        found = self.runs[fid].pop(0)
        test = found.model_copy(update={"test": f"{path}::test_x"})
        relabeled = _relabel(test, path)
        return GeneratedRun((relabeled,), 1.5, "", f"data/evidence/r/{fid}/attempt-{attempt}.py")

    def explore(self, lookups: list[Lookup]) -> list[LookupResult]:
        """Answer lookups without a sandbox.

        Args:
            lookups (list[Lookup]): The lookups.

        Returns:
            list[LookupResult]: One answer each.
        """
        self.explored.append(lookups)
        return [LookupResult(lookup=item, ok=True, text="class Cart: ...") for item in lookups]


def _relabel(found: DiffOutcome, path: str) -> DiffOutcome:
    """Point a scripted outcome's TESTFILE frames at the candidate's own test file.

    Args:
        found (DiffOutcome): The outcome, written with "TESTFILE" for the test file.
        path (str): The candidate's test file.

    Returns:
        DiffOutcome: The same outcome with those frames renamed.
    """
    sides = {}
    for name in ("old", "new"):
        result = getattr(found, name)
        if result.failure is not None:
            frames = tuple(f.replace("TESTFILE", path) for f in result.failure.frames)
            fixed = result.failure.model_copy(update={"frames": frames})
            result = result.model_copy(update={"failure": fixed})
        sides[name] = result
    return found.model_copy(update=sides)


def run(
    candidates: Sequence[Candidate],
    model: RoutedModel,
    harness: Harness,
    *,
    pr_files: Sequence[tuple[str, str]] = (("pkg/cart.py", "@@ -10 +10,3 @@\n+x = 1"),),
    warnings: Sequence[Any] = (),
    parallel: int = 1,
    saver: InMemorySaver | None = None,
    thread: str = "t1",
) -> list[dict[str, Any]]:
    """Run the PR graph on some candidates with fakes.

    Args:
        candidates (Sequence[Candidate]): The candidates.
        model (RoutedModel): The fake model.
        harness (Harness): The fake services.
        pr_files (Sequence[tuple[str, str]]): The PR's changed files and diffs.
        warnings (Sequence[Any]): Static warnings on its changed lines.
        parallel (int): Candidates at a time.
        saver (InMemorySaver | None): The checkpointer; None for a new one.
        thread (str): The thread ID.

    Returns:
        list[dict[str, Any]]: The records the graph returns.
    """
    pr = make_pr(files=pr_files)
    survey = PrSurvey(
        instance_id=pr.instance_id,
        base="b" * 40,
        head="h" * 40,
        changed={"pkg/cart.py": [10, 11, 12], "pkg/tests/test_cart.py": [3]},
        added_names=("total",),
        materials={
            c.finding_id: Materials(
                finding_id=c.finding_id,
                function="Cart.total",
                code="11 |         return self.items.total",
                hunks="--- pkg/cart.py\n@@ -10 +10,3 @@\n+x = 1",
                test_path=TEST.format(c.finding_id),
                example="# pkg/tests/test_cart.py\nimport pytest",
                fixtures=("cart(tmp_path)",),
            )
            for c in candidates
        },
    )
    inputs = PrInputs(
        pr=pr,
        candidates={c.finding_id: c for c in candidates},
        warnings=tuple(warnings),
        pieces=(),
        pytest=True,
        writer_budget=11_000,
        verify_budget=11_000,
        intent_budget=11_000,
    )
    services = Services(
        llm=SerialClient(fake_client(model)),
        survey=lambda: survey,
        run_test=harness.run_test,
        explore=harness.explore,
        save=harness.saved.append,
    )
    graph = build_pr_graph(inputs, services, saver or InMemorySaver())
    todo = [(c.finding_id, 0) for c in candidates]
    return check_pr(graph, pr.instance_id, todo, thread, parallel=parallel)


def candidate(fid: str = "f1", **changes: Any) -> Candidate:
    """Build a candidate.

    Args:
        fid (str): Its finding ID.
        **changes (Any): Changes to its claim, such as category="E.1.1".

    Returns:
        Candidate: The candidate.
    """
    return Candidate("o__r-1", fid, claim(fid, **changes))


def only(harness: Harness) -> EvidenceRecord:
    """Give the one record a run saved.

    Args:
        harness (Harness): The fake services.

    Returns:
        EvidenceRecord: The record.
    """
    (record,) = harness.saved
    return record


def test_a_test_that_fails_as_claimed_is_a_tier_1_proof() -> None:
    """Check the shortest path: one test, a proof, no sign of intent, saved as tier 1."""
    model = RoutedModel({("test", "f1"): [wrote()]})
    harness = Harness({"f1": [PROOF]})
    results = run([candidate()], model, harness)
    record = only(harness)
    assert (record.tier, record.proof, record.test_result, record.verdict) == (
        1,
        "executed",
        "proof",
        None,
    )
    assert record.test_path == "data/evidence/r/f1/attempt-1.py"
    assert record.calls == 1 and record.docker_minutes == 1.5
    assert results == [record.model_dump(mode="json")]
    assert harness.ran[0][2] == {"pkg/cart.py": [10, 11, 12]}, "test files aren't watched"


def test_a_broken_test_is_fixed_with_its_error() -> None:
    """Check a broken run sends the test and its error back, and the fix proves the claim."""
    model = RoutedModel({("test", "f1"): [wrote(), wrote(GOOD_TEST + "# fixed\n")]})
    harness = Harness({"f1": [BROKEN, PROOF]})
    run([candidate()], model, harness)
    record = only(harness)
    assert (record.tier, record.calls) == (1, 2)
    fix = model.asked[1][2]
    assert "<previous_test>" in fix and "ModuleNotFoundError" in fix and "while loading" in fix
    assert record.test_path == "data/evidence/r/f1/attempt-2.py"


def test_the_test_writer_may_look_things_up_once() -> None:
    """Check a lookup round by tool call, then a test written with the answers in a request
    that offers no tools."""
    model = RoutedModel({("test", "f1"): [called(), wrote()]})
    harness = Harness({"f1": [PROOF]})
    run([candidate()], model, harness)
    assert only(harness).tier == 1 and len(harness.explored) == 1
    assert "You may answer now, or first call the lookup tools" in model.asked[0][2]
    assert "class Cart: ..." in model.asked[1][2]
    assert "No more lookups" in model.asked[1][2]
    assert model.offered == [True, False]


def test_a_run_that_proves_nothing_goes_to_the_verifier() -> None:
    """Check a test passing on both commits, then an upheld claim: tier 3."""
    model = RoutedModel({("test", "f1"): [wrote()], ("verify", "f1"): [verdict("upheld")]})
    harness = Harness({"f1": [PASSES]})
    run([candidate()], model, harness)
    record = only(harness)
    assert (record.tier, record.test_result, record.verdict) == (3, "no_proof", "upheld")
    assert record.evidence[-1]["reasoning"].startswith("Cart(None)")


def test_the_test_writer_stops_at_its_budget() -> None:
    """Check three broken tests end the test writer, and the verifier decides."""
    model = RoutedModel(
        {
            ("test", "f1"): [wrote(), wrote(GOOD_TEST + "#2\n"), wrote(GOOD_TEST + "#3\n")],
            ("verify", "f1"): [verdict("refuted")],
        }
    )
    harness = Harness({"f1": [BROKEN, BROKEN, BROKEN]})
    run([candidate()], model, harness)
    record = only(harness)
    assert (record.tier, record.test_result, record.verdict) == (4, "broken", "refuted")
    assert model.count("test", "f1") == 3 and len(harness.ran) == 3


def test_a_refused_test_counts_against_the_budget_without_running() -> None:
    """Check a test our checks refuse is sent back with the reason, and never runs."""
    model = RoutedModel(
        {
            ("test", "f1"): [wrote("def test_a():\n    exec('x')\n"), wrote()],
        }
    )
    harness = Harness({"f1": [PROOF]})
    run([candidate()], model, harness)
    assert only(harness).tier == 1 and len(harness.ran) == 1
    assert "Our checks refused the test: the test may not use exec" in model.asked[1][2]


def test_an_evolvability_claim_goes_to_the_verifier_which_confirms_a_warning() -> None:
    """Check a claim no test can show, and a static warning confirmed as the same problem."""
    model = RoutedModel(
        {
            ("verify", "f1"): [
                called(("search_symbol", {"symbol": "Cart"})),
                verdict("refuted", [{"id": 1, "same_problem": True}]),
            ]
        }
    )
    harness = Harness({})
    found = warning(line=11, file="pkg/cart.py")
    run([candidate(category="E.1.1", expected=None)], model, harness, warnings=[found])
    record = only(harness)
    assert (record.tier, record.test_result, record.tool_confirmed) == (2, "not_tried", True)
    assert harness.ran == [] and len(harness.explored) == 1
    assert "<warnings>" in model.asked[0][2] and "pkg/cart.py:11 mypy" in model.asked[0][2]


def test_a_deliberate_change_is_not_a_proof() -> None:
    """Check the intent check runs when the PR's tests name the function, and can drop a proof."""
    model = RoutedModel(
        {
            ("test", "f1"): [wrote()],
            ("intent", "f1"): [json.dumps({"deliberate": True, "reason": "its test expects it"})],
        }
    )
    harness = Harness({"f1": [PROOF]})
    files = (
        ("pkg/cart.py", "@@ -10 +10 @@\n+x = 1"),
        ("pkg/tests/test_cart.py", "@@ -1 +1 @@\n+    assert Cart(None).total() is None"),
    )
    run([candidate()], model, harness, pr_files=files)
    record = only(harness)
    assert (record.tier, record.verdict, record.proof) == (4, "deliberate", None)
    assert record.evidence[0]["deliberate"] is True
    assert "assert Cart(None).total() is None" in model.asked[1][2]


def test_a_verifier_that_never_answers_fails_the_candidate() -> None:
    """Check three bad replies are each asked again with the reason, then the candidate fails."""
    model = RoutedModel(
        {("verify", "f1"): ['{"verdict": "maybe"}', "no json", '{"verdict": "lookup"}']}
    )
    harness = Harness({})
    run([candidate(category="E.3.1", expected=None)], model, harness)
    record = only(harness)
    assert (record.status, record.calls, record.tier) == ("failed", 3, 4)
    assert "Your reply was rejected" in model.asked[1][2]


def refusal(code: str, generation: str = "") -> httpx2.Response:
    """Build Groq's HTTP 400 refusal of a bad reply.

    Args:
        code (str): "json_validate_failed" or "tool_use_failed".
        generation (str): The refused text Groq sends back.

    Returns:
        httpx2.Response: The refusal.
    """
    message = {
        "json_validate_failed": "Failed to validate JSON.",
        "tool_use_failed": "Tool choice is none, but model called a tool",
    }[code]
    error = {"message": message, "code": code, "failed_generation": generation}
    return httpx2.Response(400, json={"error": error})


@pytest.mark.parametrize("code", ["json_validate_failed", "tool_use_failed"])
def test_a_refused_reply_is_asked_again_with_lookups_closed(
    code: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Check Groq's refusal of a bad reply is asked again, not a crash, with the same request
    but lookups closed, and the refused text is logged, cut to its start and end."""
    refused = refusal(code, "a" * 1000 + "b" * 1000)
    model = RoutedModel({("verify", "f1"): [refused, verdict("refuted")]})
    harness = Harness({})
    run([candidate(category="E.3.1", expected=None)], model, harness)
    record = only(harness)
    assert (record.status, record.calls, record.tier) == ("done", 2, 4)
    assert "You may answer now, or first call" in model.asked[0][2]
    retry = model.asked[1][2]
    assert "No more lookups" in retry and "Lookups are off now" in retry
    assert "You may answer now, or first call" not in retry
    assert model.offered == [True, False]
    logged = caplog.text
    assert f"reply refused: {code}, 2000 characters: '{'a' * 900} [...] {'b' * 600}'" in logged


@pytest.mark.parametrize(
    ("arguments", "calls", "offered"),
    [
        (json.loads(verdict("refuted")), 1, [True]),
        ({"verdict": "maybe"}, 2, [True, False]),
    ],
)
def test_a_refused_answer_sent_as_a_tool_call_is_read_when_it_checks_out(
    arguments: dict[str, Any], calls: int, offered: list[bool]
) -> None:
    """Check an answer sent as a call to a "json" tool, which Groq refuses, is used as the
    reply when it checks out, and asked again with lookups closed when it doesn't."""
    generation = json.dumps({"name": "json", "arguments": arguments})
    refused = refusal("tool_use_failed", generation)
    model = RoutedModel({("verify", "f1"): [refused, verdict("refuted")]})
    harness = Harness({})
    run([candidate(category="E.3.1", expected=None)], model, harness)
    record = only(harness)
    assert (record.status, record.calls, record.verdict) == ("done", calls, "refuted")
    assert model.offered == offered
    read = "refused: answer read" in " ".join(record.details["log"])
    assert read == (calls == 1)


def test_a_refused_lookup_under_another_tool_name_is_a_lookup() -> None:
    """Check a lookup called under a prefixed name, which Groq refuses, is answered like any
    lookup."""
    generation = json.dumps({"name": "repo_browser.read_file", "arguments": READ_CART[1]})
    model = RoutedModel({("test", "f1"): [refusal("tool_use_failed", generation), wrote()]})
    harness = Harness({"f1": [PROOF]})
    run([candidate()], model, harness)
    record = only(harness)
    assert record.tier == 1 and len(harness.explored) == 1
    assert harness.explored[0][0].path == "pkg/cart.py"
    assert "class Cart: ..." in model.asked[1][2] and model.offered == [True, False]
    assert "refused: tool call read" in " ".join(record.details["log"])


def test_an_empty_reply_to_a_request_with_tools_is_asked_again_with_lookups_closed() -> None:
    """Check a model that ends its turn with neither an answer nor a tool call is asked again
    without the tools."""
    empty = httpx2.Response(200, json=completion_body(""))
    model = RoutedModel({("test", "f1"): [empty, wrote()]})
    harness = Harness({"f1": [PROOF]})
    run([candidate()], model, harness)
    record = only(harness)
    assert (record.tier, record.calls) == (1, 2)
    assert "Lookups are off now" in model.asked[1][2] and model.offered == [True, False]
    assert "model: empty, lookups closed" in " ".join(record.details["log"])


@pytest.mark.parametrize(
    ("generation", "note"),
    [("", "it held no JSON object; answer with the JSON"), ('{"test_code": ', "JSON format")],
)
def test_a_refused_reply_without_lookups_to_close_gets_a_plain_note(
    generation: str, note: str
) -> None:
    """Check a refusal after the lookup round asks the same request again, with a note saying
    whether any JSON came out."""
    refused = refusal("json_validate_failed", generation)
    model = RoutedModel({("test", "f1"): [called(), refused, wrote()]})
    harness = Harness({"f1": [PROOF]})
    run([candidate()], model, harness)
    assert only(harness).tier == 1
    retry = model.asked[2][2]
    assert note in retry and "Lookups are off now" not in retry
    assert model.asked[1][2] in retry, "the same request, with the note after it"


def test_two_candidates_run_at_once_and_a_finished_thread_is_kept() -> None:
    """Check two candidates in parallel, and that running the thread again asks nothing new."""
    model = RoutedModel(
        {
            ("test", "f1"): [wrote()],
            ("test", "f2"): [wrote()],
            ("verify", "f2"): [verdict("refuted")],
        }
    )
    harness = Harness({"f1": [PROOF], "f2": [PASSES]})
    saver = InMemorySaver()
    results = run([candidate("f1"), candidate("f2")], model, harness, parallel=2, saver=saver)
    assert sorted((r["finding_id"], r["tier"]) for r in results) == [("f1", 1), ("f2", 4)]
    again = run([candidate("f1"), candidate("f2")], model, harness, parallel=2, saver=saver)
    assert sorted(r["finding_id"] for r in again) == ["f1", "f2"]
    assert len(model.asked) == 3 and len(harness.saved) == 2


def test_a_run_stopped_by_docker_resumes_where_it_stopped() -> None:
    """Check a Docker failure stops the run, and the same thread resumes without asking again."""
    model = RoutedModel({("test", "f1"): [wrote()]})
    harness = Harness({"f1": [PROOF]}, fail_runs=1)
    saver = InMemorySaver()
    with pytest.raises(SandboxError):
        run([candidate()], model, harness, saver=saver)
    run([candidate()], model, harness, saver=saver)
    assert only(harness).tier == 1
    assert model.count("test", "f1") == 1, "the written test is kept, not asked for again"
