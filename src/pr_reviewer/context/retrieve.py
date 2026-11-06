"""Run the probe in an indexer sandbox: a PR's candidate context, or what fix commits changed.

The indexer image is the sandbox image plus Jedi (docker/indexer/Dockerfile). For a PR:
1. a sandbox starts from it, network off, with the usual limits;
2. its working copy is checked out at the PR's last commit, and the base is found (as in M2);
3. the probe (probe.py) and a request are written in, and the probe runs with the image's
   Python, which has Jedi;
4. its answer is copied out as a file and checked (candidates.py).
For the retrieval check, the probe lists the functions given fix commits changed, from the
history alone, without checking anything out.

What the probe collects is set by RetrievalSettings. A fingerprint of those settings, the
probe's own source and the indexer image's tag keys every stored result, so changing any of
them never reuses an old one.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from docker import DockerClient
from pydantic import ValidationError

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.candidates import Candidates, FixFunctions
from pr_reviewer.core.config import ModelProfile, project_root
from pr_reviewer.reviewer.naive import input_budget_chars
from pr_reviewer.sandbox.checkout import base_of, checkout, working_copy
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.image import ImageSpec, build_image, image_exists, sandbox_image
from pr_reviewer.sandbox.runner import Sandbox, SandboxError

# The indexer image's name, and its Dockerfile relative to the project root.
INDEXER_NAME = "pr-reviewer-indexer"
INDEXER_DOCKERFILE = Path("docker") / "indexer" / "Dockerfile"

# The Python inside the indexer image that has Jedi.
INDEXER_PYTHON = "/opt/indexer/bin/python"

# Where the probe and its files go inside the sandbox (outside the repo).
PROBE_DIR = "/tmp/pr_reviewer_context"
PROBE_FILE = f"{PROBE_DIR}/probe.py"
REQUEST_FILE = f"{PROBE_DIR}/request.json"
ANSWER_FILE = f"{PROBE_DIR}/answer.json"

# Where candidates are kept, inside the project's data/ folder (not in git).
CONTEXTS = Path("data") / "contexts"


class ContextError(Exception):
    """Raised when the probe fails or gives an answer that doesn't check out."""


@dataclass(frozen=True)
class RetrievalSettings:
    """What the probe collects for each PR.

    Attributes:
        max_piece_chars (int): Pieces bigger than this are skipped; the request budget.
        callers (int): Most callers kept per changed function.
        tests (int): Most tests kept per changed function.
        short_lines (int): Called functions, callers and tests up to this many lines are
            shown whole.
        doc_lines (int): Most docstring lines shown with a called function's signature.
        jedi_seconds (float): Most time spent in Jedi per PR.
        jedi_call_seconds (float): Most time one Jedi question may take.
    """

    max_piece_chars: int
    callers: int = 5
    tests: int = 3
    short_lines: int = 30
    doc_lines: int = 10
    jedi_seconds: float = 120.0
    jedi_call_seconds: float = 10.0


def probe_source() -> str:
    """Give the probe's source, to copy into sandboxes.

    Returns:
        str: The text of probe.py.
    """
    return resources.files("pr_reviewer.context").joinpath("probe.py").read_text("utf-8")


def indexer_image(config: SandboxConfig) -> ImageSpec:
    """Describe the indexer image, built on the current sandbox image.

    Args:
        config (SandboxConfig): The sandbox settings, which name the sandbox image.

    Returns:
        ImageSpec: The image.
    """
    return ImageSpec(INDEXER_NAME, project_root() / INDEXER_DOCKERFILE, sandbox_image(config).tag)


def ready_indexer(client: DockerClient, config: SandboxConfig) -> str:
    """Build the indexer image if it's missing.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.

    Returns:
        str: The indexer image's name and tag.

    Raises:
        ImageError: If the sandbox image isn't built, or the build fails.
    """
    spec = indexer_image(config)
    if not image_exists(client, spec.tag):
        print(f"Building {spec.tag} (the sandbox image plus Jedi, about a minute)", flush=True)
        build_image(client, spec)
    return spec.tag


def settings_key(settings: RetrievalSettings, indexer_tag: str) -> str:
    """Fingerprint what decides the candidates: the settings, the probe's code, the image.

    Args:
        settings (RetrievalSettings): The retrieval settings.
        indexer_tag (str): The indexer image's name and tag.

    Returns:
        str: 64 hex characters.
    """
    probe = hashlib.sha256(probe_source().encode()).hexdigest()
    text = json.dumps(
        {"settings": asdict(settings), "probe": probe, "indexer": indexer_tag}, sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()


def retrieval_setup(
    config: SandboxConfig, profile: ModelProfile
) -> tuple[RetrievalSettings, str, str, int]:
    """Give the current retrieval settings, the indexer image tag and their key.

    Args:
        config (SandboxConfig): The sandbox settings, which name the sandbox image.
        profile (ModelProfile): The reviewer's model profile, whose request budget sizes the
            pieces.

    Returns:
        tuple[RetrievalSettings, str, str, int]: The settings, the indexer image's name and
        tag, the settings key, and the request budget in characters.
    """
    budget = input_budget_chars(profile)
    settings = RetrievalSettings(max_piece_chars=budget)
    tag = indexer_image(config).tag
    return settings, tag, settings_key(settings, tag), budget


def candidates_path(key: str, instance_id: str) -> Path:
    """Give the file a PR's candidates are kept in.

    Args:
        key (str): The settings key (settings_key()).
        instance_id (str): The PR.

    Returns:
        Path: data/contexts/<first 12 characters of the key>/<instance_id>.json.
    """
    return project_root() / CONTEXTS / key[:12] / f"{instance_id}.json"


def run_probe(
    sandbox: Sandbox, mode: str, request: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Run the probe in a sandbox and read its answer.

    Args:
        sandbox (Sandbox): The sandbox, network off.
        mode (str): "retrieve" or "fixdefs".
        request (dict[str, Any]): The probe's request.
        timeout (float): Seconds the probe may run.

    Returns:
        dict[str, Any]: The answer, as parsed JSON (not checked yet).

    Raises:
        ContextError: If the probe fails or writes no valid JSON.
        SandboxError: If Docker or the sandbox fails.
    """
    if sandbox.online:
        raise SandboxError("The probe must run with the network off")
    made = sandbox.run(["mkdir", "-p", PROBE_DIR], timeout=60, label="mkdir")
    if not made.ok:
        raise SandboxError(f"Can't make {PROBE_DIR}: {made.output.strip()[-300:]}")
    sandbox.write_files(
        PROBE_DIR, {"probe.py": probe_source(), "request.json": json.dumps(request)}
    )
    result = sandbox.run(
        [INDEXER_PYTHON, PROBE_FILE, mode, REQUEST_FILE, ANSWER_FILE],
        timeout=timeout,
        label=f"probe-{mode}",
    )
    if not result.ok:
        raise ContextError(
            f"The probe ({mode}) failed: {result.describe()}: {result.output.strip()[-800:]}"
        )
    try:
        answer = json.loads(sandbox.read_file(ANSWER_FILE).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContextError(f"The probe's answer isn't valid JSON: {exc}") from None
    if not isinstance(answer, dict):
        raise ContextError("The probe's answer isn't a JSON object")
    return answer


def retrieve_candidates(
    client: DockerClient,
    config: SandboxConfig,
    image: str,
    pr: Instance,
    settings: RetrievalSettings,
    log_dir: Path | None = None,
) -> Candidates:
    """Collect a PR's candidate context in a fresh indexer sandbox.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.
        image (str): The indexer image's name and tag.
        pr (Instance): The PR.
        settings (RetrievalSettings): What to collect.
        log_dir (Path | None): Folder for the sandbox's command logs.

    Returns:
        Candidates: The checked candidates.

    Raises:
        ContextError: If the probe fails or its answer doesn't check out.
        RepoError, CheckoutError: If the PR's commits can't be checked out.
        SandboxError: If Docker or the sandbox fails.
    """
    timeout = config.timeouts.test_minutes * 60
    with Sandbox(client, config, image, purpose="context", log_dir=log_dir) as sandbox:
        path, head = checkout(sandbox, pr, "pr")
        base = base_of(sandbox, path, pr)
        request = {"root": path, "base": base, "head": head, **asdict(settings)}
        answer = run_probe(sandbox, "retrieve", request, timeout)
    try:
        candidates = Candidates.model_validate({**answer, "instance_id": pr.instance_id})
    except ValidationError as exc:
        raise ContextError(
            f"{pr.instance_id}: the probe's answer doesn't check out: {exc}"
        ) from None
    if candidates.head != head or candidates.base != base:
        raise ContextError(f"{pr.instance_id}: the probe answered for other commits")
    return candidates


def fix_functions(
    client: DockerClient,
    config: SandboxConfig,
    image: str,
    pr: Instance,
    commits: list[str],
    log_dir: Path | None = None,
) -> dict[str, FixFunctions]:
    """List the functions each of a PR's fix commits changed, in a fresh indexer sandbox.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.
        image (str): The indexer image's name and tag.
        pr (Instance): The PR.
        commits (list[str]): Full IDs of its fix commits.
        log_dir (Path | None): Folder for the sandbox's command logs.

    Returns:
        dict[str, FixFunctions]: What each commit changed, by commit ID.

    Raises:
        ContextError: If the probe fails or its answer doesn't check out.
        SandboxError: If Docker or the sandbox fails.
    """
    known = {commit.sha for commit in pr.all_commits}
    unknown = [commit for commit in commits if commit not in known]
    if unknown:
        raise ContextError(f"{pr.instance_id}: {unknown[0][:10]} isn't one of the PR's commits")
    timeout = config.timeouts.test_minutes * 60
    with Sandbox(client, config, image, purpose="fixdefs", log_dir=log_dir) as sandbox:
        path = working_copy(sandbox, pr.repo)
        answer = run_probe(sandbox, "fixdefs", {"root": path, "commits": commits}, timeout)
    try:
        fixes = answer["fixes"]
        return {commit: FixFunctions.model_validate(fixes[commit]) for commit in commits}
    except (KeyError, TypeError, ValidationError) as exc:
        raise ContextError(
            f"{pr.instance_id}: the probe's fix answer doesn't check out: {exc}"
        ) from None
