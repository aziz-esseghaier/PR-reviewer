"""Run ruff, mypy, bandit and semgrep on a PR's changed files, inside the PR's built environment.

The tools live in the tools image (docker/tools/Dockerfile), at /opt/tools. They reach a PR's
sandbox without changing the PR's image: /opt/tools is copied once into a Docker volume
(named after the image's tag), which every static sandbox attaches read-only at the same place.

For one PR (run_static()):
1. a sandbox starts from the PR's prepared image (difftest/environments.py): its working copy
   is at the PR's last commit and installed, with the network off and the usual limits;
2. git lists the lines the PR added or changed since its base, in Python files;
3. each tool checks the changed files, with fixed rules that ignore the repo's own settings:
   - ruff: pyflakes (F), bugbear (B) and pylint's errors (PLE);
   - mypy: its default checks plus the inside of unannotated functions, with the
     environment's Python and installed packages, so it sees the real types of the libraries
     the code uses. It first follows imports into the
     rest of the repo silently; when that runs out of time or memory, it checks the changed
     files alone (imports not followed);
   - bandit: security problems of medium severity or worse;
   - semgrep: the correctness and security rules for Python from semgrep-rules;
4. each tool writes JSON, read back as files; warnings on untouched lines are set aside.
Every warning on the changed files is kept in the result; the reviewer sees only those on
changed lines.

Imported by other modules; the file isn't run directly.
"""

import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import docker
from docker import DockerClient
from pydantic import BaseModel, ConfigDict, Field

from pr_reviewer.benchmark.dataset import Instance
from pr_reviewer.context.probe import parse_diff
from pr_reviewer.core.config import project_root
from pr_reviewer.sandbox.checkout import base_of, checkout
from pr_reviewer.sandbox.config import LABEL, SandboxConfig
from pr_reviewer.sandbox.image import ImageSpec, build_image, image_exists, sandbox_image
from pr_reviewer.sandbox.runner import DOCKER_ERRORS, USER, RunResult, Sandbox, SandboxError
from pr_reviewer.setup_agent.install import VENV
from pr_reviewer.static.warnings import PARSERS, TOOLS, Tool, ToolWarning, on_changed_lines

# Raised whenever what a run finds changes, so stored results aren't reused.
VERSION = 1

# The tools image's name, its Dockerfile and the other file its build copies in.
TOOLS_NAME = "pr-reviewer-tools"
TOOLS_DOCKERFILE = Path("docker") / "tools" / "Dockerfile"
TOOLS_FILES = ("requirements.txt",)

# Where the tools are, in the tools image and in every static sandbox.
TOOLS_DIR = "/opt/tools"

# Where a run keeps its files inside the sandbox (outside the repo).
WORK = "/tmp/pr_reviewer_static"

# Where results are kept, inside the project's data/ folder (not in git).
STATIC = Path("data") / "static"

# The oldest Python version mypy 2.4 accepts as a target. An older environment is checked as
# this version, with its own installed packages.
MYPY_OLDEST = (3, 10)

# How mypy treats imports, in the order tried.
MypyMode = Literal["silent", "skip"]
MYPY_MODES: tuple[MypyMode, ...] = ("silent", "skip")

# How a tool's run ended.
RunStatus = Literal["ok", "failed", "timeout", "out_of_memory"]


class StaticError(Exception):
    """Raised when the tools can't be made ready, or a run gives output that doesn't check out."""


@dataclass(frozen=True)
class StaticSettings:
    """The rules every tool uses, the same for every repo.

    Attributes:
        ruff_select (str): ruff's rule groups.
        bandit_severity (str): bandit's lowest severity reported.
        semgrep_rules (str): Which semgrep rules: the folders under semgrep-rules/python/.
        mypy_flags (tuple[str, ...]): mypy's options besides the environment and imports.
        tool_minutes (float): Most time one tool may run on one PR.
    """

    ruff_select: str = "F,B,PLE"
    bandit_severity: str = "medium"
    semgrep_rules: str = "correctness and security"
    # Most of these repos don't annotate their functions, and mypy skips the inside of an
    # unannotated function by default: with that alone it found nothing on changed lines.
    mypy_flags: tuple[str, ...] = ("--ignore-missing-imports", "--check-untyped-defs")
    tool_minutes: float = 10.0


@dataclass(frozen=True)
class ReadyTools:
    """The tools, ready to attach to sandboxes.

    Attributes:
        image (str): The tools image's name and tag.
        volume (str): The Docker volume holding a copy of its /opt/tools.
        versions (str): The tools' versions and the rules' commit, one per line.
    """

    image: str
    volume: str
    versions: str


class ToolRun(BaseModel):
    """How one tool's run on one PR went.

    Attributes:
        tool (Tool): The tool.
        status (RunStatus): "ok", or why it gave no warnings to read.
        seconds (float): How long it ran (all tries).
        mode (str): For mypy, how it treated imports ("silent" or "skip"); "" otherwise.
        detail (str): The end of its output when it failed.
        warnings (int): Warnings it raised on the changed files.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: Tool
    status: RunStatus
    seconds: float
    mode: str = ""
    detail: str = ""
    warnings: int = 0


class StaticResult(BaseModel):
    """What the tools found on one PR.

    Attributes:
        instance_id (str): The PR.
        version (int): VERSION of the code that made it.
        image (str): The PR image the tools ran in.
        base (str): The base commit.
        head (str): The PR's last commit, where the files were checked.
        changed (dict[str, list[int]]): The lines the PR added or changed, by Python file.
        runs (list[ToolRun]): Each tool's run.
        warnings (list[ToolWarning]): Every warning on the changed files.
        kept (list[ToolWarning]): The warnings that touch a changed line.
        seconds (float): How long the whole run took.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    version: int
    image: str
    base: str = Field(pattern=r"^[0-9a-f]{40}$")
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    changed: dict[str, list[int]]
    runs: list[ToolRun]
    warnings: list[ToolWarning]
    kept: list[ToolWarning]
    seconds: float


def tools_image(config: SandboxConfig) -> ImageSpec:
    """Describe the tools image, built on the current sandbox image.

    Args:
        config (SandboxConfig): The sandbox settings, which name the sandbox image.

    Returns:
        ImageSpec: The image.
    """
    return ImageSpec(
        TOOLS_NAME, project_root() / TOOLS_DOCKERFILE, sandbox_image(config).tag, TOOLS_FILES
    )


def volume_name(tag: str) -> str:
    """Name the volume that holds a tools image's /opt/tools.

    Args:
        tag (str): The tools image's name and tag.

    Returns:
        str: Such as "pr-reviewer-tools-3f1a9c0e2b7d".
    """
    return f"{TOOLS_NAME}-{tag.rsplit(':', 1)[1]}"


def _read_versions(client: DockerClient, image: str, volume: str | None) -> str:
    """Read /opt/tools/VERSION in a short-lived container, network off.

    Args:
        client (DockerClient): Connection to Docker.
        image (str): The image to start (our own: the tools or sandbox image).
        volume (str | None): A volume to attach at /opt/tools, or None.

    Returns:
        str: The file's text, or "" when it's missing or isn't a versions file.
    """
    volumes = {volume: {"bind": TOOLS_DIR, "mode": "rw"}} if volume else None
    try:
        # Not run with remove=True: Docker can remove the container before its output is
        # read, and the output is then an error message instead of the file.
        container = client.containers.run(
            image,
            ["sh", "-c", f"cat {TOOLS_DIR}/VERSION 2>/dev/null; true"],
            user=USER,
            network_mode="none",
            labels={LABEL: "tools-volume"},
            volumes=volumes,
            detach=True,
        )
        try:
            container.wait(timeout=300)
            output = container.logs(stdout=True, stderr=False)
        finally:
            with contextlib.suppress(*DOCKER_ERRORS):
                container.remove(force=True)
    except DOCKER_ERRORS as exc:
        raise StaticError(f"Can't read the tools' versions from {image}: {exc}") from None
    text = bytes(output).decode("utf-8", errors="replace").strip()
    return text if text.startswith("ruff ") else ""


def ready_tools(client: DockerClient, config: SandboxConfig) -> ReadyTools:
    """Build the tools image if it's missing, and fill its volume.

    Docker copies an image's folder into an empty named volume the first time the volume is
    attached there, so starting a tools container with the new volume at /opt/tools fills it.
    The copy is then read back from a plain sandbox container, to be sure it's there.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.

    Returns:
        ReadyTools: The image, the volume and the tools' versions.

    Raises:
        ImageError: If the sandbox image isn't built, or the build fails.
        StaticError: If the volume can't be filled.
    """
    spec = tools_image(config)
    tag = spec.tag
    if not image_exists(client, tag):
        print(f"Building {tag} (the static tools, a few minutes)", flush=True)
        build_image(client, spec)
    volume = volume_name(tag)
    base = sandbox_image(config).tag
    versions = _read_versions(client, base, volume) if _volume_exists(client, volume) else ""
    if not versions:
        with contextlib.suppress(*DOCKER_ERRORS):
            client.volumes.get(volume).remove(force=True)
        try:
            client.volumes.create(volume, labels={"pr-reviewer.tools": tag})
        except DOCKER_ERRORS as exc:
            raise StaticError(f"Can't create the volume {volume}: {exc}") from None
        _read_versions(client, tag, volume)
        versions = _read_versions(client, base, volume)
        if not versions:
            raise StaticError(f"The volume {volume} is still empty after filling it")
    return ReadyTools(tag, volume, versions)


def _volume_exists(client: DockerClient, name: str) -> bool:
    """Tell whether a Docker volume exists.

    Args:
        client (DockerClient): Connection to Docker.
        name (str): The volume's name.

    Returns:
        bool: True when it exists.
    """
    try:
        client.volumes.get(name)
    except docker.errors.NotFound:
        return False
    except DOCKER_ERRORS as exc:
        raise StaticError(f"Can't look up the volume {name}: {exc}") from None
    return True


def settings_key(settings: StaticSettings, tools_tag: str) -> str:
    """Fingerprint what decides a static result: the settings, the code, the tools image.

    Args:
        settings (StaticSettings): The rules.
        tools_tag (str): The tools image's name and tag (it fixes the versions and rules).

    Returns:
        str: 64 hex characters.
    """
    text = json.dumps(
        {"settings": asdict(settings), "version": VERSION, "tools": tools_tag}, sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()


def static_setup(config: SandboxConfig) -> tuple[StaticSettings, str, str]:
    """Give the current static settings, the tools image's tag and their key, without Docker.

    Args:
        config (SandboxConfig): The sandbox settings, which name the sandbox image.

    Returns:
        tuple[StaticSettings, str, str]: The settings, the tools image's name and tag, and
        the settings key.
    """
    settings = StaticSettings()
    tag = tools_image(config).tag
    return settings, tag, settings_key(settings, tag)


def result_path(key: str, instance_id: str) -> Path:
    """Give the file a PR's static result is kept in.

    Args:
        key (str): The settings key.
        instance_id (str): The PR.

    Returns:
        Path: data/static/<first 12 characters of the key>/<instance_id>.json.
    """
    return project_root() / STATIC / key[:12] / f"{instance_id}.json"


def _version_flag(python: str, oldest: tuple[int, int]) -> str:
    """Give a target Python version, raised to the oldest a tool accepts.

    Args:
        python (str): The environment's version, such as "3.8".
        oldest (tuple[int, int]): The oldest version the tool accepts.

    Returns:
        str: Such as "3.9".
    """
    major, minor = (int(part) for part in python.split(".")[:2])
    major, minor = max((major, minor), oldest)
    return f"{major}.{minor}"


def tool_command(
    tool: Tool,
    files: list[str],
    output: str,
    python: str,
    settings: StaticSettings,
    mypy_mode: MypyMode = "silent",
) -> list[str]:
    """Write the command that runs one tool on some files, writing JSON to a file.

    Args:
        tool (Tool): The tool.
        files (list[str]): The files, relative to the repo root (the command's folder).
        output (str): Where the JSON goes, inside the sandbox.
        python (str): The environment's Python version, such as "3.10".
        settings (StaticSettings): The rules.
        mypy_mode (MypyMode): For mypy, how to treat imports.

    Returns:
        list[str]: The command. mypy prints its JSON, so its command sends that to the file.
    """
    bin_dir = f"{TOOLS_DIR}/bin"
    if tool == "ruff":
        version = "py" + _version_flag(python, (3, 7)).replace(".", "")
        return [
            f"{bin_dir}/ruff",
            "check",
            "--isolated",
            "--no-cache",
            "--exit-zero",
            "--select",
            settings.ruff_select,
            "--target-version",
            version,
            "--output-format",
            "json",
            "--output-file",
            output,
            "--",
            *files,
        ]
    if tool == "bandit":
        return [
            f"{bin_dir}/bandit",
            "--format",
            "json",
            "--output",
            output,
            "--severity-level",
            settings.bandit_severity,
            "--quiet",
            "--",
            *files,
        ]
    if tool == "semgrep":
        return [
            f"{bin_dir}/semgrep",
            "scan",
            "--config",
            f"{TOOLS_DIR}/semgrep-rules",
            "--json",
            "--output",
            output,
            "--metrics",
            "off",
            "--disable-version-check",
            "--quiet",
            "--no-git-ignore",
            "--timeout",
            "30",
            "--",
            *files,
        ]
    mypy = [
        f"{bin_dir}/mypy",
        "--config-file",
        f"{TOOLS_DIR}/mypy.ini",
        "--python-executable",
        f"{VENV}/bin/python",
        "--python-version",
        _version_flag(python, MYPY_OLDEST),
        "--follow-imports",
        mypy_mode,
        *settings.mypy_flags,
        "--cache-dir",
        f"{WORK}/mypy-cache",
        "--output",
        "json",
        "--",
        *files,
    ]
    # mypy prints its JSON: send it to the file, and its other messages to the log.
    return ["sh", "-c", 'out="$1"; shift; exec "$@" > "$out"', "sh", output, *mypy]


def _status(result: RunResult, ok_codes: tuple[int, ...]) -> RunStatus:
    """Tell how a tool's run ended.

    Args:
        result (RunResult): The run.
        ok_codes (tuple[int, ...]): Exit codes that mean it finished (with or without
            warnings).

    Returns:
        RunStatus: "ok", "timeout", "out_of_memory" or "failed".
    """
    if result.timed_out:
        return "timeout"
    if result.out_of_memory:
        return "out_of_memory"
    return "ok" if result.exit_code in ok_codes else "failed"


# Exit codes that mean a tool finished: ruff always 0 (--exit-zero); bandit 1 with warnings;
# mypy 1 with errors; semgrep 0 (it only fails on findings with --error).
OK_CODES: dict[Tool, tuple[int, ...]] = {
    "ruff": (0,),
    "mypy": (0, 1),
    "bandit": (0, 1),
    "semgrep": (0,),
}


def _run_tool(
    box: Sandbox,
    tool: Tool,
    root: str,
    files: list[str],
    python: str,
    settings: StaticSettings,
) -> tuple[ToolRun, list[ToolWarning]]:
    """Run one tool on the changed files and read its warnings.

    Args:
        box (Sandbox): The sandbox, network off.
        tool (Tool): The tool.
        root (str): The repo root inside the sandbox.
        files (list[str]): The changed files, relative to the root.
        python (str): The environment's Python version.
        settings (StaticSettings): The rules.

    Returns:
        tuple[ToolRun, list[ToolWarning]]: How the run went, and its warnings (none when it
        failed).

    Raises:
        SandboxError: If Docker or the sandbox fails.
    """
    output = f"{WORK}/{tool}.json"
    timeout = settings.tool_minutes * 60
    modes: tuple[MypyMode, ...] = MYPY_MODES if tool == "mypy" else ("silent",)
    seconds = 0.0
    run = ToolRun(tool=tool, status="failed", seconds=0.0)
    for mode in modes:
        box.run(["rm", "-f", output], timeout=60, label="rm")
        command = tool_command(tool, files, output, python, settings, mode)
        result = box.run(command, timeout=timeout, workdir=root, label=tool)
        seconds += result.seconds
        status = _status(result, OK_CODES[tool])
        detail = "" if status == "ok" else result.output.strip()[-600:]
        run = ToolRun(
            tool=tool,
            status=status,
            seconds=round(seconds, 1),
            mode=mode if tool == "mypy" else "",
            detail=detail,
        )
        if status == "ok":
            break
        if status not in ("timeout", "out_of_memory"):
            # Running again without following imports only helps a run that was too big.
            break
    if run.status != "ok":
        return run, []
    try:
        text = box.read_file(output).decode("utf-8", errors="replace")
        warnings = PARSERS[tool](text, root)
    except (SandboxError, ValueError) as exc:
        failed = run.model_copy(update={"status": "failed", "detail": f"unreadable output: {exc}"})
        return failed, []
    return run.model_copy(update={"warnings": len(warnings)}), warnings


def changed_lines(box: Sandbox, root: str, base: str, head: str) -> dict[str, list[int]]:
    """List the lines a PR added or changed in Python files, as git sees them.

    Args:
        box (Sandbox): The sandbox, with the working copy at `head`.
        root (str): The working copy.
        base (str): The base commit.
        head (str): The PR's last commit.

    Returns:
        dict[str, list[int]]: The new or changed lines, by file, for files that have some
        (a file whose only change is a deletion is left out).

    Raises:
        SandboxError: If git or the sandbox fails.
    """
    made = box.run(["mkdir", "-p", WORK], timeout=60, label="mkdir")
    if not made.ok:
        raise SandboxError(f"Can't make {WORK}: {made.output.strip()[-300:]}")
    diff_file = f"{WORK}/changes.diff"
    diff = box.run(
        [
            "sh",
            "-c",
            'out="$1"; shift; exec "$@" > "$out"',
            "sh",
            diff_file,
            "git",
            "-C",
            root,
            "-c",
            "core.quotePath=false",
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "-M",
            base,
            head,
            "--",
            "*.py",
        ],
        timeout=600,
        label="diff",
    )
    if not diff.ok:
        raise SandboxError(f"git diff failed: {diff.output.strip()[-300:]}")
    changes = parse_diff(box.read_file(diff_file).decode("utf-8", errors="replace"), "new")
    return {path: sorted(info["lines"]) for path, info in sorted(changes.items()) if info["lines"]}


def analyze(
    box: Sandbox,
    root: str,
    changed: dict[str, list[int]],
    python: str,
    settings: StaticSettings,
) -> tuple[list[ToolRun], list[ToolWarning]]:
    """Run every tool on the changed files, in a sandbox that has the tools at /opt/tools.

    Args:
        box (Sandbox): The sandbox, network off, with the working copy at the PR's state.
        root (str): The working copy.
        changed (dict[str, list[int]]): The changed lines, by file (changed_lines()).
        python (str): The environment's Python version.
        settings (StaticSettings): The rules.

    Returns:
        tuple[list[ToolRun], list[ToolWarning]]: Each tool's run, and every warning on the
        changed files. No tool runs when no file changed.

    Raises:
        SandboxError: If Docker or the sandbox fails.
    """
    runs: list[ToolRun] = []
    warnings: list[ToolWarning] = []
    files = sorted(changed)
    if files:
        for tool in TOOLS:
            run, found = _run_tool(box, tool, root, files, python, settings)
            runs.append(run)
            warnings.extend(w for w in found if w.file in changed)
    return runs, warnings


def run_static(
    client: DockerClient,
    config: SandboxConfig,
    tools: ReadyTools,
    pr: Instance,
    image: str,
    python: str,
    settings: StaticSettings,
    log_dir: Path | None = None,
) -> StaticResult:
    """Run every tool on a PR's changed Python files, in a sandbox from its prepared image.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.
        tools (ReadyTools): The tools and their volume.
        pr (Instance): The PR.
        image (str): The PR's prepared image at its PR state.
        python (str): The environment's Python version, from its recipe.
        settings (StaticSettings): The rules.
        log_dir (Path | None): Folder for the sandbox's command logs.

    Returns:
        StaticResult: The warnings and how each tool ran. A PR that changes no Python line
        gets no tool run.

    Raises:
        RepoError, CheckoutError: If the PR's commits can't be checked out.
        SandboxError: If Docker or the sandbox fails.
    """
    started = time.monotonic()
    with Sandbox(
        client, config, image, purpose="static", log_dir=log_dir, volumes={tools.volume: TOOLS_DIR}
    ) as box:
        root, head = checkout(box, pr, "pr")
        base = base_of(box, root, pr)
        changed = changed_lines(box, root, base, head)
        runs, warnings = analyze(box, root, changed, python, settings)
    return StaticResult(
        instance_id=pr.instance_id,
        version=VERSION,
        image=image,
        base=base,
        head=head,
        changed=changed,
        runs=runs,
        warnings=warnings,
        kept=on_changed_lines(warnings, changed),
        seconds=round(time.monotonic() - started, 1),
    )


def save_result(path: Path, result: StaticResult) -> None:
    """Write a static result to its file.

    Args:
        path (Path): The file. Its folder is created if needed.
        result (StaticResult): The result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=1), encoding="utf-8", newline="\n")


def load_result(path: Path) -> StaticResult:
    """Read a static result from its file.

    Args:
        path (Path): The file.

    Returns:
        StaticResult: The checked result.
    """
    return StaticResult.model_validate_json(path.read_text(encoding="utf-8"))
