"""Sandbox containers: where untrusted PR code runs, with a limit on everything it can use.

A Sandbox is one Docker container started from the sandbox image (sandbox/image.py). Every one
gets:
- the limits from configs/sandbox.yaml: CPUs, memory with no swap, processes, and a lifetime
  after which the container stops itself, even if our program crashed. The container is also
  pinned to that many CPUs, so programs inside see only those (pinned_cpus());
- no Linux privileges: all capabilities dropped, no gaining privileges through setuid
  programs, and commands run as the non-root user "sandbox";
- nothing from this machine: no Windows folders, no Docker socket, no settings from .env. The
  only shared data is the repo volume (sandbox/repos.py), attached read-only unless the
  sandbox is the one that downloads the repos;
- a network switch, off at the start. Off, the container has nothing but its own loopback
  interface, so even name lookups fail. On, it joins an internal network whose only way out is
  the proxy (sandbox/proxy.py), which lets through only the hosts allowed in
  configs/sandbox.yaml, such as PyPI and GitHub, and never this machine.

Each command has its own time limit, enforced inside the container by `timeout` and from here
as a backstop, and its output is saved to a log file. The result says whether the command ran
out of time or memory: the kernel counts every process it kills for memory in the container's
memory.events file.

Imported by other modules; the file isn't run directly.
"""

import contextlib
import io
import math
import shlex
import tarfile
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import docker
from docker import DockerClient
from docker.models.containers import Container
from docker.models.networks import Network

from pr_reviewer.sandbox.config import LABEL, Limits, SandboxConfig
from pr_reviewer.sandbox.proxy import ensure_network, proxy_environment

# What a failed Docker call raises: the SDK's own errors, or, when a call goes over the
# client's time limit or loses its connection, the requests library's, which are OSErrors.
DOCKER_ERRORS = (docker.errors.DockerException, OSError)

# The user commands run as: "sandbox" in the image, not root.
USER = "1000:1000"

# Where the shared repo copies are attached inside a sandbox.
REPOS_MOUNT = "/repos"

# Exit codes of `timeout`: 124 when it stopped the command, 137 when it had to kill it.
TIMEOUT_EXIT = 124
KILLED_EXIT = 137

# After a command's time limit, `timeout` waits this long before killing it outright.
KILL_AFTER_SECONDS = 10

# How much longer than its limit we wait for a command before stopping the whole container.
GRACE_SECONDS = 30

# Most output kept in memory per command. The log file keeps all of it.
OUTPUT_LIMIT = 1_000_000

# Biggest file read_file() copies out of a sandbox.
READ_LIMIT = 50_000_000

# The kernel counts memory kills here (cgroup v2, seen from inside the container).
MEMORY_EVENTS = "/sys/fs/cgroup/memory.events"

# Seconds a call to Docker (other than a command's output) may take.
DOCKER_TIMEOUT = 600


class SandboxError(Exception):
    """Raised when Docker can't be reached, or a sandbox can't run a command."""


@dataclass(frozen=True)
class RunResult:
    """What happened when a command ran in a sandbox.

    Attributes:
        command (tuple[str, ...]): The command, as given.
        exit_code (int): Its exit code. 124 or 137 when it was stopped for running too long.
        seconds (float): How long it ran.
        timed_out (bool): Whether it was stopped for going over its time limit.
        out_of_memory (bool): Whether the kernel killed a process for using too much memory.
        output (str): What it printed (stdout and stderr together). Only the end is kept when
            it printed more than OUTPUT_LIMIT bytes.
        log_path (Path | None): The log file with all of its output, if logs are kept.
    """

    command: tuple[str, ...]
    exit_code: int
    seconds: float
    timed_out: bool
    out_of_memory: bool
    output: str
    log_path: Path | None

    @property
    def ok(self) -> bool:
        """Whether the command succeeded: exit code 0, in time and memory."""
        return self.exit_code == 0 and not self.timed_out and not self.out_of_memory

    def describe(self) -> str:
        """Say in a few words how the command ended.

        Returns:
            str: Such as "exit code 0 after 1.2 s" or "stopped: over its time limit".
        """
        if self.timed_out:
            return f"stopped: over its time limit ({self.seconds:.0f} s)"
        if self.out_of_memory:
            return f"stopped: out of memory (exit code {self.exit_code})"
        return f"exit code {self.exit_code} after {self.seconds:.1f} s"


def docker_client() -> DockerClient:
    """Connect to Docker.

    Returns:
        DockerClient: The connection.

    Raises:
        SandboxError: If Docker isn't running or can't be reached.
    """
    try:
        # Saving or removing a big sandbox can take minutes on a slow disk; the SDK's default
        # 60 seconds isn't enough. Command output streams don't use this limit.
        client = docker.from_env(timeout=DOCKER_TIMEOUT)
        client.ping()
    except DOCKER_ERRORS as exc:
        raise SandboxError(f"Can't reach Docker ({exc}). Is Docker Desktop running?") from None
    return client


def remove_leftovers(client: DockerClient, prefix: str = "pr-reviewer-") -> int:
    """Remove sandbox containers still around, running or stopped.

    Sandboxes remove themselves when closed. One is left behind only when our program was
    killed, and it stops by itself at the end of its lifetime.

    Args:
        client (DockerClient): Connection to Docker.
        prefix (str): Only remove sandboxes whose name starts with this. The default matches
            them all.

    Returns:
        int: How many containers were removed.
    """
    containers = [
        container
        for container in client.containers.list(all=True, filters={"label": LABEL})
        if (container.name or "").startswith(prefix)
    ]
    for container in containers:
        container.remove(force=True)
    return len(containers)


# Tries to open a connection; prints "reachable" or the reason it failed.
_REACH_SCRIPT = """
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=float(sys.argv[3])).close()
    print("reachable")
except OSError as exc:
    print("unreachable:", exc)
"""


def can_reach(sandbox: "Sandbox", host: str = "github.com", port: int = 443) -> bool:
    """Try to open a direct connection from inside a sandbox, without the proxy.

    Args:
        sandbox (Sandbox): The sandbox to try from.
        host (str): Host to connect to.
        port (int): Port to connect to.

    Returns:
        bool: True when the connection opened.
    """
    result = sandbox.run(
        ["python3", "-c", _REACH_SCRIPT, host, str(port), "5"], timeout=30, label="reach"
    )
    return result.ok and result.output.strip() == "reachable"


# Downloads a page the way tools do, through the proxy settings; prints the HTTP status, or
# the reason it failed.
_FETCH_SCRIPT = """
import sys, urllib.error, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as response:
        print("status", response.status)
except urllib.error.HTTPError as exc:
    print("status", exc.code)
except Exception as exc:
    print("failed:", exc)
"""


def fetch(sandbox: "Sandbox", url: str) -> str:
    """Download a page from inside a sandbox, through the proxy like pip or git would.

    Args:
        sandbox (Sandbox): The sandbox to try from.
        url (str): The page, such as "https://pypi.org/simple/".

    Returns:
        str: "status <code>" when an answer came back, otherwise "failed: <reason>". For
        HTTPS, an answer can only come from the site itself, never from the proxy.
    """
    result = sandbox.run(["python3", "-c", _FETCH_SCRIPT, url, "20"], timeout=60, label="fetch")
    return result.output.strip() or result.describe()


def _tar(files: Mapping[str, bytes | str]) -> bytes:
    """Pack files into a tar archive owned by the sandbox user.

    Args:
        files (Mapping[str, bytes | str]): File contents by relative path.

    Returns:
        bytes: The archive.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.uid = info.gid = 1000
            info.mtime = int(time.time())
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def pinned_cpus(machine_cpus: int, limit: float) -> str:
    """Choose the CPUs a sandbox runs on: the machine's last ones, as many as its CPU limit.

    A CPU limit alone still shows every CPU of the machine to the programs inside. Build tools
    such as ninja and make then start a compile job per CPU, and together the jobs run out of
    memory (10 jobs in 2 GB on an 8-CPU machine). Pinned, `nproc` shows only the sandbox's
    CPUs. The last ones are used, since the first does most of the system's own work.

    Args:
        machine_cpus (int): CPUs of the machine Docker runs on (its VM, with Docker Desktop).
        limit (float): The sandbox's CPU limit, which may be a fraction, such as 1.5.

    Returns:
        str: The CPU numbers in Docker's format, such as "6,7".
    """
    count = max(1, min(machine_cpus, math.ceil(limit)))
    return ",".join(str(cpu) for cpu in range(machine_cpus - count, machine_cpus))


class Sandbox:
    """One sandbox container: create it, run commands in it, switch its network, remove it.

    Use it in a `with` block, so the container is removed even when something fails:

        with Sandbox(client, config, image, purpose="check") as sandbox:
            result = sandbox.run(["git", "--version"], timeout=60)

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings: network, repo volume and limits.
        image (str): The sandbox image's name and tag (see sandbox/image.py).
        purpose (str): A short word for what the sandbox is for, used in its name and logs.
        network (bool): Whether the network starts on. Off unless a step needs it.
        repos_writable (bool): Attach the repo volume writable. Only for the sandbox that
            downloads the repos, which runs no PR code.
        log_dir (Path | None): Folder for one log file per command. None keeps no logs.
        limits (Limits | None): Limits to use instead of the config's, such as in tests.
    """

    def __init__(
        self,
        client: DockerClient,
        config: SandboxConfig,
        image: str,
        *,
        purpose: str,
        network: bool = False,
        repos_writable: bool = False,
        log_dir: Path | None = None,
        limits: Limits | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._image = image
        self._limits = limits or config.limits
        self._repos_writable = repos_writable
        self._start_online = network
        self.name = f"pr-reviewer-{purpose}-{uuid.uuid4().hex[:8]}"
        self.log_dir = log_dir / self.name if log_dir is not None else None
        self._container: Container | None = None
        self._network: Network | None = None
        self._online = False
        self._commands = 0
        self._memory_kills = 0

    def __enter__(self) -> Self:
        """Create and start the container.

        Returns:
            Self: The started sandbox.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove the container, whatever happened inside the block."""
        self.close()

    @property
    def online(self) -> bool:
        """Whether the network is on."""
        return self._online

    def start(self) -> None:
        """Create and start the container with all its limits, network off unless asked.

        Raises:
            SandboxError: If Docker can't create or start it.
        """
        limits = self._limits
        memory = f"{round(limits.memory_gb * 1024)}m"
        mode = "rw" if self._repos_writable else "ro"
        try:
            self._network = ensure_network(self._client, self._config)
            self._container = self._client.containers.create(
                self._image,
                # The container's main process. When it ends, the container stops.
                command=["sleep", str(round(limits.lifetime_minutes * 60))],
                name=self.name,
                hostname="sandbox",
                labels={LABEL: self.name},
                user=USER,
                working_dir="/work",
                # Tools reach the internet only through the proxy, and only while online.
                environment=proxy_environment(self._config),
                nano_cpus=round(limits.cpus * 1e9),
                cpuset_cpus=pinned_cpus(self._client.info()["NCPU"], limits.cpus),
                mem_limit=memory,
                memswap_limit=memory,
                pids_limit=limits.processes,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                init=True,
                network=self._config.network,
                volumes={self._config.repos_volume: {"bind": REPOS_MOUNT, "mode": mode}},
            )
            self._container.start()
            self._online = True
            if not self._start_online:
                self.set_network(False)
            self._memory_kills = self._read_memory_kills()
        except DOCKER_ERRORS as exc:
            with contextlib.suppress(SandboxError):
                self.close()
            raise SandboxError(f"Can't start a sandbox from {self._image}: {exc}") from None
        except SandboxError:
            with contextlib.suppress(SandboxError):
                self.close()
            raise
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Remove the container and everything in it. Safe to call more than once.

        Raises:
            SandboxError: If Docker can't remove it. It then stops at the end of its lifetime,
                and the next run's remove_leftovers() removes it.
        """
        if self._container is not None:
            container, self._container = self._container, None
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except DOCKER_ERRORS as exc:
                raise SandboxError(f"Can't remove the sandbox {self.name}: {exc}") from None

    def set_network(self, on: bool) -> None:
        """Turn the network on or off.

        Args:
            on (bool): True to join the private network, False to leave it.

        Raises:
            SandboxError: If Docker can't switch it.
        """
        if on == self._online:
            return
        container, network = self._live(), self._network
        assert network is not None
        try:
            if on:
                network.connect(container)
            else:
                network.disconnect(container, force=True)
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't turn the network {'on' if on else 'off'}: {exc}") from None
        self._online = on

    def run(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        label: str = "",
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        """Run one command in the sandbox and wait for it, within a time limit.

        The command is a list of arguments, run directly without a shell, so nothing in it
        needs quoting.

        Args:
            command (Sequence[str]): The program and its arguments.
            timeout (float): Seconds it may run before it's stopped.
            label (str): A short word for the log file's name, such as "checkout".
            workdir (str | None): Folder inside the sandbox to run it in. None means /work.
            env (Mapping[str, str] | None): Extra environment variables for this command.

        Returns:
            RunResult: Exit code, output, time, and whether it ran out of time or memory.

        Raises:
            SandboxError: If the sandbox has stopped, or the command overran even the backstop,
                in which case the whole container is stopped.
        """
        container = self._live()
        command = tuple(command)
        wrapped = ["timeout", f"--kill-after={KILL_AFTER_SECONDS}s", f"{timeout:g}s", *command]
        self._commands += 1
        log_path = None
        if self.log_dir is not None:
            name = f"{self._commands:03d}-{label or command[0].rsplit('/', 1)[-1]}.log"
            log_path = self.log_dir / name
        started = time.monotonic()
        api = self._client.api
        try:
            exec_id = api.exec_create(
                container.id,
                wrapped,
                stdout=True,
                stderr=True,
                user=USER,
                workdir=workdir,
                environment=dict(env) if env else None,
            )["Id"]
            stream = api.exec_start(exec_id, stream=True)
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't run {shlex.join(command)}: {exc}") from None
        output = bytearray()
        reader = threading.Thread(
            target=self._collect,
            args=(stream, output, log_path, command, timeout),
            daemon=True,
        )
        reader.start()
        reader.join(timeout + KILL_AFTER_SECONDS + GRACE_SECONDS)
        if reader.is_alive():
            self.close()
            raise SandboxError(
                f"{shlex.join(command)} kept running past its {timeout:g} s limit; "
                "the sandbox was stopped"
            )
        seconds = time.monotonic() - started
        try:
            exit_code = int(api.exec_inspect(exec_id)["ExitCode"])
            kills = self._read_memory_kills()
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't tell how {shlex.join(command)} ended: {exc}") from None
        out_of_memory = kills > self._memory_kills
        self._memory_kills = kills
        timed_out = not out_of_memory and (
            exit_code == TIMEOUT_EXIT or (exit_code == KILLED_EXIT and seconds >= timeout)
        )
        result = RunResult(
            command=command,
            exit_code=exit_code,
            seconds=seconds,
            timed_out=timed_out,
            out_of_memory=out_of_memory,
            output=output.decode("utf-8", errors="replace"),
            log_path=log_path,
        )
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{result.describe()}]\n")
        return result

    def save_image(self, repository: str, tag: str, labels: Mapping[str, str]) -> str:
        """Save the sandbox's current files as a new image, to start later sandboxes from.

        The repo volume isn't part of it (volumes never are), and the network state isn't
        either: a sandbox started from the image begins with the network off, like any other.
        The sandbox isn't paused while it's saved, so nothing should be running in it then.

        Args:
            repository (str): The image name, such as "pr-reviewer-env".
            tag (str): The image tag.
            labels (Mapping[str, str]): Labels to put on the image.

        Returns:
            str: The image name and tag.

        Raises:
            SandboxError: If Docker can't save it, or takes too long.
        """
        container = self._live()
        changes = [f"LABEL {key}={value}" for key, value in labels.items()]
        try:
            # Docker once stayed stuck for ten minutes saving a paused sandbox, and pausing an
            # idle one protects nothing.
            container.commit(repository=repository, tag=tag, changes=changes, pause=False)
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't save {self.name} as an image: {exc}") from None
        return f"{repository}:{tag}"

    def write_files(self, folder: str, files: Mapping[str, bytes | str]) -> None:
        """Put files into the sandbox, owned by the sandbox user.

        Args:
            folder (str): An existing folder inside the sandbox, such as "/work".
            files (Mapping[str, bytes | str]): File contents by path relative to `folder`.

        Raises:
            SandboxError: If the folder doesn't exist or Docker refuses the files.
        """
        try:
            ok = self._live().put_archive(folder, _tar(files))
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't write files to {folder}: {exc}") from None
        if not ok:
            raise SandboxError(f"Can't write files to {folder}")

    def read_file(self, path: str, max_bytes: int = READ_LIMIT) -> bytes:
        """Copy one file out of the sandbox.

        Unlike a command's output, the file isn't cut at OUTPUT_LIMIT.

        Args:
            path (str): The file's absolute path inside the sandbox.
            max_bytes (int): Most bytes accepted; a bigger file is refused.

        Returns:
            bytes: The file's content.

        Raises:
            SandboxError: If the file is missing, isn't a regular file, is too big, or Docker
                refuses.
        """
        try:
            stream, stat = self._live().get_archive(path)
            if int((stat or {}).get("size", 0)) > max_bytes:
                raise SandboxError(f"{path} is bigger than {max_bytes:,} bytes")
            archive = b"".join(stream)
        except docker.errors.NotFound:
            raise SandboxError(f"{path} isn't in the sandbox") from None
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't read {path}: {exc}") from None
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            member = tar.next()
            if member is None or not member.isfile():
                raise SandboxError(f"{path} isn't a regular file")
            extracted = tar.extractfile(member)
            assert extracted is not None
            return extracted.read()

    def _live(self) -> Container:
        """Get the container, checking it's still running.

        Returns:
            Container: The running container.

        Raises:
            SandboxError: If the sandbox was closed, or stopped at the end of its lifetime.
        """
        if self._container is None:
            raise SandboxError("The sandbox is closed")
        try:
            self._container.reload()
        except DOCKER_ERRORS as exc:
            raise SandboxError(f"Can't reach the sandbox {self.name}: {exc}") from None
        if self._container.status != "running":
            raise SandboxError(
                f"The sandbox {self.name} has stopped ({self._container.status}); it may have "
                "reached the end of its lifetime"
            )
        return self._container

    def _read_memory_kills(self) -> int:
        """Count the processes the kernel has killed in this container for using too much memory.

        Returns:
            int: The count so far, or 0 if the container doesn't show it.
        """
        api = self._client.api
        exec_id = api.exec_create(self._live().id, ["cat", MEMORY_EVENTS], user=USER)["Id"]
        text = api.exec_start(exec_id).decode("utf-8", errors="replace")
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            if key == "oom_kill" and value.strip().isdigit():
                return int(value)
        return 0

    @staticmethod
    def _collect(
        stream: Iterator[bytes],
        output: bytearray,
        log_path: Path | None,
        command: tuple[str, ...],
        timeout: float,
    ) -> None:
        """Read a command's output as it comes, into memory and the log file.

        Args:
            stream (Iterator[bytes]): The command's output, piece by piece.
            output (bytearray): Collects the output; trimmed from the front past OUTPUT_LIMIT.
            log_path (Path | None): The log file to write, if any.
            command (tuple[str, ...]): The command, for the log's first line.
            timeout (float): Its time limit, for the log's first line.
        """
        log = log_path.open("wb") if log_path is not None else None
        try:
            if log is not None:
                log.write(f"$ {shlex.join(command)}   [limit {timeout:g} s]\n".encode())
            for chunk in stream:
                if log is not None:
                    log.write(chunk)
                output.extend(chunk)
                if len(output) > OUTPUT_LIMIT:
                    del output[: len(output) - OUTPUT_LIMIT]
        finally:
            if log is not None:
                log.close()
