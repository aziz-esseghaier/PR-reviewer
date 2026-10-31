"""
Tests for the sandbox containers in sandbox/runner.py.

The first tests need nothing. The ones marked `docker` start real sandboxes with small limits
and prove each protection: commands run as a user without privileges; the CPU, memory and
process limits are set; a command over its time limit is stopped; a program that uses too
much memory is killed and reported; the network is off until switched on, and on means the
proxy only (that part needs an internet connection); the repo volume is read-only; and a
sandbox stops at the end of its lifetime. They're skipped when Docker isn't running or the
images aren't built. tests/sandbox/test_proxy.py checks what the proxy lets through.

How to run (from the project folder):
    pytest tests/sandbox/test_runner.py
"""

import io
import tarfile
import time
from pathlib import Path

import pytest
from docker import DockerClient

from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.runner import (
    RunResult,
    Sandbox,
    SandboxError,
    _tar,
    can_reach,
    fetch,
    remove_leftovers,
)
from tests.sandbox.conftest import TEST_LIMITS, TEST_PREFIX


def result(exit_code: int = 0, *, timed_out: bool = False, oom: bool = False) -> RunResult:
    """Make a command result.

    Args:
        exit_code (int): The exit code.
        timed_out (bool): Whether it went over its time limit.
        oom (bool): Whether it ran out of memory.

    Returns:
        RunResult: The result.
    """
    return RunResult(("true",), exit_code, 1.25, timed_out, oom, "", None)


def test_a_result_is_ok_only_when_it_ends_well_in_time_and_memory() -> None:
    """Check `ok` and the short description of each way a command can end."""
    assert result().ok
    assert result().describe() == "exit code 0 after 1.2 s"
    assert not result(1).ok
    assert not result(124, timed_out=True).ok
    assert result(124, timed_out=True).describe().startswith("stopped: over its time limit")
    assert not result(137, oom=True).ok
    assert result(137, oom=True).describe() == "stopped: out of memory (exit code 137)"


def test_files_are_packed_for_the_sandbox_user() -> None:
    """Check that written files belong to the sandbox user (uid 1000), not root."""
    with tarfile.open(fileobj=io.BytesIO(_tar({"a.txt": "hi", "b.bin": b"\x00"}))) as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert set(members) == {"a.txt", "b.bin"}
    assert members["a.txt"].uid == members["a.txt"].gid == 1000
    assert members["a.txt"].size == 2


@pytest.mark.docker
def test_commands_run_as_a_user_without_privileges(sandbox: Sandbox) -> None:
    """Check the user is uid 1000 and holds no Linux capabilities."""
    assert sandbox.run(["id", "-u"], timeout=30).output.strip() == "1000"
    status = sandbox.run(["cat", "/proc/self/status"], timeout=30).output
    assert "CapEff:\t0000000000000000" in status


@pytest.mark.docker
def test_cpu_memory_and_process_limits_are_set(sandbox: Sandbox) -> None:
    """Check the kernel's limits match the config: 1 CPU, 256 MB without swap, 64 processes."""

    def read(name: str) -> str:
        return sandbox.run(["cat", f"/sys/fs/cgroup/{name}"], timeout=30).output.strip()

    assert read("cpu.max") == "100000 100000"
    assert read("memory.max") == str(256 * 1024 * 1024)
    assert read("memory.swap.max") == "0"
    assert read("pids.max") == str(TEST_LIMITS.processes)


@pytest.mark.docker
def test_a_command_over_its_time_limit_is_stopped(sandbox: Sandbox) -> None:
    """Check that `sleep 60` with a 1-second limit stops after about a second."""
    stopped = sandbox.run(["sleep", "60"], timeout=1)
    assert stopped.timed_out
    assert not stopped.ok
    assert stopped.seconds < 15


@pytest.mark.docker
def test_a_program_using_too_much_memory_is_killed(sandbox: Sandbox) -> None:
    """Check that allocating 600 MB in a 256 MB sandbox is killed and reported as such."""
    hog = sandbox.run(["python3", "-c", "x = bytearray(600 * 1024 * 1024)"], timeout=60)
    assert hog.out_of_memory
    assert not hog.timed_out
    after = sandbox.run(["true"], timeout=30)
    assert after.ok, "a later command must not be blamed for the earlier kill"


@pytest.mark.docker
def test_too_many_processes_are_refused(sandbox: Sandbox) -> None:
    """Check that starting 100 processes in a sandbox allowed 64 fails."""
    script = "import subprocess\nps = [subprocess.Popen(['sleep', '5']) for _ in range(100)]"
    spawned = sandbox.run(["python3", "-c", script], timeout=60)
    assert not spawned.ok
    assert "Resource temporarily unavailable" in spawned.output


@pytest.mark.docker
def test_the_network_is_off_until_switched_on(sandbox: Sandbox, test_config: SandboxConfig) -> None:
    """Check the sandbox starts with no way out, gets the proxy when on, and loses it again.

    Needs an internet connection to pypi.org for the online part.
    """
    proxy = (test_config.proxy.container, test_config.proxy.port)
    assert not sandbox.online
    assert not can_reach(sandbox, *proxy)
    assert fetch(sandbox, "https://pypi.org/simple/").startswith("failed")
    sandbox.set_network(True)
    assert can_reach(sandbox, *proxy)
    assert fetch(sandbox, "https://pypi.org/simple/") == "status 200"
    sandbox.set_network(False)
    assert not can_reach(sandbox, *proxy)
    assert fetch(sandbox, "https://pypi.org/simple/").startswith("failed")


@pytest.mark.docker
def test_the_repo_volume_is_read_only(sandbox: Sandbox) -> None:
    """Check that nothing in the sandbox can write to the shared repo copies."""
    touched = sandbox.run(["touch", "/repos/x"], timeout=30)
    assert not touched.ok
    assert "Read-only file system" in touched.output


@pytest.mark.docker
def test_written_files_belong_to_the_sandbox_user(sandbox: Sandbox) -> None:
    """Check write_files() puts files the sandbox user can read and change."""
    sandbox.write_files("/work", {"note.txt": "hello"})
    assert sandbox.run(["cat", "/work/note.txt"], timeout=30).output == "hello"
    assert sandbox.run(["stat", "-c", "%U", "/work/note.txt"], timeout=30).output.strip() == (
        "sandbox"
    )


@pytest.mark.docker
def test_each_command_is_logged(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str, tmp_path: Path
) -> None:
    """Check that a command's line, output and outcome go to its own log file."""
    with Sandbox(docker, test_config, sandbox_image, purpose="test", log_dir=tmp_path) as box:
        done = box.run(["echo", "hi there"], timeout=30, label="greet")
    assert done.log_path is not None
    assert done.log_path.name == "001-greet.log"
    log = done.log_path.read_text(encoding="utf-8")
    assert log.startswith("$ echo 'hi there'")
    assert "hi there\n" in log
    assert log.rstrip().endswith("]")


@pytest.mark.docker
def test_a_sandbox_stops_at_the_end_of_its_lifetime(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> None:
    """Check that a sandbox with a 20-second lifetime refuses commands after it."""
    short = TEST_LIMITS.model_copy(update={"lifetime_minutes": 20 / 60})
    with Sandbox(docker, test_config, sandbox_image, purpose="test", limits=short) as box:
        # The lifetime counts from the container's start, which came before this point.
        assert box.run(["true"], timeout=30).ok
        time.sleep(25)
        with pytest.raises(SandboxError, match="has stopped"):
            box.run(["true"], timeout=30)


@pytest.mark.docker
def test_leftover_sandboxes_are_removed(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> None:
    """Check that a sandbox left behind (never closed) is found and removed."""
    left = Sandbox(docker, test_config, sandbox_image, purpose="test-leftover")
    left.start()
    assert remove_leftovers(docker, prefix=f"{TEST_PREFIX}-leftover") == 1
    assert remove_leftovers(docker, prefix=f"{TEST_PREFIX}-leftover") == 0
