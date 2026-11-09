"""Tests for answering lookups in a real indexer sandbox (context/lookups.py).

Makes a small repo inside an indexer sandbox, then answers lookups there as a review does:
the lookup tool and the probe it reuses are copied in, run with the network off, and their
answer comes back as a file. Marked `docker`: skipped when Docker isn't running or the
sandbox image isn't built (the indexer image is built on it in seconds).

How to run (from the project folder):
    pytest tests/context/test_lookups_docker.py
"""

from collections.abc import Iterator

import pytest
from docker import DockerClient

from pr_reviewer.context.lookups import Lookup, LookupLimits, answer_in_sandbox
from pr_reviewer.context.retrieve import ready_indexer
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.runner import Sandbox
from tests.context.test_lookups import FILES

pytestmark = pytest.mark.docker

REPO = "/work/repo"


@pytest.fixture
def box(docker: DockerClient, test_config: SandboxConfig, sandbox_image: str) -> Iterator[Sandbox]:
    """Start an indexer sandbox, network off, and remove it after the test.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image (skips the test when it isn't built).

    Yields:
        Sandbox: The sandbox.
    """
    image = ready_indexer(docker, test_config)
    with Sandbox(docker, test_config, image, purpose="test-lookup") as started:
        yield started


def test_lookups_are_answered_inside_the_sandbox(box: Sandbox) -> None:
    """Check read_file and search_symbol answers come back from the sandbox, in order."""
    assert box.run(["mkdir", "-p", f"{REPO}/pkg", f"{REPO}/tests"], timeout=60).ok
    box.write_files(REPO, FILES)
    lookups = [
        Lookup(tool="search_symbol", symbol="total"),
        Lookup(tool="read_file", path="pkg/use.py", start_line=4, end_line=5),
    ]
    answers = answer_in_sandbox(box, REPO, lookups, LookupLimits(), 300)
    assert answers[0]["ok"] and answers[0]["text"].startswith("search_symbol total: 2 definitions")
    assert answers[1]["text"].startswith("read_file pkg/use.py, lines 4-5 of 5:")
