"""Shared fixtures for the sandbox tests.

- `docker`: a connection to Docker. Tests that use it are skipped when Docker isn't running.
- `test_config`: the real sandbox config, but with its own networks, proxy and repo volume
  (removed after the tests, so the real ones are never touched) and small limits, so the
  limit tests finish in seconds.
- `sandbox_image`: the built sandbox image. Tests that use it are skipped until both images
  are built (pr-reviewer sandbox build), because the first build takes about 11 minutes.
- `sandbox`: a fresh sandbox from that image, removed after the test.

pytest loads this file on its own; it isn't run directly.
"""

import contextlib
from collections.abc import Iterator

import docker as docker_sdk
import pytest
from docker import DockerClient

from pr_reviewer.sandbox import image as images
from pr_reviewer.sandbox.config import Limits, SandboxConfig, load_sandbox_config
from pr_reviewer.sandbox.proxy import remove_proxy
from pr_reviewer.sandbox.runner import Sandbox, SandboxError, docker_client, remove_leftovers

# Every test sandbox's name starts with this, so cleanup never touches a real one.
TEST_PREFIX = "pr-reviewer-test"

# Small limits: one CPU, 256 MB, 64 processes, 10 minutes.
TEST_LIMITS = Limits(cpus=1, memory_gb=0.25, processes=64, lifetime_minutes=10)


@pytest.fixture(scope="session")
def docker() -> Iterator[DockerClient]:
    """Connect to Docker, or skip the test when it isn't running.

    After the last test, removes the test sandboxes, the test repo volume and the test network.

    Yields:
        DockerClient: The connection.
    """
    try:
        client = docker_client()
    except SandboxError:
        pytest.skip("Docker isn't running")
    yield client
    remove_leftovers(client, prefix=TEST_PREFIX)
    config = make_test_config()
    remove_proxy(client, config)
    with contextlib.suppress(docker_sdk.errors.NotFound):
        client.volumes.get(config.repos_volume).remove(force=True)
    for network in (config.network, config.proxy.egress_network):
        with contextlib.suppress(docker_sdk.errors.NotFound):
            client.networks.get(network).remove()


def make_test_config() -> SandboxConfig:
    """Give the real sandbox config with test names and small limits.

    The test sandboxes get their own repo volume, networks and proxy, so the real ones are
    never touched, and everything can be removed after the tests.

    Returns:
        SandboxConfig: The test config.
    """
    real = load_sandbox_config()
    proxy = real.proxy.model_copy(
        update={
            "container": f"{real.proxy.container}-test",
            "egress_network": f"{real.proxy.egress_network}-test",
        }
    )
    return real.model_copy(
        update={
            "network": f"{real.network}-test",
            "repos_volume": f"{real.repos_volume}-test",
            "proxy": proxy,
            "limits": TEST_LIMITS,
        }
    )


@pytest.fixture(scope="session")
def test_config() -> SandboxConfig:
    """Give the sandbox config for tests (see make_test_config).

    Returns:
        SandboxConfig: The test config.
    """
    return make_test_config()


@pytest.fixture(scope="session")
def sandbox_image(docker: DockerClient, test_config: SandboxConfig) -> str:
    """Give the built sandbox image, or skip the test until both images are built.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config, which names the images.

    Returns:
        str: The sandbox image's name and tag.
    """
    sandbox = images.sandbox_image(test_config)
    for spec in (sandbox, images.proxy_image(test_config)):
        if not images.image_exists(docker, spec.tag):
            pytest.skip(f"Build the images first: pr-reviewer sandbox build ({spec.tag})")
    return sandbox.tag


@pytest.fixture
def sandbox(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> Iterator[Sandbox]:
    """Start a fresh test sandbox, network off, and remove it after the test.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.

    Yields:
        Sandbox: The started sandbox.
    """
    with Sandbox(docker, test_config, sandbox_image, purpose="test") as started:
        yield started
