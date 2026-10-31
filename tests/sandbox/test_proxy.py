"""
Tests for the proxy in sandbox/proxy.py: the only way out of a sandbox whose network is on.

The first tests check the proxy's settings without Docker. The ones marked `docker` start a
test proxy and a sandbox with its network on, and prove what gets through: allowed hosts
over HTTPS (git too), but not other sites, not this machine (the database and pgAdmin), and
nothing directly around the proxy. They need an internet connection, and are skipped when
Docker isn't running or the images aren't built.

How to run (from the project folder):
    pytest tests/sandbox/test_proxy.py
"""

from collections.abc import Iterator

import pytest
from docker import DockerClient

from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.proxy import proxy_environment, tinyproxy_settings
from pr_reviewer.sandbox.runner import Sandbox, can_reach, fetch
from tests.sandbox.conftest import make_test_config


def test_proxy_settings_allow_only_the_listed_hosts() -> None:
    """Check the settings: only the internal network may use it, HTTPS to port 443 only,
    every other host refused, and each allowed host matched exactly."""
    config = make_test_config()
    settings, patterns = tinyproxy_settings(config, "172.30.0.0/16")
    lines = settings.splitlines()
    assert "Allow 172.30.0.0/16" in lines
    assert "ConnectPort 443" in lines
    assert "FilterDefaultDeny Yes" in lines
    assert r"^pypi\.org$" in patterns.splitlines()
    assert len(patterns.splitlines()) == len(config.proxy.allowed_hosts)


def test_sandboxes_are_told_to_use_the_proxy() -> None:
    """Check both spellings of the proxy settings point at the proxy container."""
    env = proxy_environment(make_test_config())
    url = "http://pr-reviewer-proxy-test:8888"
    assert env["HTTPS_PROXY"] == env["https_proxy"] == env["HTTP_PROXY"] == url
    assert env["NO_PROXY"] == "localhost,127.0.0.1"


@pytest.fixture
def online(
    docker: DockerClient, test_config: SandboxConfig, sandbox_image: str
) -> Iterator[Sandbox]:
    """Start a test sandbox with its network on.

    Args:
        docker (DockerClient): Connection to Docker.
        test_config (SandboxConfig): The test config.
        sandbox_image (str): The sandbox image.

    Yields:
        Sandbox: The sandbox, online through the test proxy.
    """
    with Sandbox(docker, test_config, sandbox_image, purpose="test", network=True) as box:
        yield box


@pytest.mark.docker
def test_allowed_hosts_are_reached_through_the_proxy(online: Sandbox) -> None:
    """Check pip's index and git's GitHub work with the network on."""
    assert fetch(online, "https://pypi.org/simple/") == "status 200"
    remote = online.run(
        ["git", "ls-remote", "https://github.com/psf/requests.git", "HEAD"], timeout=120
    )
    assert remote.ok, remote.output
    assert remote.output.split()[1] == "HEAD"


@pytest.mark.docker
def test_other_sites_are_refused(online: Sandbox) -> None:
    """Check a site not on the list is refused by the proxy."""
    assert "403" in fetch(online, "https://example.com/")


@pytest.mark.docker
def test_this_machine_is_out_of_reach(online: Sandbox) -> None:
    """Check the database and pgAdmin on this machine can't be reached, directly or via the
    proxy, although the network is on."""
    for port in (5432, 5050):
        assert not can_reach(online, "host.docker.internal", port)
    assert fetch(online, "http://host.docker.internal:5050/") == "status 403"
    assert fetch(online, "https://host.docker.internal:5432/").startswith("failed")


@pytest.mark.docker
def test_nothing_goes_around_the_proxy(online: Sandbox) -> None:
    """Check a direct connection to the internet fails, even to an allowed host."""
    assert not can_reach(online, "pypi.org", 443)
    assert not can_reach(online, "1.1.1.1", 443)


@pytest.mark.docker
def test_the_proxy_runs_without_privileges(
    docker: DockerClient, test_config: SandboxConfig, online: Sandbox
) -> None:
    """Check the proxy container: no Linux capabilities, read-only files, not root."""
    proxy = docker.containers.get(test_config.proxy.container)
    assert proxy.attrs["HostConfig"]["CapDrop"] == ["ALL"]
    assert proxy.attrs["HostConfig"]["ReadonlyRootfs"] is True
    assert proxy.attrs["Config"]["User"] == "tinyproxy"
