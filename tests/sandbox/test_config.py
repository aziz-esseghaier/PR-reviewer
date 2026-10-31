"""
Tests for the sandbox settings (sandbox/config.py) and the image tag (sandbox/image.py).

Checks the project's configs/sandbox.yaml holds the limits agreed for the sandboxes, that a
bad file is refused without repeating its values, and that the image tag follows the
Dockerfile's content. No Docker needed.

How to run (from the project folder):
    pytest tests/sandbox/test_config.py
"""

from pathlib import Path

import pytest

from pr_reviewer.core.config import ConfigError
from pr_reviewer.sandbox.config import load_sandbox_config
from pr_reviewer.sandbox.image import ImageSpec

GOOD = """
image: pr-reviewer-sandbox
network: pr-reviewer-sandbox
repos_volume: pr-reviewer-repos
proxy:
  image: pr-reviewer-proxy
  container: pr-reviewer-proxy
  egress_network: pr-reviewer-egress
  port: 8888
  allowed_hosts: [pypi.org, github.com]
limits: {cpus: 2, memory_gb: 2, processes: 512, lifetime_minutes: 120}
timeouts: {install_minutes: 30, test_minutes: 15, download_minutes: 60}
"""


def test_project_config_holds_the_agreed_limits() -> None:
    """Check the real config: 2 CPUs, 2 GB, 512 processes, 2 hours, 30/15/60 minutes."""
    config = load_sandbox_config()
    limits, timeouts = config.limits, config.timeouts
    assert (limits.cpus, limits.memory_gb, limits.processes) == (2, 2, 512)
    assert limits.lifetime_minutes == 120
    assert (timeouts.install_minutes, timeouts.test_minutes) == (30, 15)
    assert timeouts.download_minutes == 60


def test_bad_config_is_refused_without_repeating_values(tmp_path: Path) -> None:
    """Check a wrong limit and an unknown key are named, but their values aren't shown."""
    path = tmp_path / "sandbox.yaml"
    path.write_text(
        GOOD.replace("memory_gb: 2", "memory_gb: -1") + "token: sk-secret-123\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError) as caught:
        load_sandbox_config(path)
    message = str(caught.value)
    assert "limits.memory_gb" in message
    assert "token: Extra inputs are not permitted" in message
    assert "sk-secret-123" not in message


def test_missing_config_is_refused(tmp_path: Path) -> None:
    """Check a missing file gives a clear error."""
    with pytest.raises(ConfigError, match="not found"):
        load_sandbox_config(tmp_path / "none.yaml")


@pytest.mark.parametrize(
    "host", ["PyPI.org", "pypi.org:443", "*.github.com", "https://pypi.org", "localhost"]
)
def test_allowed_hosts_must_be_plain_host_names(tmp_path: Path, host: str) -> None:
    """Check ports, wildcards, addresses and single names can't slip into the allow-list."""
    path = tmp_path / "sandbox.yaml"
    path.write_text(GOOD.replace("[pypi.org, github.com]", f"[github.com, '{host}']"), "utf-8")
    with pytest.raises(ConfigError, match=r"proxy.allowed_hosts.*\(entry 2 isn't\)"):
        load_sandbox_config(path)


def test_image_tag_follows_the_dockerfile(tmp_path: Path) -> None:
    """Check an edited Dockerfile gets a new tag, and the same content the same tag."""
    first, second = tmp_path / "a" / "Dockerfile", tmp_path / "b" / "Dockerfile"
    for dockerfile, content in ((first, "FROM x\n"), (second, "FROM y\n")):
        dockerfile.parent.mkdir()
        dockerfile.write_text(content, encoding="utf-8")
    one, two = ImageSpec("pr-reviewer-sandbox", first), ImageSpec("pr-reviewer-sandbox", second)
    assert one.tag.startswith("pr-reviewer-sandbox:")
    assert len(one.tag.split(":")[1]) == 12
    assert one.tag != two.tag
    second.write_text("FROM x\n", encoding="utf-8")
    assert one.tag == two.tag
