"""Sandbox settings: read and check configs/sandbox.yaml.

The file names the Docker objects the sandboxes use (images, networks, proxy, repo volume),
the hosts the proxy lets sandboxes reach, and the limits every sandbox container gets: CPUs,
memory, processes, how long it may live, and how long one command may run. Checking it with
pydantic makes a typo fail at startup with a clear message, instead of starting a container
with a limit missing.

Imported by other modules; the file isn't run directly.
"""

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pr_reviewer.core.config import ConfigError, describe_errors, find_config

# Where the sandbox config lives, relative to the project root.
SANDBOX_CONFIG = Path("configs") / "sandbox.yaml"

# A Docker object name: lowercase letters, digits and separators.
DOCKER_NAME = r"^[a-z0-9][a-z0-9._-]{0,63}$"

# A host name the proxy may forward to, such as "pypi.org". No ports, paths or wildcards.
HOST_NAME = r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$"

# Every container the project starts for sandboxing carries this label, so leftovers can be
# found and removed.
LABEL = "pr-reviewer.sandbox"


class ProxyConfig(BaseModel):
    """The proxy: the only way out of a sandbox whose network is on.

    Attributes:
        image (str): Name of the proxy image, built from docker/proxy/Dockerfile.
        container (str): Name of the proxy container. One serves all sandboxes.
        egress_network (str): The ordinary network the proxy reaches the internet through.
            Sandboxes never join it.
        port (int): The port the proxy listens on.
        allowed_hosts (tuple[str, ...]): The only hosts sandboxes can reach, by exact name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = Field(pattern=DOCKER_NAME)
    container: str = Field(pattern=DOCKER_NAME)
    egress_network: str = Field(pattern=DOCKER_NAME)
    port: int = Field(ge=1024, le=65535)
    allowed_hosts: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_hosts")
    @classmethod
    def _check_hosts(cls, hosts: tuple[str, ...]) -> tuple[str, ...]:
        """Check every allowed host is a plain host name.

        Args:
            hosts (tuple[str, ...]): The allowed hosts as written.

        Returns:
            tuple[str, ...]: The unchanged hosts.

        Raises:
            ValueError: If an entry has a port, a path, a wildcard or capital letters.
        """
        for number, host in enumerate(hosts, start=1):
            if not re.fullmatch(HOST_NAME, host):
                raise ValueError(
                    "entries must be plain lowercase host names, such as pypi.org "
                    f"(entry {number} isn't)"
                )
        return hosts


class Limits(BaseModel):
    """What one sandbox container may use.

    Attributes:
        cpus (float): CPU cores it may use at most.
        memory_gb (float): Memory in GB, with no swap on top. A program that needs more is
            stopped by the kernel.
        processes (int): Most processes (and threads) it may have at once.
        lifetime_minutes (float): The container stops itself after this long, even if the
            program that started it crashed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: float = Field(gt=0, le=64)
    memory_gb: float = Field(gt=0, le=256)
    processes: int = Field(ge=16, le=100_000)
    lifetime_minutes: float = Field(gt=0, le=24 * 60)


class Timeouts(BaseModel):
    """How long one command inside a sandbox may run, by kind.

    Attributes:
        install_minutes (float): Installing a repo and its dependencies.
        test_minutes (float): Running tests or a check.
        download_minutes (float): Downloading a repo's shared copy from GitHub.
        difftest_minutes (float): One run of a few chosen tests by the differential runner.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    install_minutes: float = Field(gt=0, le=24 * 60)
    test_minutes: float = Field(gt=0, le=24 * 60)
    download_minutes: float = Field(gt=0, le=24 * 60)
    difftest_minutes: float = Field(gt=0, le=24 * 60)


class SandboxConfig(BaseModel):
    """The checked sandbox settings.

    Attributes:
        image (str): Name of the sandbox image. Its tag is the Dockerfile's fingerprint.
        network (str): The internal network a sandbox joins while its network is on. It leads
            nowhere but to the proxy.
        repos_volume (str): The Docker volume holding one shared git copy per repo.
        proxy (ProxyConfig): The proxy and the hosts it lets sandboxes reach.
        limits (Limits): Limits of every sandbox container.
        timeouts (Timeouts): Time limits of one command.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = Field(pattern=DOCKER_NAME)
    network: str = Field(pattern=DOCKER_NAME)
    repos_volume: str = Field(pattern=DOCKER_NAME)
    proxy: ProxyConfig
    limits: Limits
    timeouts: Timeouts


def load_sandbox_config(path: Path | None = None) -> SandboxConfig:
    """Read and check the sandbox config file.

    Args:
        path (Path | None): The YAML file to read. None means configs/sandbox.yaml, found from
            the current folder upwards.

    Returns:
        SandboxConfig: The checked settings.

    Raises:
        ConfigError: If the file is missing, isn't valid YAML or doesn't match the expected shape.
    """
    path = path or find_config(SANDBOX_CONFIG)
    if not path.is_file():
        raise ConfigError(f"Sandbox config not found: {path}")
    try:
        # safe_load only builds plain data (dicts, lists, strings). yaml.load could run code.
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return SandboxConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Problems in {path}:\n{describe_errors(exc)}") from None
