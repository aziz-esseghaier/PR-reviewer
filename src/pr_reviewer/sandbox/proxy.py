"""The proxy: the only way out of a sandbox whose network is on.

A sandbox with its network on joins an "internal" Docker network. That network has no route to
the internet or to this machine, only to the containers on it. The proxy container sits on it
and on an ordinary network (the egress network), and forwards a sandbox's web requests only to
the hosts allowed in configs/sandbox.yaml. Tools in the sandbox find the proxy through the
usual settings (HTTP_PROXY and HTTPS_PROXY), which pip, git, apt and conda all read. So install
code can download packages, but can't reach the database or pgAdmin on this machine, or any
site that isn't allowed.

The proxy is tinyproxy, run as an unprivileged user, with no Linux privileges, a read-only file
system and small limits. HTTPS goes through it as an encrypted tunnel to port 443 of an allowed
host; it can't see inside. It starts when the first sandbox needs it and keeps running, and it
is started again with new settings when configs/sandbox.yaml changes.

Imported by other modules; the file isn't run directly.
"""

import hashlib
import re
import time

import docker
from docker import DockerClient
from docker.models.containers import Container
from docker.models.networks import Network

from pr_reviewer.sandbox.config import LABEL, SandboxConfig
from pr_reviewer.sandbox.image import ImageError, image_exists, proxy_image

# The proxy container carries this label, with a fingerprint of its settings.
PROXY_LABEL = "pr-reviewer.proxy"

# Starts tinyproxy with the settings passed in two environment variables. The container's file
# system is read-only, so the files go to /tmp, a small in-memory folder.
_START = (
    'printf "%s" "$PROXY_SETTINGS" > /tmp/tinyproxy.conf'
    ' && printf "%s" "$PROXY_FILTER" > /tmp/filter'
    " && exec tinyproxy -d -c /tmp/tinyproxy.conf"
)

# How long the proxy may take to start.
START_SECONDS = 20


class ProxyError(ImageError):
    """Raised when the proxy or its networks can't be set up."""


def proxy_url(config: SandboxConfig) -> str:
    """Give the proxy's address, as seen from a sandbox on the internal network.

    Args:
        config (SandboxConfig): The sandbox settings.

    Returns:
        str: Such as "http://pr-reviewer-proxy:8888".
    """
    return f"http://{config.proxy.container}:{config.proxy.port}"


def proxy_environment(config: SandboxConfig) -> dict[str, str]:
    """Give the settings that make tools in a sandbox use the proxy.

    Both spellings are set, because some tools read only the upper-case names and others only
    the lower-case ones.

    Args:
        config (SandboxConfig): The sandbox settings.

    Returns:
        dict[str, str]: Environment variables for the sandbox container.
    """
    url = proxy_url(config)
    local = "localhost,127.0.0.1"
    return {
        "HTTP_PROXY": url,
        "HTTPS_PROXY": url,
        "http_proxy": url,
        "https_proxy": url,
        "NO_PROXY": local,
        "no_proxy": local,
    }


def tinyproxy_settings(config: SandboxConfig, allow_from: str) -> tuple[str, str]:
    """Write tinyproxy's settings and its list of allowed hosts.

    Args:
        config (SandboxConfig): The sandbox settings, with the port and allowed hosts.
        allow_from (str): The internal network's address range, such as "172.20.0.0/16". Only
            containers on it may use the proxy.

    Returns:
        tuple[str, str]: The settings file, and the filter file: one pattern per allowed host,
        matching that exact name only.
    """
    settings = "\n".join(
        [
            f"Port {config.proxy.port}",
            "Listen 0.0.0.0",
            "Timeout 600",
            "MaxClients 50",
            f"Allow {allow_from}",
            # HTTPS tunnels only to the usual port, so no tunnel can reach a database port.
            "ConnectPort 443",
            # Refuse every host not in the filter file, matched on the host name alone.
            "FilterDefaultDeny Yes",
            "FilterURLs Off",
            "FilterType ere",
            'Filter "/tmp/filter"',
            "DisableViaHeader Yes",
            "LogLevel Notice",
            "",
        ]
    )
    patterns = "".join(f"^{re.escape(host)}$\n" for host in config.proxy.allowed_hosts)
    return settings, patterns


def _network(client: DockerClient, name: str, *, internal: bool) -> Network:
    """Get a network of the right kind, creating it (or replacing an unused wrong one) if needed.

    Args:
        client (DockerClient): Connection to Docker.
        name (str): The network's name.
        internal (bool): True for a network with no route out.

    Returns:
        Network: The network.

    Raises:
        ProxyError: If a network of that name exists with the wrong kind and is in use.
    """
    try:
        network = client.networks.get(name)
    except docker.errors.NotFound:
        return client.networks.create(
            name, driver="bridge", internal=internal, labels={LABEL: "network"}
        )
    if bool(network.attrs.get("Internal")) == internal:
        return network
    network.reload()
    if network.attrs.get("Containers"):
        raise ProxyError(
            f"The network {name} is the wrong kind and still in use. Remove the containers on "
            "it first: pr-reviewer sandbox cleanup"
        )
    network.remove()
    return client.networks.create(
        name, driver="bridge", internal=internal, labels={LABEL: "network"}
    )


def ensure_network(client: DockerClient, config: SandboxConfig) -> Network:
    """Make sure the internal network and its proxy are up, and give the internal network.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.

    Returns:
        Network: The internal network sandboxes join while their network is on.

    Raises:
        ImageError: If the proxy image isn't built yet.
        ProxyError: If the networks or the proxy can't be set up.
    """
    internal = _network(client, config.network, internal=True)
    egress = _network(client, config.proxy.egress_network, internal=False)
    _ensure_proxy(client, config, internal, egress)
    return internal


def remove_proxy(client: DockerClient, config: SandboxConfig) -> bool:
    """Stop and remove the proxy container, if it's there.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings, which name the container.

    Returns:
        bool: True when a proxy container was removed.
    """
    try:
        client.containers.get(config.proxy.container).remove(force=True)
    except docker.errors.NotFound:
        return False
    return True


def proxy_status(client: DockerClient, config: SandboxConfig) -> str:
    """Say whether the proxy is running.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.

    Returns:
        str: The container's state, such as "running", or "not started".
    """
    try:
        return str(client.containers.get(config.proxy.container).status)
    except docker.errors.NotFound:
        return "not started"


def _ensure_proxy(
    client: DockerClient, config: SandboxConfig, internal: Network, egress: Network
) -> None:
    """Start the proxy, or start it again when its settings or image changed.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.
        internal (Network): The internal network the sandboxes use.
        egress (Network): The ordinary network the proxy goes out through.

    Raises:
        ImageError: If the proxy image isn't built yet.
        ProxyError: If the proxy doesn't start.
    """
    tag = proxy_image(config).tag
    if not image_exists(client, tag):
        raise ImageError(
            f"The proxy image {tag} isn't built. Build it with: pr-reviewer sandbox build"
        )
    subnet = internal.attrs["IPAM"]["Config"][0]["Subnet"]
    settings, patterns = tinyproxy_settings(config, subnet)
    wanted = "\n".join([tag, settings, patterns, internal.id or "", egress.id or ""])
    fingerprint = hashlib.sha256(wanted.encode()).hexdigest()[:16]
    try:
        existing = client.containers.get(config.proxy.container)
    except docker.errors.NotFound:
        existing = None
    if existing is not None:
        if existing.labels.get(PROXY_LABEL) == fingerprint and existing.status == "running":
            return
        existing.remove(force=True)
    container = client.containers.create(
        tag,
        command=["sh", "-c", _START],
        name=config.proxy.container,
        environment={"PROXY_SETTINGS": settings, "PROXY_FILTER": patterns},
        labels={PROXY_LABEL: fingerprint},
        network=egress.name,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        read_only=True,
        tmpfs={"/tmp": "size=16m"},
        mem_limit="128m",
        memswap_limit="128m",
        nano_cpus=500_000_000,
        pids_limit=64,
        init=True,
        # Comes back by itself after a crash or a Docker Desktop restart.
        restart_policy={"Name": "always"},
    )
    internal.connect(container)
    container.start()
    _wait_until_running(container)


def _wait_until_running(container: Container) -> None:
    """Wait until the proxy has started and stayed up for a moment.

    Args:
        container (Container): The proxy container.

    Raises:
        ProxyError: If it stops instead, with the end of its log.
    """
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        # tinyproxy stops within a second on a bad setting, so a proxy still up after two
        # seconds has read its settings.
        time.sleep(2)
        container.reload()
        if container.status == "running":
            return
        if container.status in ("exited", "dead"):
            break
    log = container.logs(tail=10).decode("utf-8", errors="replace").strip()
    raise ProxyError(f"The proxy didn't start ({container.status}): {log}")
