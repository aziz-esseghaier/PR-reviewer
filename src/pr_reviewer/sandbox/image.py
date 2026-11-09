"""The images: the sandbox (docker/sandbox/Dockerfile), the proxy (docker/proxy/Dockerfile),
and images built on top of the sandbox, such as the indexer (docker/indexer/Dockerfile).

An image's tag is the first 12 characters of its Dockerfile's SHA-256 fingerprint, such as
pr-reviewer-sandbox:3f1a9c0e2b7d. Editing a Dockerfile gives a new tag, so a stale image is
never used by mistake, and an unchanged one is never rebuilt. Docker reuses the steps it has
already built, so a rebuild after a small edit is quick; the first sandbox build downloads git
and Python and took about 11 minutes on a slow connection. An image built on another one gets
that image's tag as the build argument SANDBOX_IMAGE, and the tag is part of its fingerprint.

Imported by other modules; the file isn't run directly.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import docker
from docker import DockerClient

from pr_reviewer.core.config import project_root
from pr_reviewer.sandbox.config import SandboxConfig

# The Dockerfiles, relative to the project root.
SANDBOX_DOCKERFILE = Path("docker") / "sandbox" / "Dockerfile"
PROXY_DOCKERFILE = Path("docker") / "proxy" / "Dockerfile"

# Called with each line of build output.
BuildProgress = Callable[[str], None]


class ImageError(Exception):
    """Raised when an image can't be built, or is needed before it's built."""


@dataclass(frozen=True)
class ImageSpec:
    """An image to build: its name and its Dockerfile.

    Attributes:
        name (str): The image name, without a tag, such as "pr-reviewer-sandbox".
        dockerfile (Path): The Dockerfile. Its folder is all Docker receives when building.
        base (str): The name and tag of the image it's built on, passed to the build as
            SANDBOX_IMAGE; "" for an image built from a public one.
        files (tuple[str, ...]): Other files in the Dockerfile's folder the build copies in,
            such as "requirements.txt". They're part of the fingerprint too.
    """

    name: str
    dockerfile: Path
    base: str = ""
    files: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        """The name and tag the current Dockerfile builds, such as "pr-reviewer-sandbox:3f1a..."."""
        content = self.dockerfile.read_bytes()
        if self.base:
            content += b"\nbase " + self.base.encode()
        for name in self.files:
            content += f"\nfile {name}\n".encode() + (self.dockerfile.parent / name).read_bytes()
        digest = hashlib.sha256(content).hexdigest()[:12]
        return f"{self.name}:{digest}"


def sandbox_image(config: SandboxConfig) -> ImageSpec:
    """Describe the sandbox image.

    Args:
        config (SandboxConfig): The sandbox settings, which name it.

    Returns:
        ImageSpec: The image.

    Raises:
        ConfigError: If the project folder can't be found.
    """
    return ImageSpec(config.image, project_root() / SANDBOX_DOCKERFILE)


def proxy_image(config: SandboxConfig) -> ImageSpec:
    """Describe the proxy image.

    Args:
        config (SandboxConfig): The sandbox settings, which name it.

    Returns:
        ImageSpec: The image.

    Raises:
        ConfigError: If the project folder can't be found.
    """
    return ImageSpec(config.proxy.image, project_root() / PROXY_DOCKERFILE)


def image_exists(client: DockerClient, tag: str) -> bool:
    """Tell whether an image is already built.

    Args:
        client (DockerClient): Connection to Docker.
        tag (str): The image name and tag.

    Returns:
        bool: True when Docker has the image.
    """
    try:
        client.images.get(tag)
    except docker.errors.ImageNotFound:
        return False
    return True


def build_image(
    client: DockerClient, spec: ImageSpec, on_progress: BuildProgress | None = None
) -> str:
    """Build an image, unless it's already built.

    Args:
        client (DockerClient): Connection to Docker.
        spec (ImageSpec): The image to build.
        on_progress (BuildProgress | None): Called with each line of build output.

    Returns:
        str: The image name and tag.

    Raises:
        ImageError: If a build step fails.
    """
    tag = spec.tag
    if image_exists(client, tag):
        return tag
    if spec.base and not image_exists(client, spec.base):
        raise ImageError(f"Building {tag} needs {spec.base}, which isn't built")
    # The low-level call streams the build output as it happens.
    stream = client.api.build(
        path=str(spec.dockerfile.parent),
        tag=tag,
        rm=True,
        forcerm=True,
        decode=True,
        labels={"pr-reviewer.dockerfile": tag.rsplit(":", 1)[1]},
        buildargs={"SANDBOX_IMAGE": spec.base} if spec.base else None,
    )
    for chunk in stream:
        if "error" in chunk:
            raise ImageError(f"Building {tag} failed: {str(chunk['error']).strip()}")
        line = str(chunk.get("stream", "")).strip()
        if line and on_progress is not None:
            on_progress(line)
    if not image_exists(client, tag):
        raise ImageError(f"Building {tag} finished without an image")
    return tag
