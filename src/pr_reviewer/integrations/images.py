"""The Docker images the Action needs, and building them all for the registry.

Building the images takes a long time (the sandbox image compiles an old Python), so a
workflow in this repo builds them once and pushes them to GitHub's container registry, and the
Action pulls them. Each tag is a fingerprint of its Dockerfile (sandbox/image.py), so the
Action asks for exactly the images its own code expects, and an image built from another
Dockerfile is never used by mistake.

Imported by other modules; the file isn't run directly.
"""

from docker import DockerClient

from pr_reviewer.context.retrieve import indexer_image, ready_indexer
from pr_reviewer.sandbox.commands import ready_image
from pr_reviewer.sandbox.config import SandboxConfig
from pr_reviewer.sandbox.image import proxy_image, sandbox_image
from pr_reviewer.static.tools import ready_tools, tools_image


def image_tags(config: SandboxConfig) -> list[str]:
    """List the images the reviewer needs, by name and tag.

    Args:
        config (SandboxConfig): The sandbox settings, which name the images.

    Returns:
        list[str]: The sandbox, proxy, indexer and tools images, such as
        "pr-reviewer-sandbox:3f1a9c0e2b7d", in the order they must be built.
    """
    return [
        sandbox_image(config).tag,
        proxy_image(config).tag,
        indexer_image(config).tag,
        tools_image(config).tag,
    ]


def build_all(client: DockerClient, config: SandboxConfig) -> list[str]:
    """Build every image the reviewer needs that's missing.

    Args:
        client (DockerClient): Connection to Docker.
        config (SandboxConfig): The sandbox settings.

    Returns:
        list[str]: The images, as image_tags() lists them.
    """
    ready_image(client, config)
    ready_indexer(client, config)
    ready_tools(client, config)
    return image_tags(config)
