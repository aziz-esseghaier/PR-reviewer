"""Shared fixtures for the static tests: the Docker ones of tests/sandbox.

`docker`, `test_config` and `sandbox_image` give a Docker connection, the test sandbox config
(its own networks, proxy and repo volume) and the built image; tests using them are skipped
when Docker isn't running or the images aren't built.

pytest loads this file on its own; it isn't run directly.
"""

from tests.sandbox.conftest import docker, sandbox_image, test_config

__all__ = ["docker", "sandbox_image", "test_config"]
