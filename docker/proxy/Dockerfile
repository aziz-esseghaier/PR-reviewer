# The proxy image: the only way out for a sandbox whose network is on.
#
# tinyproxy, a small HTTP/HTTPS proxy, on the same pinned Debian 12 slim image as the sandboxes.
# It runs as the unprivileged "tinyproxy" user. Its settings (the allowed hosts, which network
# may use it) aren't baked in: the project writes them when it starts the container, from
# configs/sandbox.yaml (see src/pr_reviewer/sandbox/proxy.py).
#
# Built by: pr-reviewer sandbox build
FROM debian:bookworm-slim@sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3

RUN apt-get update \
    && apt-get install -y --no-install-recommends tinyproxy \
    && rm -rf /var/lib/apt/lists/*

USER tinyproxy
