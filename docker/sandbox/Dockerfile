# The sandbox image: what every container that holds PR code starts from.
#
# Debian 12 (bookworm) slim, pinned to an exact image, plus git, Python 3 and a few basic tools.
# Commands run as the non-root user "sandbox" (uid 1000). /repos is where the shared repo copies
# are mounted (read-only, except for the container that downloads them), and /work is where a
# sandbox keeps its own working copy. Later milestones add the Python versions and build tools
# each repo needs on top of this image.
#
# Built by: pr-reviewer sandbox build
FROM debian:bookworm-slim@sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3

# git: copies and checkouts. python3: small checks inside the sandbox. procps: ps, for
# debugging. coreutils (already in the image) provides `timeout`, which enforces time limits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git procps python3 \
    && rm -rf /var/lib/apt/lists/*

# A fresh Docker volume mounted at /repos takes this folder's owner, so the sandbox user can
# write the shared copies in the one container that downloads them.
RUN useradd --create-home --uid 1000 --shell /bin/bash sandbox \
    && mkdir /repos /work \
    && chown sandbox:sandbox /repos /work

# Git never stops to ask for a password; a fetch that needs one fails instead.
ENV GIT_TERMINAL_PROMPT=0

USER sandbox
WORKDIR /work
