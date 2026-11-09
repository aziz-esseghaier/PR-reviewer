# The tools image: the sandbox image plus the static analysis tools (src/pr_reviewer/static/).
# It adds:
# - ruff, mypy, bandit and semgrep, with every package they need pinned by version and
#   SHA-256 (requirements.txt), in their own environment at /opt/tools;
# - semgrep's Python rules for correctness and security, from the semgrep-rules repo at a fixed
#   commit, in /opt/tools/semgrep-rules. They're under the Semgrep Rules License (its LICENSE
#   file is kept with them): free to use on our own code and research, not to resell;
# - an empty mypy config, so mypy can be told to ignore a repo's own settings;
# - /opt/tools/VERSION, listing each tool's version and the rules' commit.
# PR code never runs here. The tools reach PR sandboxes as a read-only copy of /opt/tools in a
# Docker volume (static/tools.py), so the PR's own image stays as it is.
#
# SANDBOX_IMAGE is the sandbox image's name and tag, passed in by the build. This image's tag
# is a fingerprint of this file, requirements.txt and that tag, so it's rebuilt when any
# changes.
#
# Built by: pr-reviewer static run (when missing, in a few minutes)
ARG SANDBOX_IMAGE
FROM ${SANDBOX_IMAGE}

USER root

COPY requirements.txt /tmp/tools-requirements.txt
RUN uv venv --python /usr/bin/python3 /opt/tools \
    && uv pip install --python /opt/tools/bin/python --require-hashes --no-cache \
        -r /tmp/tools-requirements.txt \
    && rm /tmp/tools-requirements.txt

# semgrep-rules at a fixed commit: git checks the commit ID, which fixes every file's content.
ARG RULES_COMMIT=16b9e6d45f10649f6a531ec4838ed9924372c3bc
RUN git init -q /tmp/rules \
    && git -C /tmp/rules fetch -q --depth 1 https://github.com/semgrep/semgrep-rules "$RULES_COMMIT" \
    && git -C /tmp/rules checkout -q FETCH_HEAD \
    && test "$(git -C /tmp/rules rev-parse HEAD)" = "$RULES_COMMIT" \
    && mkdir -p /opt/tools/semgrep-rules \
    && cd /tmp/rules/python \
    && find . -name '*.yaml' ! -name '*.test.yaml' \
        \( -path '*/correctness/*' -o -path '*/security/*' \) \
        -exec cp --parents {} /opt/tools/semgrep-rules/ \; \
    && cp /tmp/rules/LICENSE /opt/tools/semgrep-rules/LICENSE \
    && cd / \
    && rm -rf /tmp/rules

# The empty mypy config, the versions file, and a check that every tool starts and semgrep
# accepts every rule.
RUN printf '[mypy]\n' > /opt/tools/mypy.ini \
    && printf 'x = 1\n' > /tmp/smoke.py \
    && export HOME=/tmp/smoke-home \
    && { /opt/tools/bin/ruff --version; \
         /opt/tools/bin/mypy --version; \
         /opt/tools/bin/bandit --version | head -n 1; \
         echo "semgrep $(/opt/tools/bin/semgrep --version --disable-version-check)"; \
         echo "semgrep-rules $RULES_COMMIT $(find /opt/tools/semgrep-rules -name '*.yaml' | wc -l) rule files"; \
       } > /opt/tools/VERSION \
    && /opt/tools/bin/semgrep scan --config /opt/tools/semgrep-rules --metrics off \
        --disable-version-check --quiet --json --output /tmp/smoke.json /tmp/smoke.py \
    && /opt/tools/bin/python -c "import json, sys; errors = json.load(open('/tmp/smoke.json'))['errors']; sys.exit(f'semgrep rule errors: {errors}' if errors else 0)" \
    && rm -rf /tmp/smoke.py /tmp/smoke.json /tmp/smoke-home \
    && cat /opt/tools/VERSION

USER sandbox
WORKDIR /work
