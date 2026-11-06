# The indexer image: the sandbox image plus Jedi, for finding the code around a PR's change
# (src/pr_reviewer/context/). It adds one layer: Jedi and its parser, parso, pinned by version
# and SHA-256, in their own environment at /opt/indexer. The sandbox image, and every image
# built from it (verified environments, prepared PR images), stay exactly as they are.
#
# SANDBOX_IMAGE is the sandbox image's name and tag, passed in by the build. This image's own
# tag is a fingerprint of this file and of that tag, so it's rebuilt when either changes.
#
# Built by: pr-reviewer context build (when missing, in seconds)
ARG SANDBOX_IMAGE
FROM ${SANDBOX_IMAGE}

USER root
RUN printf '%s\n' \
        "jedi==0.20.1 --hash=sha256:0fb16d86c4a4c73c37ba518c77419975e30fcc620658a8d14fbb5720cdd34142" \
        "parso==0.8.7 --hash=sha256:a8926eb2a1b915486941fdbd31e86a4baf88fe8c210f25f2f35ecec5b574ca1c" \
        > /tmp/indexer-requirements.txt \
    && uv venv --python /usr/bin/python3 /opt/indexer \
    && uv pip install --python /opt/indexer/bin/python --require-hashes --no-cache \
        -r /tmp/indexer-requirements.txt \
    && rm /tmp/indexer-requirements.txt \
    && /opt/indexer/bin/python -c "import jedi, parso; print('jedi', jedi.__version__, 'parso', parso.__version__)"
USER sandbox
WORKDIR /work
