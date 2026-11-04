# The sandbox image: what every container that holds PR code starts from.
#
# Debian 12 (bookworm) slim, pinned to an exact image, plus git, Python 3 and a fixed set of
# build tools. Commands run as the non-root user "sandbox" (uid 1000), so nothing inside can
# install system packages: everything a repo needs from the system must be here. When a setup
# fails for a missing system library, or for a build download the proxy blocks, it's added
# below, by hand.
#
# Python versions come from uv (pinned below), which installs any version from 3.8 on into
# the sandbox user's home, without root; 3.7 is built in, at /opt/python/3.7. /repos is where the shared repo copies are
# mounted (read-only, except for the container that downloads them), and /work is where a
# sandbox keeps its own working copy and virtual environment.
#
# Built by: pr-reviewer sandbox build
FROM debian:bookworm-slim@sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3

# git: copies and checkouts. python3: small checks inside the sandbox. procps: ps, for
# debugging. coreutils (already in the image) provides `timeout`, which enforces time limits.
# The rest is what packages with C, C++ or Fortran parts need to compile (scikit-learn,
# astropy, matplotlib...): compilers, ccache to reuse earlier compile results, and the headers
# of common libraries. tzdata and graphviz are used by some repos' tests.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git procps python3 \
        build-essential gfortran pkg-config ccache \
        libfreetype-dev libpng-dev libjpeg-dev zlib1g-dev libffi-dev libssl-dev \
        libopenblas-dev liblapack-dev libhdf5-dev \
        graphviz tzdata \
    && rm -rf /var/lib/apt/lists/*

# uv, copied from its official image, pinned by digest (uv 0.13.1).
COPY --from=ghcr.io/astral-sh/uv@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /uvx /usr/local/bin/

# A fresh Docker volume mounted at /repos takes this folder's owner, so the sandbox user can
# write the shared copies in the one container that downloads them.
RUN useradd --create-home --uid 1000 --shell /bin/bash sandbox \
    && mkdir /repos /work \
    && chown sandbox:sandbox /repos /work

# Python 3.7, which uv can't install (its builds start at 3.8). PRs from 2019 and 2020 need it,
# because the package versions of their time often don't support 3.8. Built from python.org's
# source, checked against the file's SHA-256 (it matches the MD5 python.org lists).
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget libbz2-dev liblzma-dev libsqlite3-dev \
        libreadline-dev libncursesw5-dev libgdbm-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && wget -q https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tar.xz \
    && echo "7911051ed0422fd54b8f59ffc030f7cf2ae30e0f61bda191800bb040dce4f9d2  Python-3.7.17.tar.xz" \
        | sha256sum -c - \
    && tar xf Python-3.7.17.tar.xz \
    && cd Python-3.7.17 \
    && ./configure --prefix=/opt/python/3.7 >/dev/null \
    && make -j"$(nproc)" >/dev/null \
    && make altinstall >/dev/null \
    && cd / \
    && rm -rf /tmp/Python-3.7.17* \
    && /opt/python/3.7/bin/python3.7 -c "import ssl, sqlite3, bz2, lzma, ctypes, zlib"

# Headers added after setups failed without them: libmemcached for pylibmc (in Django's test
# requirements).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmemcached-dev \
    && rm -rf /var/lib/apt/lists/*

# Source archives that matplotlib's build downloads from sites the proxy doesn't allow:
# FreeType 2.6.1 (SourceForge or Savannah) and qhull (qhull.org; meson builds get it from
# GitHub). They're fetched once here, checked against the SHA-256 values in matplotlib's own
# build files, and put where its build looks before downloading:
# - matplotlib before 3.9 (setup.py) looks in ~/.cache/matplotlib, by SHA-256;
# - matplotlib 3.9 on (meson) looks in MESON_PACKAGE_CACHE_DIR, by file name.
# The sandbox user owns both folders, since matplotlib also keeps its font cache in the first.
RUN mkdir -p /home/sandbox/.cache/matplotlib /opt/build-cache/meson \
    && cd /tmp \
    && (wget -q -O freetype-2.6.1.tar.gz \
            https://downloads.sourceforge.net/project/freetype/freetype2/2.6.1/freetype-2.6.1.tar.gz \
        || wget -q -O freetype-2.6.1.tar.gz \
            https://download.savannah.gnu.org/releases/freetype/freetype-old/freetype-2.6.1.tar.gz) \
    && wget -q -O qhull-2020-src-8.0.2.tgz http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz \
    && wget -q -O qhull-8.0.2.tgz https://github.com/qhull/qhull/archive/v8.0.2/qhull-8.0.2.tar.gz \
    && printf '%s\n' \
        "0a3c7dfbda6da1e8fce29232e8e96d987ababbbf71ebc8c75659e4132c367014  freetype-2.6.1.tar.gz" \
        "b5c2d7eb833278881b952c8a52d20179eab87766b00b865000469a45c1838b7e  qhull-2020-src-8.0.2.tgz" \
        "8774e9a12c70b0180b95d6b0b563c5aa4bea8d5960c15e18ae3b6d2521d64f8b  qhull-8.0.2.tgz" \
        | sha256sum -c - \
    && cp freetype-2.6.1.tar.gz \
        /home/sandbox/.cache/matplotlib/0a3c7dfbda6da1e8fce29232e8e96d987ababbbf71ebc8c75659e4132c367014 \
    && mv qhull-2020-src-8.0.2.tgz \
        /home/sandbox/.cache/matplotlib/b5c2d7eb833278881b952c8a52d20179eab87766b00b865000469a45c1838b7e \
    && mv freetype-2.6.1.tar.gz qhull-8.0.2.tgz /opt/build-cache/meson/ \
    && chown -R sandbox:sandbox /home/sandbox/.cache /opt/build-cache
ENV MESON_PACKAGE_CACHE_DIR=/opt/build-cache/meson

# Git never stops to ask for a password; a fetch that needs one fails instead.
ENV GIT_TERMINAL_PROMPT=0
# UTF-8 text everywhere, as most test suites expect.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# Compilers go through ccache, so rebuilding a repo at another commit only compiles what changed.
ENV PATH=/usr/lib/ccache:$PATH CCACHE_DIR=/home/sandbox/.ccache
# uv copies files instead of linking them (the cache and the environment may be on different
# file systems), and asks for Python downloads quietly.
ENV UV_LINK_MODE=copy UV_NO_PROGRESS=1

USER sandbox
WORKDIR /work
