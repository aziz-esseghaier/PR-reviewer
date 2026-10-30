"""Where the SWR-Bench files come from, and how to download and check them.

Everything comes from one pinned commit of the SWR-Bench repo (github.com/ZZR0/SWRench), and
each file has a SHA-256 fingerprint. A file whose fingerprint differs is never used, so the
data and the official scoring code can't change silently, for example after an upstream
update or a half-finished download.

- The dataset (85 MB) is downloaded into data/, which git ignores (`pr-reviewer bench
  download`).
- The official scripts are copied unchanged into third_party/swrbench/ and kept in git, with
  SWR-Bench's MIT license. check_official_files() tells whether they were edited.

Imported by other modules; the file isn't run directly.
"""

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import httpx2

from pr_reviewer.core.config import project_root

SOURCE_REPO = "ZZR0/SWRench"
# "Prepare public release with MIT license and documentation", 2026-06-05.
SOURCE_COMMIT = "67ae1d4395ac05f800d62b0e13678eadeb9fa5c7"

# Seconds to wait for the server to answer or send the next piece of a file.
DOWNLOAD_TIMEOUT_S = 60.0


class DatasetError(Exception):
    """Raised when a benchmark file is missing, can't be downloaded or doesn't match its pin."""


@dataclass(frozen=True)
class PinnedFile:
    """One file of the SWR-Bench repo at the pinned commit.

    Attributes:
        path (str): Path inside the SWR-Bench repo, such as "data/swr_datasets_d5c5.jsonl".
        sha256 (str): SHA-256 fingerprint of the file's bytes.
        size (int): Size in bytes.
    """

    path: str
    sha256: str
    size: int

    @property
    def name(self) -> str:
        """The file name without its folders, such as "swr_datasets_d5c5.jsonl"."""
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        """Where to download the file at the pinned commit."""
        return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{self.path}"


DATASET = PinnedFile(
    "data/swr_datasets_d5c5.jsonl",
    "7048e7a92ea9f7a1e6812646694a040582448cc1dd9d515264a4b55071aff95d",
    84_826_523,
)

# The official scripts copied into third_party/swrbench/, by file name.
OFFICIAL_FILES = (
    PinnedFile(
        "swrbench/evaluation_struct.py",
        "917505d1265fb3783e765c965960f24205a1d8778e7004f51226321553a5711f",
        56_263,
    ),
    PinnedFile(
        "swrbench/generation.py",
        "e2f5d682bd27d1a8b5bcc52d6082384aae484bfb32f862a97a8b37526cf7383c",
        14_307,
    ),
    PinnedFile(
        "swrbench/utils.py",
        "f25710dbf5ac3ae738c2d40bee02061d824a64b9e5e15a68e3bec4a97f9d888a",
        3_450,
    ),
    PinnedFile(
        "LICENSE",
        "568f99d2d8ac1f8a41402078816325f7a80d38c3e94432975e398584126ae9bf",
        1_061,
    ),
)


def default_dataset_path() -> Path:
    """Where the dataset is kept: data/ in the project folder.

    Returns:
        Path: The dataset file's path. The file may not be downloaded yet.

    Raises:
        ConfigError: If the project folder can't be found.
    """
    return project_root() / "data" / DATASET.name


def official_dir() -> Path:
    """Where the official scripts are kept: third_party/swrbench/ in the project folder.

    Returns:
        Path: The folder.

    Raises:
        ConfigError: If the project folder can't be found.
    """
    return project_root() / "third_party" / "swrbench"


def sha256_of(path: Path) -> str:
    """Compute a file's SHA-256 fingerprint, reading it in pieces to keep memory low.

    Args:
        path (Path): The file to read.

    Returns:
        str: The fingerprint as 64 hex characters.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def matches_pin(file: PinnedFile, path: Path) -> bool:
    """Tell whether a path already holds the pinned file, byte for byte.

    Args:
        file (PinnedFile): The file expected there.
        path (Path): Where to look.

    Returns:
        bool: True when the file exists and its fingerprint matches.
    """
    return path.is_file() and path.stat().st_size == file.size and sha256_of(path) == file.sha256


def download(file: PinnedFile, dest: Path, http: httpx2.Client | None = None) -> None:
    """Download a pinned file and save it only if its fingerprint matches.

    The bytes go to a temporary ".part" file first, which replaces `dest` only after the
    check, so a stopped or changed download never leaves a bad file at `dest`.

    Args:
        file (PinnedFile): The file to download.
        dest (Path): Where to save it. Missing folders are created.
        http (httpx2.Client | None): HTTP client to use. Tests pass one with a fake transport.
            None makes a new client and closes it afterwards.

    Raises:
        DatasetError: If the download fails or the bytes don't match the fingerprint.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    client = http or httpx2.Client(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    try:
        with (
            nullcontext(client) if http else client,
            client.stream("GET", file.url) as response,
        ):
            response.raise_for_status()
            with partial.open("wb") as out:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    out.write(chunk)
    except httpx2.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DatasetError(f"Downloading {file.url} failed: {exc}") from exc
    if digest.hexdigest() != file.sha256:
        partial.unlink(missing_ok=True)
        raise DatasetError(
            f"The downloaded {file.name} doesn't match its pinned fingerprint, so it wasn't "
            f"saved. The file at {file.url} has changed, or the download was cut short."
        )
    partial.replace(dest)


def check_official_files(folder: Path | None = None) -> list[str]:
    """Compare the copied official scripts with their pinned fingerprints.

    Args:
        folder (Path | None): Folder holding the copies. None means third_party/swrbench/.

    Returns:
        list[str]: One line per missing or edited file. Empty when all of them match.
    """
    folder = folder or official_dir()
    problems = []
    for file in OFFICIAL_FILES:
        path = folder / file.name
        if not path.is_file():
            problems.append(f"{file.name} is missing from {folder}")
        elif not matches_pin(file, path):
            problems.append(f"{file.name} differs from SWR-Bench's original")
    return problems
