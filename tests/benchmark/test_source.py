"""
Tests for the pinned SWR-Bench files in benchmark/source.py.

Downloads go through a fake HTTP transport, so no test uses the network. Checks that a
matching file is saved, that a changed or failed download leaves nothing behind, and that the
official scripts in third_party/swrbench/ are unchanged copies.

How to run (from the project folder):
    pytest tests/benchmark/test_source.py
"""

import hashlib
import shutil
from pathlib import Path

import httpx2
import pytest

from pr_reviewer.benchmark.source import (
    OFFICIAL_FILES,
    SOURCE_COMMIT,
    DatasetError,
    PinnedFile,
    check_official_files,
    download,
    matches_pin,
)

OFFICIAL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "swrbench"
CONTENT = b'{"instance_id": "psf__requests-1"}\n'
PINNED = PinnedFile("data/tiny.jsonl", hashlib.sha256(CONTENT).hexdigest(), len(CONTENT))


def serving(status: int, body: bytes, seen: list[httpx2.Request] | None = None) -> httpx2.Client:
    """Make an HTTP client whose every request gets the same fake answer.

    Args:
        status (int): HTTP status code of the answer.
        body (bytes): Body of the answer.
        seen (list[httpx2.Request] | None): Collects the requests sent, when given.

    Returns:
        httpx2.Client: The client.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return httpx2.Response(status, content=body)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_download_saves_a_matching_file(tmp_path: Path) -> None:
    """Check that a file with the pinned fingerprint is saved, from the pinned commit."""
    dest = tmp_path / "data" / "tiny.jsonl"
    seen: list[httpx2.Request] = []
    download(PINNED, dest, serving(200, CONTENT, seen))
    assert dest.read_bytes() == CONTENT
    assert matches_pin(PINNED, dest)
    assert f"/ZZR0/SWRench/{SOURCE_COMMIT}/data/tiny.jsonl" in str(seen[0].url)
    assert list(dest.parent.iterdir()) == [dest]  # no .part file left


def test_changed_file_is_not_saved(tmp_path: Path) -> None:
    """Check that bytes with a different fingerprint never reach the destination."""
    dest = tmp_path / "tiny.jsonl"
    with pytest.raises(DatasetError, match="doesn't match its pinned fingerprint"):
        download(PINNED, dest, serving(200, b"something else\n"))
    assert list(tmp_path.iterdir()) == []


def test_failed_download_explains_and_leaves_nothing(tmp_path: Path) -> None:
    """Check that an HTTP error becomes a DatasetError and leaves no partial file."""
    with pytest.raises(DatasetError, match="404"):
        download(PINNED, tmp_path / "tiny.jsonl", serving(404, b"Not Found"))
    assert list(tmp_path.iterdir()) == []


def test_matches_pin_needs_the_exact_bytes(tmp_path: Path) -> None:
    """Check that a missing file or one with other bytes doesn't count as the pinned file."""
    path = tmp_path / "tiny.jsonl"
    assert not matches_pin(PINNED, path)
    path.write_bytes(CONTENT.replace(b"1", b"2"))  # same size, different bytes
    assert not matches_pin(PINNED, path)
    path.write_bytes(CONTENT)
    assert matches_pin(PINNED, path)


def test_official_files_are_unchanged_copies() -> None:
    """Check that the scripts in third_party/swrbench/ match SWR-Bench's originals."""
    assert check_official_files(OFFICIAL_DIR) == []


def test_edited_or_missing_official_files_are_reported(tmp_path: Path) -> None:
    """Check that an edited script and a missing one are both named."""
    for file in OFFICIAL_FILES:
        shutil.copyfile(OFFICIAL_DIR / file.name, tmp_path / file.name)
    with (tmp_path / "evaluation_struct.py").open("a", encoding="utf-8") as script:
        script.write("# a local tweak\n")
    (tmp_path / "utils.py").unlink()
    assert check_official_files(tmp_path) == [
        "evaluation_struct.py differs from SWR-Bench's original",
        f"utils.py is missing from {tmp_path}",
    ]
