"""Command-line entry point for the PR reviewer."""

import argparse
from collections.abc import Sequence

from pr_reviewer import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `pr-reviewer` command.

    Returns:
        argparse.ArgumentParser: Parser with the global options.
    """
    parser = argparse.ArgumentParser(
        prog="pr-reviewer",
        description="Evidence-first AI reviewer for Python pull requests.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the chosen command.

    Args:
        argv (Sequence[str] | None): Arguments to parse. None means the real command line.

    Returns:
        int: Exit code, 0 on success.
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
