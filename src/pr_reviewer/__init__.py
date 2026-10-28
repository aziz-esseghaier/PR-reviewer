"""The pr_reviewer package: an evidence-first AI reviewer for Python pull requests.

The reviewer comments on a PR only when it can back the comment with evidence, and SWR-Bench
measures whether that rule improves precision. This file only exposes `__version__`, read
from the installed package metadata. The code lives in the subpackages: core, db, sandbox,
setup_agent, reviewer, benchmark and integrations.

Imported, not run. The command line is in cli.py (`pr-reviewer --help`).
"""

from importlib.metadata import version

__version__ = version("pr-reviewer")
