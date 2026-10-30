"""SWR-Bench's official scoring code, loaded unchanged from third_party/swrbench/.

The scoring script, evaluation_struct.py, holds the judge prompts, the JSON formats the judge
must answer in, the checks on its answers and the metric formulas. We use all of it as is.
It imports run_chat from SWR-Bench's utils.py to call the model, but that file crashes on
import and has no cache, so the script is loaded with a stand-in `utils` module whose run_chat
the caller supplies (see benchmark/judge.py).

The script's fingerprint is checked first, so an edited copy is never run.

Imported by other modules; the file isn't run directly.
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from pr_reviewer.benchmark.source import DatasetError, check_official_files, official_dir

# The official run_chat: (model, messages, temperature, max_tokens, response_format,
# max_retries) -> the reply text, or None when the call failed.
RunChat = Callable[..., str | None]


def _no_model(*args: Any, **kwargs: Any) -> str | None:
    """Stand in for run_chat when the caller only needs the prompts and checks.

    Raises:
        RuntimeError: Always: this copy of the script has no model to call.
    """
    raise RuntimeError("This copy of SWR-Bench's scoring script has no model to call.")


def load_evaluation(run_chat: RunChat = _no_model, folder: Path | None = None) -> ModuleType:
    """Load SWR-Bench's evaluation_struct.py, with our run_chat in place of theirs.

    Args:
        run_chat (RunChat): What the script calls to reach the judge model. The default
            refuses, for callers that only build prompts or check answers.
        folder (Path | None): Folder holding the official files. None means
            third_party/swrbench/.

    Returns:
        ModuleType: The loaded script. Its functions, such as evaluate_one() and
        create_change_pr_prompt(), are SWR-Bench's own.

    Raises:
        DatasetError: If an official file is missing or was edited.
    """
    folder = folder or official_dir()
    problems = check_official_files(folder)
    if problems:
        raise DatasetError(f"SWR-Bench's scoring script can't be used: {'; '.join(problems)}")
    spec = importlib.util.spec_from_file_location(
        "swrbench_evaluation", folder / "evaluation_struct.py"
    )
    if spec is None or spec.loader is None:
        raise DatasetError(f"Can't load {folder / 'evaluation_struct.py'}")
    module = importlib.util.module_from_spec(spec)
    stand_in = ModuleType("utils", "Stand-in for SWR-Bench's utils.py: only run_chat.")
    stand_in.__dict__["run_chat"] = run_chat
    # The script runs `from utils import run_chat` while loading. Put our stand-in where that
    # import looks, and put back whatever was there afterwards.
    previous = sys.modules.get("utils")
    sys.modules["utils"] = stand_in
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["utils"]
        else:
            sys.modules["utils"] = previous
    return module
