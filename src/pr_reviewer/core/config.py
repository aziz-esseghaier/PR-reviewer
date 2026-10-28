"""Load project settings from the environment and the .env file."""

from dotenv import find_dotenv, load_dotenv


class ConfigError(Exception):
    """Raised when a required setting is missing or invalid."""


def load_env() -> None:
    """Load variables from the nearest .env file into the process environment.

    The search starts in the current folder and walks up, so commands work from the project
    root or any subfolder. Variables already set in the environment keep their value.
    """
    load_dotenv(find_dotenv(usecwd=True))
