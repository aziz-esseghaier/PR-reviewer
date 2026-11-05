"""Project settings: the .env file and the model config.

Loads .env into the environment (API keys, database settings), finds the project folder, finds
and reads configs/models.yaml, and checks it with pydantic, so a typo, an unknown provider or a
missing key fails at startup with a clear message. The config holds the providers (with the
rule for when their daily quota resets) and the model profiles (with their per-minute and
per-day limits). Error messages never repeat the values you typed, in case a key ended up in the
wrong place. Also reads a provider's API key from the environment.

Imported by other modules; the file isn't run directly.
"""

import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Where the model config lives, relative to the project root.
MODELS_CONFIG = Path("configs") / "models.yaml"


class ConfigError(Exception):
    """Raised when a required setting is missing or invalid."""


def load_env() -> None:
    """Load variables from the nearest .env file into the process environment.

    The search starts in the current folder and walks up, so commands work from the project
    root or any subfolder. Variables already set in the environment keep their value.
    """
    load_dotenv(find_dotenv(usecwd=True))


def find_config(relative_path: Path) -> Path:
    """Find a config file by looking in the current folder, then in each parent folder.

    This matches how the .env file is found, so commands work from any project subfolder.

    Args:
        relative_path (Path): Path of the file relative to the project root,
            such as configs/models.yaml.

    Returns:
        Path: The first matching file.

    Raises:
        ConfigError: If no folder from here up to the drive root contains the file.
    """
    here = Path.cwd()
    for folder in (here, *here.parents):
        candidate = folder / relative_path
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"Can't find {relative_path} here or in any parent folder. "
        "Run the command inside the project, or pass --config."
    )


def project_root() -> Path:
    """Find the project folder: the nearest folder, from here up, that holds pyproject.toml.

    Returns:
        Path: The project folder.

    Raises:
        ConfigError: If no folder from here up to the drive root holds pyproject.toml.
    """
    here = Path.cwd()
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    raise ConfigError(
        "Can't find the project folder (no pyproject.toml here or in any parent folder). "
        "Run the command inside the project."
    )


@dataclass(frozen=True)
class FixedReset:
    """A daily quota that resets all at once at a fixed local time, such as midnight Pacific.

    Attributes:
        at (dt.time): Local time of the reset.
        zone (ZoneInfo): Time zone that time is in. Handles daylight saving.
    """

    at: dt.time
    zone: ZoneInfo


# How a provider's daily quota comes back:
# - FixedReset: all at once at a set time each day (Google: midnight Pacific).
# - "gradual": a little at a time, refilling the whole quota over 24 hours (Groq).
# - "rolling": unknown, so we count the last 24 hours. Safe, but may stop too early.
ResetRule = FixedReset | Literal["gradual", "rolling"]

DAILY_RESET_HELP = (
    "daily_reset must be 'gradual' or a time and time zone, such as '00:00 America/Los_Angeles'"
)


def parse_daily_reset(text: str | None) -> ResetRule:
    """Turn a provider's daily_reset setting into a reset rule.

    Args:
        text (str | None): "gradual", a time and time zone such as "00:00 America/Los_Angeles",
            or None when the setting is left out.

    Returns:
        ResetRule: FixedReset, "gradual", or "rolling" when the setting is left out.

    Raises:
        ValueError: If the text is neither "gradual" nor a valid time and time zone.
    """
    if text is None:
        return "rolling"
    if text.strip() == "gradual":
        return "gradual"
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s+(\S+)", text.strip())
    if match:
        hour, minute, zone = match.groups()
        try:
            return FixedReset(at=dt.time(int(hour), int(minute)), zone=ZoneInfo(zone))
        except (ValueError, ZoneInfoNotFoundError):
            pass
    raise ValueError(DAILY_RESET_HELP)


def describe_reset(rule: ResetRule) -> str:
    """Describe a reset rule in a few words, for messages and reports.

    Args:
        rule (ResetRule): The rule to describe.

    Returns:
        str: Such as "resets at 00:00 America/Los_Angeles".
    """
    if isinstance(rule, FixedReset):
        return f"resets at {rule.at:%H:%M} {rule.zone.key}"
    if rule == "gradual":
        return "refills gradually over 24 h"
    return "counted over the last 24 h"


class ProviderConfig(BaseModel):
    """Connection settings for one LLM provider.

    Attributes:
        base_url (str): Address of the provider's OpenAI-compatible API.
        api_key_env (str): Name of the environment variable that holds the API key.
            The key itself never goes in a config file.
        timeout_s (float): Seconds to wait for a reply before giving up.
        daily_reset (str | None): When the daily quota comes back: "gradual", or a time and
            time zone such as "00:00 America/Los_Angeles". None means unknown, so the last
            24 hours are counted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = Field(pattern=r"^https://")
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    timeout_s: float = Field(default=120.0, gt=0)
    daily_reset: str | None = None

    @field_validator("daily_reset")
    @classmethod
    def _check_daily_reset(cls, value: str | None) -> str | None:
        """Make sure daily_reset can be understood.

        Args:
            value (str | None): The setting as written in the config.

        Returns:
            str | None: The unchanged setting.

        Raises:
            ValueError: If the setting can't be parsed.
        """
        parse_daily_reset(value)
        return value

    @property
    def reset_rule(self) -> ResetRule:
        """The provider's daily reset rule, parsed from daily_reset.

        Returns:
            ResetRule: FixedReset, "gradual" or "rolling".
        """
        return parse_daily_reset(self.daily_reset)


class ModelLimits(BaseModel):
    """A model's free-tier limits. The client waits, or stops, rather than go over them.

    Attributes:
        requests_per_minute (int | None): Most requests in any 60 seconds. None means no limit.
        tokens_per_minute (int | None): Most tokens, prompt plus reply, in any 60 seconds.
            None means no limit.
        requests_per_day (int | None): Most requests per day, counted under the provider's
            daily_reset rule. None means no limit.
        tokens_per_day (int | None): Most tokens per day, counted the same way.
            None means no limit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requests_per_minute: int | None = Field(default=None, gt=0)
    tokens_per_minute: int | None = Field(default=None, gt=0)
    requests_per_day: int | None = Field(default=None, gt=0)
    tokens_per_day: int | None = Field(default=None, gt=0)


class ModelProfile(BaseModel):
    """One named model choice: the provider, the model, its sampling settings and its limits.

    Attributes:
        provider (str): Name of an entry in the providers section.
        model (str): Model ID as the provider spells it.
        temperature (float): Sampling randomness. 0 gives the most repeatable answers.
        max_tokens (int): Longest reply the model may write, in tokens. For a reasoning model,
            its thinking counts too.
        reasoning_effort (Literal["low", "medium", "high"] | None): How long a reasoning
            model, such as gpt-oss, thinks before answering. None leaves it to the provider.
        limits (ModelLimits): Per-minute and per-day limits to stay under. Empty means none.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    limits: ModelLimits = ModelLimits()


class ModelsConfig(BaseModel):
    """The providers and model profiles from configs/models.yaml.

    Attributes:
        providers (dict[str, ProviderConfig]): Providers by name, such as "groq".
        models (dict[str, ModelProfile]): Model profiles by name. Experiments refer to these names.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: dict[str, ProviderConfig]
    models: dict[str, ModelProfile]

    @model_validator(mode="after")
    def _check_providers_exist(self) -> Self:
        """Make sure every profile names a provider that is defined.

        Returns:
            Self: The unchanged config.

        Raises:
            ValueError: If a profile names an unknown provider.
        """
        for name, profile in self.models.items():
            if profile.provider not in self.providers:
                known = ", ".join(sorted(self.providers))
                raise ValueError(
                    f"model {name!r} uses unknown provider {profile.provider!r} (known: {known})"
                )
        return self

    def profile(self, name: str) -> ModelProfile:
        """Look up a model profile by name.

        Args:
            name (str): Profile name, such as "gemini-2.5-flash".

        Returns:
            ModelProfile: The profile.

        Raises:
            ConfigError: If no profile has that name.
        """
        if name not in self.models:
            known = ", ".join(sorted(self.models))
            raise ConfigError(f"No model profile named {name!r}. Configured profiles: {known}")
        return self.models[name]

    def provider(self, name: str) -> ProviderConfig:
        """Look up a provider by name.

        Args:
            name (str): Provider name, such as "groq".

        Returns:
            ProviderConfig: The provider's settings.

        Raises:
            ConfigError: If no provider has that name.
        """
        if name not in self.providers:
            known = ", ".join(sorted(self.providers))
            raise ConfigError(f"No provider named {name!r}. Configured providers: {known}")
        return self.providers[name]


def describe_errors(exc: ValidationError) -> str:
    """Turn pydantic validation errors into short lines that don't repeat the bad values.

    Pydantic's own message includes each rejected value. A value could be an API key pasted
    into the wrong place, so we leave values out.

    Args:
        exc (ValidationError): The validation failure.

    Returns:
        str: One line per problem, such as "models.groq.temprature: Extra inputs are not permitted".
    """
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "(top level)"
        lines.append(f"  {where}: {error['msg']}")
    return "\n".join(lines)


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Read and check the model config file.

    Args:
        path (Path | None): The YAML file to read. None means configs/models.yaml,
            found from the current folder upwards.

    Returns:
        ModelsConfig: The checked providers and model profiles.

    Raises:
        ConfigError: If the file is missing, isn't valid YAML or doesn't match the expected shape.
    """
    path = path or find_config(MODELS_CONFIG)
    if not path.is_file():
        raise ConfigError(f"Model config not found: {path}")
    try:
        # safe_load only builds plain data (dicts, lists, strings). yaml.load could run code.
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return ModelsConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Problems in {path}:\n{describe_errors(exc)}") from None


def read_api_key(provider: ProviderConfig) -> str:
    """Read a provider's API key from the environment.

    Args:
        provider (ProviderConfig): The provider whose key to read.

    Returns:
        str: The API key. Never print or log it.

    Raises:
        ConfigError: If the environment variable is missing or empty.
    """
    key = os.environ.get(provider.api_key_env, "").strip()
    if not key:
        raise ConfigError(f"{provider.api_key_env} is not set. Add your API key to .env.")
    return key
