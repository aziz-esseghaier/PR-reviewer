"""Tests for loading and checking the model config."""

from pathlib import Path

import pytest

from pr_reviewer.core.config import (
    MODELS_CONFIG,
    ConfigError,
    ProviderConfig,
    find_config,
    load_models_config,
    read_api_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_CONFIG = """
providers:
  fake:
    base_url: https://llm.example.test/v1
    api_key_env: FAKE_LLM_KEY
models:
  fake-small:
    provider: fake
    model: fake-model-v1
"""


def write_config(folder: Path, text: str) -> Path:
    """Write a model config file into a folder.

    Args:
        folder (Path): Folder to write into.
        text (str): YAML content.

    Returns:
        Path: The written file.
    """
    path = folder / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_config_loads_with_defaults(tmp_path: Path) -> None:
    """Check that a minimal config loads and unset settings get their defaults."""
    config = load_models_config(write_config(tmp_path, VALID_CONFIG))
    profile = config.profile("fake-small")
    assert profile.provider == "fake"
    assert profile.temperature == 0.0
    assert profile.max_tokens == 1024
    assert config.provider("fake").timeout_s == 120.0


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    """Check that a profile pointing at a provider that doesn't exist fails with its name."""
    text = VALID_CONFIG.replace("provider: fake", "provider: nowhere")
    with pytest.raises(ConfigError, match="fake-small.*nowhere"):
        load_models_config(write_config(tmp_path, text))


def test_misspelled_setting_is_rejected(tmp_path: Path) -> None:
    """Check that a typo such as 'temprature' fails instead of being silently ignored."""
    text = VALID_CONFIG + "    temprature: 0.5\n"
    with pytest.raises(ConfigError, match="temprature"):
        load_models_config(write_config(tmp_path, text))


def test_error_message_does_not_repeat_bad_values(tmp_path: Path) -> None:
    """Check that a key pasted where the variable name belongs isn't echoed in the error."""
    text = VALID_CONFIG.replace("FAKE_LLM_KEY", "gsk_pretend_secret_value")
    with pytest.raises(ConfigError) as exc_info:
        load_models_config(write_config(tmp_path, text))
    assert "api_key_env" in str(exc_info.value)
    assert "gsk_pretend_secret_value" not in str(exc_info.value)


def test_broken_yaml_is_reported(tmp_path: Path) -> None:
    """Check that a YAML syntax error becomes a ConfigError."""
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_models_config(write_config(tmp_path, "providers: [unclosed"))


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """Check that a missing config file becomes a ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_models_config(tmp_path / "missing.yaml")


def test_unknown_profile_lists_the_configured_ones(tmp_path: Path) -> None:
    """Check that asking for an unknown profile names the profiles that do exist."""
    config = load_models_config(write_config(tmp_path, VALID_CONFIG))
    with pytest.raises(ConfigError, match="fake-small"):
        config.profile("nope")


def test_find_config_searches_parent_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that the config is found when the command runs from a subfolder."""
    (tmp_path / "configs").mkdir()
    write_config(tmp_path / "configs", VALID_CONFIG)
    subfolder = tmp_path / "src" / "deep"
    subfolder.mkdir(parents=True)
    monkeypatch.chdir(subfolder)
    assert find_config(MODELS_CONFIG) == tmp_path / MODELS_CONFIG


def test_read_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the key comes from the named variable, and a missing one names the variable."""
    provider = ProviderConfig(base_url="https://llm.example.test/v1", api_key_env="FAKE_LLM_KEY")
    monkeypatch.setenv("FAKE_LLM_KEY", "  test-key  ")
    assert read_api_key(provider) == "test-key"
    monkeypatch.delenv("FAKE_LLM_KEY")
    with pytest.raises(ConfigError, match="FAKE_LLM_KEY"):
        read_api_key(provider)


def test_project_model_config_is_valid() -> None:
    """Check that configs/models.yaml loads and offers models from at least two providers."""
    config = load_models_config(REPO_ROOT / MODELS_CONFIG)
    providers_used = {profile.provider for profile in config.models.values()}
    assert len(providers_used) >= 2
