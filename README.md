# PR reviewer

An AI pull-request reviewer for Python repos. It posts a comment only when it can back the comment with evidence. SWR-Bench measures whether that rule improves precision.

## Setup

You need conda (Anaconda or Miniconda) and Docker Desktop. Conda installs Python 3.12.

```
conda env create -f environment.yml
conda activate pr-reviewer
cp .env.example .env          # then fill in your API keys
```

When `pyproject.toml` gets new dependencies, run `pip install -e ".[dev]"` again inside the environment.

## Checks

```
pytest
ruff check .
mypy
```
