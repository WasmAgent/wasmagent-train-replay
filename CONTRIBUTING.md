# Contributing to wasmagent-train-replay

Thanks for your interest in contributing! This guide covers setting up a
local development environment, running the test suite, and linting your
changes before opening a pull request.

## Prerequisites

- **Python 3.10+** (3.10, 3.11, and 3.12 are all supported)
- `pip` (bundled with Python; upgrade with `pip install --upgrade pip`)
- `git`
- `make` (optional — only needed if you prefer `make test` over invoking
  `pytest` directly)

Confirm your Python version before continuing:

```bash
python --version   # must report 3.10 or higher
```

## Installation

Clone the repository and install the package in editable mode along with
its development extras:

```bash
git clone https://github.com/WasmAgent/wasmagent-train-replay.git
cd wasmagent-train-replay
pip install -e ".[dev]"
```

The `.[dev]` extra pulls in the tooling needed for development and testing
(`pytest`, `ruff`, `mypy`, and `hatchling` build backend).

It is recommended (but not required) to work inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

Run the full test suite with either of the following:

```bash
make test          # if you have make installed
# or
pytest tests/
```

Both commands execute the `tests/` suite via `pytest`. Aim to keep the full
suite green before submitting changes.

## Linting and type checking

All new code must pass linting and type checks:

```bash
ruff check         # lint (catches unused imports, style issues, etc.)
mypy               # static type checking (strict mode)
```

Fix everything these report before opening a pull request — the project
enforces both in review.

## Pull request checklist

- [ ] New code has corresponding tests in `tests/`
- [ ] Type annotations are used throughout (passes `mypy`)
- [ ] `ruff check` reports no issues
- [ ] `pytest tests/` (or `make test`) passes
- [ ] No new dependencies added without updating `pyproject.toml`

Happy hacking!
