# Contributing to wasmagent-train-replay

Thank you for your interest in contributing! This guide will help you set up the
development environment and run tests.

## Prerequisites

- **Python 3.10+** — this project requires Python 3.10 or newer.

## Install

Clone the repository and install the package in editable mode with development
dependencies:

```bash
git clone https://github.com/WasmAgent/wasmagent-train-replay.git
cd wasmagent-train-replay
pip install -e ".[dev]"
```

## Run tests

You can run the full test suite with:

```bash
make test
```

or equivalently:

```bash
pytest tests/
```

## Lint and type-check

We use **ruff** for linting and **mypy** for static type checking:

```bash
ruff check train_replay tests
mypy train_replay
```

Both are included in the `[dev]` extra, so they are already available after
running the install step above.

## Code style

- All new code must include type annotations (mypy strict mode).
- Follow the ruff lint rules — run `ruff check` before submitting changes.
- All new features and bug fixes must have corresponding tests in `tests/`.
- Do not add new dependencies without updating `pyproject.toml`.
