# Contributing to wasmagent-train-replay

Thanks for your interest in contributing! This guide covers setting up a local
development environment and running the tests, lint, and type checks before
opening a pull request.

## Prerequisites

- **Python 3.10+** (the package requires `>=3.10`, declared in `pyproject.toml`)
- `pip` (bundled with Python; upgrade with `python -m pip install --upgrade pip`)
- A virtual environment tool, e.g. `venv`:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  ```

## Install

Install the package in editable mode together with the development extras
(`pytest`, `pytest-cov`, `ruff`, `mypy`):

```bash
pip install -e ".[dev]"
```

You can also use the provided Makefile target:

```bash
make install
```

> No additional dependencies are required for development — everything you need
> to build, test, lint, and typecheck is declared in the `dev` extra above.

## Run tests

Run the full test suite with:

```bash
make test
```

or directly with pytest:

```bash
pytest tests/
```

`make test` additionally enables coverage reporting
(`pytest tests/ -v --cov=train_replay --cov-report=term-missing`).

## Lint and typecheck

Format/lint checks use [ruff](https://docs.astral.sh/ruff/), and static type
checks use [mypy](https://mypy-lang.org/) in strict mode:

```bash
ruff check train_replay tests
mypy train_replay
```

Or via the Makefile:

```bash
make lint
make typecheck
```

## Pull request checklist

Before opening a PR, please ensure:

- [ ] New code has corresponding tests in `tests/`
- [ ] `pytest tests/` (or `make test`) passes
- [ ] `ruff check train_replay tests` reports no issues
- [ ] `mypy train_replay` reports no type errors
- [ ] Type annotations are used throughout (mypy strict mode)
- [ ] No new dependencies are added without updating `pyproject.toml`

## Commit messages

Follow the existing commit style (e.g. `feat:`, `fix:`, `docs:`, `chore:`,
`test:`, `refactor:`). Keep the summary line concise and use the body to explain
the *why* behind non-trivial changes.
