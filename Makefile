.PHONY: install test lint typecheck demo clean

install:
	uv sync --frozen --extra dev

test:
	pytest tests/ -v --cov=train_replay --cov-report=term-missing

lint:
	ruff check train_replay tests

typecheck:
	mypy train_replay

demo:
	python examples/fault_injection_demo.py

clean:
	rm -rf dist build *.egg-info .mypy_cache .pytest_cache .coverage
