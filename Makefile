.PHONY: check format test types verify

test:
	uv run pytest

check:
	make test
	make format
	make types

format:
	uv run ruff check --fix
	uv run ruff format .

types:
	uv run mypy src/mma tests/

coverage:
	uv run pytest --cov=src/mma

# Run all tests and all linters
verify: test format
