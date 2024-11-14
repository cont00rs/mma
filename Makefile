.PHONY: format test verify

test:
	uv run pytest

format:
	uv run ruff check --fix

coverage:
	uv run pytest --cov=src/mma

# Run all tests and all linters
verify: test format
