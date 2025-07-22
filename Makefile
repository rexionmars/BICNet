.PHONY: install test lint format clean docs

# Installation
install:
	uv sync

install-dev:
	uv sync --dev

# Testing
test:
	uv run pytest tests/

test-cov:
	uv run pytest tests/ --cov=bicnet --cov-report=html

# Code quality
lint:
	uv run flake8 bicnet/
	uv run mypy bicnet/

format:
	uv run black bicnet/ tests/

format-check:
	uv run black --check bicnet/ tests/

# Examples
run-examples:
	uv run python bicnet/examples/main.py

run-brain-example:
	uv run python bicnet/examples/brain_example.py

run-rat-example:
	uv run python bicnet/examples/rat_example.py

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .coverage htmlcov/

# Documentation
docs:
	@echo "Documentation available in:"
	@echo "  - README.md: Overview and quick start"
	@echo "  - INSTALL.md: Installation instructions"
	@echo "  - CONTRIBUTING.md: Development guide"
	@echo "  - bicnet/docs/API.md: API reference"

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make lint' to check code quality"