# Default recipe to list all available commands
default:
    @just --list

# Run all quality gates (Lint, Format, Types, Tests)
check-all: lint format-check type-check test

# Run the pytest suite with code coverage tracking
test:
    pytest

# Run ruff linter and automatically fix safe code violations
lint:
    ruff check . --fix

# 🔍 Check formatting rules without changing files
format-check:
    ruff format --check .

# Automatically format all source files using ruff
format:
    ruff format .

# Run static type checking across the source directory
type-check:
    mypy src/

# Clean up temporary cache directories and build artifacts
clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build src/*.egg-info
