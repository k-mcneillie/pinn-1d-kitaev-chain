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

# Clean up temporary cache directories and build artifacts, including
# __pycache__/.ipynb_checkpoints/*.pyc nested anywhere in the repo, not
# just the top level
clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build src/*.egg-info coverage.xml
    find . -name .git -prune -o -type d -name "__pycache__" -exec rm -rf {} +
    find . -name .git -prune -o -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
    find . -name .git -prune -o -type f -name "*.py[co]" -exec rm -f {} +

latex-build:
    @if [ "$$(basename $$(pwd))" = "write_up" ]; then \
        pdflatex main.tex && \
        bibtex main && \
        pdflatex main.tex && \
        pdflatex main.tex; \
    else \
        (cd docs/write_up && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex); \
    fi
    just latex-clean

latex-clean:
    @if [ "$$(basename $$(pwd))" = "write_up" ]; then \
        rm -f *.{aux,log,bbl,blg,out,toc,bib,fls,fdb_latexmk}; \
        rm -f texput.pdf; \
    else \
        rm -f docs/write_up/*.{aux,log,bbl,blg,out,toc,bib,fls,fdb_latexmk}; \
        rm -f docs/write_up/texput.pdf; \
        rm -f main.log; \
    fi
