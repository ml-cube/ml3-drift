# https://github.com/casey/just

# Prevent showing the recipe name when running
# set quiet

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

# Synchronize the environment by installing all the dependencies
dev-sync:
    uv sync --cache-dir .uv_cache --all-extras

dev-sync-extra extra:
	uv sync --cache-dir .uv_cache --extra {{extra}}

# Synchronize the environment by installing all the dependencies except the dev ones
prod-sync:
	uv sync --cache-dir .uv_cache --all-extras --no-dev

# Install the pre-commit hooks
install-hooks:
	uv run pre-commit install

# Run ruff formatting
format:
	uv run ruff format

# Run ruff linting and mypy type checking
lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package ml3_drift

# Run the tests with pytest
# Default value for testWorkers, to run `just testWorkers=4`
testWorkers := "auto"
test:
	uv run pytest --verbose --color=yes -n {{testWorkers}} --exitfirst tests

# Run linters, formatters and tests
validate: format lint test

# --------------------------------------------------
# Documentation

# Generate the documentation
build-docs:
	uv run mkdocs build

# Serve the documentation locally
serve-docs:
	uv run mkdocs serve

# --------------------------------------------------
# Publishing
publish new_version:
	# just publish new_version=0.0.1
	# The __version__ variable in src/ml3_drift/__init__.py must be updated manually as of now.
	# The build tool retrieves it from there.
	# We'll fix this soon :)
	git tag -a v{{ new_version }} -m "Release v{{ new_version }}"
	git push origin v{{ new_version }}
