# https://github.com/casey/just

# Prevent showing the recipe name when running
set quiet

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

# --------------------------------------------------
# Developer Setup

# Synchronize the environment by installing all the dependencies
dev-sync:
    uv sync --cache-dir .uv_cache --all-extras

# Synchronize the environment by installing the specified extra dependency
# Currently used within the CI to install extra dependencies and test them.
dev-sync-extra extra:
	uv sync --cache-dir .uv_cache --extra {{extra}}

# Synchronize the environment by installing all the dependencies except the dev ones
prod-sync:
	uv sync --cache-dir .uv_cache --all-extras --no-default-groups

# Synchronize the environment by installing the extra dependency
# specified. Doesn't install the dev dependencies.
prod-sync-extra extra:
	uv sync --cache-dir .uv_cache --extra {{extra}} --no-default-groups

# Install the pre-commit hooks
install-hooks:
	uv run pre-commit install

# --------------------------------------------------
# Validation

# Run ruff formatting
format:
	uv run ruff format

# Run ruff linting and mypy type checking
lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package ml3_drift


# Default value for testWorkers is auto (meaning all workers available)
# If you want to pass a custom value (such as 4): `just testWorkers=4 test`
# We also run ruff on tests files (it's so fast that it's worth it)

# Little caveat: when running tests with only an extra installed, you'd like
# to avoid having docs dependencies installed (since, for instance, a mkdocs plugin
# requires Pandas, which is one of our extra dependencies). This happens by default
# since docs dependencies are not installed as default dependencies by uv (see pyproject.toml).
# They are only installed when building / serving the documentation. However, if you first
# build the documentation, then run the tests, you will have the docs dependencies installed.
# Should not be a practical problem (especially since in CI environments we don't install docs dependencies),
# but it's worth noting.

# Run the tests with pytest
testWorkers := "auto"
test:
    uv run ruff format tests
    uv run ruff check tests --fix
    uv run pytest --verbose --color=yes -n {{testWorkers}} --exitfirst tests

# Run linters, formatters and tests
validate: format lint test

# --------------------------------------------------
# Documentation

# Generate the documentation
build-docs:
    # Make sure mkdocs is installed
    uv run --group docs mkdocs build

# Serve the documentation locally
serve-docs:
    uv run --group docs mkdocs serve

# --------------------------------------------------
# Publishing
publish new_version:
	# just publish new_version=0.0.1
	# The __version__ variable in src/ml3_drift/__init__.py must be updated manually as of now.
	# The build tool retrieves it from there.
	# We'll fix this soon :)
	git tag -a v{{ new_version }} -m "Release v{{ new_version }}"
	git push origin v{{ new_version }}
