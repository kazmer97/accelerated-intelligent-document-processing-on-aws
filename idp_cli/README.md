# IDP CLI

Command-line interface for the AWS GenAI IDP Accelerator.

## Overview

The IDP CLI provides a programmatic interface for batch document processing, evaluation workflows, and analytics integration.

## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install idp-cli
uv pip install idp-cli
```

### Using pip

```bash
pip install idp-cli
```

## Documentation

For detailed documentation, see [IDP CLI Documentation](../docs/idp-cli.md).

## Usage

```bash
# If installed with UV
uv run idp-cli --help

# If installed with pip
idp-cli --help
```

## Development

### Quick Start

```bash
# Clone the repository and navigate to idp_cli
cd idp_cli

# Install development dependencies with UV (includes testing and linting)
uv sync --group dev

# Run tests
uv run --group dev pytest

# Run linting
uv run --group dev ruff check

# Format code
uv run --group dev ruff format
```

### From Project Root

```bash
# Run all tests (including idp_cli tests)
make test

# Run linting and formatting
make lint

# Full workflow: setup + lint + test
make all
```

### Dependency Groups

The `dev` group includes everything needed for development: testing, linting, and dev tools.

For `idp_common` package testing, use `--all-extras` to install all optional features:

```bash
# For idp_common development/testing (includes all optional features)
cd lib/idp_common_pkg
uv sync --all-extras --group dev

# For idp_cli development/testing
cd idp_cli
uv sync --group dev
```

Note: `--all-extras` installs all optional features (ocr, classification, extraction, etc.) needed for comprehensive testing.
