# Lambda Dependency Groups

## Overview

Lambda functions use **dependency groups** defined in the root `/pyproject.toml` to manage their Python dependencies. This approach provides:

- **Centralized dependency management** - All Lambda dependencies in one place
- **Minimal package sizes** - Each Lambda only gets what it needs
- **Consistent versions** - UV lock file ensures reproducible builds
- **Fast builds** - UV's parallel resolver and caching

## How It Works

1. **Dependency groups** are defined in `/pyproject.toml` under `[dependency-groups]`
2. Each Lambda has a **Makefile** that references its dependency group
3. **SAM Build** uses the Makefile to install dependencies with `uv pip install --target`
4. Dependencies are **locked** via `uv.lock` for reproducibility

## Dependency Group Naming

Groups follow this pattern:

- `lambda-<feature>` - Feature-specific functions (e.g., `lambda-ocr`, `lambda-classification`)
- `lambda-<component>` - Component-specific functions (e.g., `lambda-agent-processor`, `lambda-queue`)
- `lambda-minimal` - Functions with minimal dependencies (boto3 only)
- `lambda-document-resolver` - GraphQL resolvers for document operations

## Lambda Makefile Template

Each Lambda directory contains a `Makefile`:

```makefile
build-<FunctionName>:
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 "<path-to-root>[<dependency-group>]"
	cp *.py "$(ARTIFACTS_DIR)/"
	rm -rf "$(ARTIFACTS_DIR)"/__pycache__

.PHONY: build-<FunctionName>
```

**Example** (OCR function at `patterns/pattern-2/src/ocr_function/`):

```makefile
build-OCRFunction:
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 "../../../../[lambda-ocr]"
	cp *.py "$(ARTIFACTS_DIR)/"
	rm -rf "$(ARTIFACTS_DIR)"/__pycache__
```

## SAM Template Configuration

Each Lambda must include this metadata:

```yaml
Metadata:
  BuildMethod: makefile
```

## Adding a New Lambda

1. **Define dependency group** in `/pyproject.toml`:
   ```toml
   [dependency-groups]
   lambda-my-feature = [
       "idp_common[core,my-feature]",
       "other-dependency>=1.0.0",
   ]
   ```

2. **Lock dependencies**: `uv lock`

3. **Create Makefile** in lambda directory with build target

4. **Update SAM template** with `BuildMethod: makefile` metadata

5. **Build**: `sam build MyFunction`

## Updating Dependencies

1. Edit dependency group in `/pyproject.toml`
2. Run `uv lock`
3. Rebuild: `sam build <FunctionName>`

## Finding Lambda Mappings

To find which dependency group a Lambda uses:

```bash
# Find the Lambda's Makefile
find patterns/ src/lambda/ -name "Makefile" -path "*/<lambda-dir>/Makefile"

# Check the dependency group in the uv pip install line
grep "uv pip install" <lambda-dir>/Makefile
```
