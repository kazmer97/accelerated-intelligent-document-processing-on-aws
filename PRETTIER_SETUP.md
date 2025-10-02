# Code Formatting Setup

This project uses **pre-commit** hooks to automatically format code on commit.

## Why Pre-commit Instead of Just Prettier?

✅ **Unified Solution**: One tool for Python, YAML, JSON, Markdown  
✅ **No Node.js Required**: Uses Python's pre-commit (prettier runs via hook)  
✅ **Automatic on Commit**: Formats code automatically when you commit  
✅ **Ruff Integration**: Python linting + formatting in one fast tool  
✅ **Security Checks**: Detects secrets, validates YAML, checks file sizes

## What Gets Formatted

- **Python** (`.py`) → Ruff (linting + formatting)
- **YAML** (`.yaml`, `.yml`) → Prettier
- **JSON** (`.json`) → Prettier
- **Markdown** (`.md`) → Prettier
- **General** → Trailing whitespace, EOF newlines, merge conflicts

## Files Excluded from Formatting

The following files are **intentionally excluded** from formatting:

### Template/Example Files with Placeholders

Files containing `<TOKEN>` placeholders used by the build system:

- `**/testing/**/*.json` - Test fixtures with placeholder data
- `**/examples/**/*.json` - Example files with template tokens
- `**/*Example*.json` - Any file with "Example" in the name
- `**/*Template*.json` - Template files

### Build Artifacts

- `.aws-sam/` - SAM build outputs
- `node_modules/` - NPM dependencies
- `*.lock` files - Lock files

### Why Exclude Placeholders?

The project uses `<TOKEN>` syntax for build-time replacement (e.g., `<VERSION>`,
`<ARTIFACT_BUCKET_TOKEN>`). These tokens are replaced by `publish.py` during the
build process and must remain unchanged.

## Installation

### Option 1: Pre-commit (Recommended)

Install pre-commit hooks (Python-based, no Node.js required):

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

Now formatting happens automatically on `git commit`!

### Option 2: Standalone Tools

If you prefer manual control:

```bash
# For YAML/JSON/Markdown formatting only
npm install
```

## Usage

### With Pre-commit (Automatic)

Files are automatically formatted when you commit:

```bash
git add .
git commit -m "Your message"  # Formatting happens automatically
```

### Manual Formatting

#### All files (Python + YAML + JSON + Markdown)

```bash
pre-commit run --all-files
```

#### Only specific hooks

```bash
# Only Python formatting
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files

# Only YAML/JSON/Markdown formatting
pre-commit run prettier --all-files
```

#### Using NPM (YAML/JSON/Markdown only)

```bash
# Format all files
npm run format

# Format only YAML files
npm run format:yaml

# Check formatting (without changing files)
npm run format:check
```

## Configuration Files

### Pre-commit Configuration

`.pre-commit-config.yaml` - Defines all formatting hooks:

- **ruff**: Python linting and formatting
- **prettier**: YAML, JSON, Markdown formatting
- **pre-commit-hooks**: General file checks
- **detect-secrets**: Security scanning

### Prettier Configuration

`.prettierrc.json` - YAML/JSON/Markdown formatting settings:

### YAML Files

- **Print Width**: 120 characters
- **Tab Width**: 2 spaces
- **No Tabs**: Uses spaces for indentation
- **Single Quote**: `false` (uses double quotes in YAML)
- **Prose Wrap**: `preserve` (doesn't reformat long lines)

### CloudFormation Templates

- Special handling for `template.yaml` and `*.template.yaml` files
- Preserves CloudFormation intrinsic function formatting

## VS Code Integration

Install the Prettier extension:

1. Open VS Code
2. Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Windows/Linux)
3. Search for "Prettier - Code formatter"
4. Install the extension

### Configure VS Code Settings

Add to `.vscode/settings.json`:

```json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "[yaml]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.formatOnSave": true
  },
  "[yml]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.formatOnSave": true
  }
}
```

## Pre-commit Hook (Optional)

To automatically format files before committing, you can use husky and
lint-staged:

```bash
npm install --save-dev husky lint-staged
npx husky init
```

Add to `package.json`:

```json
{
  "lint-staged": {
    "*.{yaml,yml}": "prettier --write"
  }
}
```

Create `.husky/pre-commit`:

```bash
#!/bin/sh
npx lint-staged
```

## Files Ignored

See `.prettierignore` for the list of ignored files/directories:

- `node_modules/`
- `.aws-sam/`
- Build outputs
- Environment files
- Lock files
- And more...

## Testing

Test prettier on a specific file:

```bash
npx prettier --write template.yaml
```

Check formatting without changing:

```bash
npx prettier --check template.yaml
```
