# Makefile for IDP Accelerator - UV + Hatchling build system

# Define color codes
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m  # No Color

# Virtual environment and UV paths
VENV := .venv
UV := $(shell command -v uv 2> /dev/null)

# Default target - ensure UV and venv, then run lint and test
all: setup lint test

# Install UV if not present
install-uv:
ifndef UV
	@printf "$(YELLOW)📦 UV not found. Installing UV...$(NC)\n"
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@printf "$(GREEN)✅ UV installed!$(NC)\n"
	@printf "$(YELLOW)⚠️  Please restart your shell or run: source ~/.bashrc (or ~/.zshrc)$(NC)\n"
	@printf "$(YELLOW)⚠️  Then re-run make$(NC)\n"
	@exit 1
else
	@printf "$(GREEN)✅ UV is already installed: $(UV)$(NC)\n"
endif

# Create .venv using UV if it doesn't exist
$(VENV):
	@printf "$(BLUE)🏗️  Creating virtual environment with UV...$(NC)\n"
	@$(UV) venv
	@printf "$(GREEN)✅ Virtual environment created at $(VENV)$(NC)\n"

# Setup: ensure UV is installed and create venv
setup: install-uv $(VENV)
	@printf "$(GREEN)✅ Setup complete!$(NC)\n"

# Initialize UV environment (creates .venv and syncs all dependencies)
init: setup
	@printf "$(BLUE)🚀 Initializing UV workspace...$(NC)\n"
	@$(UV) sync --all-extras --group dev
	@printf "$(GREEN)✅ Workspace initialized with all features + dev dependencies$(NC)\n"

# Lock dependencies
lock: install-uv
	@printf "$(BLUE)🔒 Locking dependencies...$(NC)\n"
	@$(UV) lock
	@printf "$(GREEN)✅ Dependencies locked (uv.lock updated)$(NC)\n"

# Sync local development environment
sync: setup
	@printf "$(BLUE)📦 Syncing development environment...$(NC)\n"
	@$(UV) sync --all-extras --group dev
	@printf "$(GREEN)✅ Environment synced with all features$(NC)\n"

# Update dependencies
update: setup
	@printf "$(BLUE)⬆️  Updating dependencies...$(NC)\n"
	@$(UV) lock --upgrade
	@printf "$(GREEN)✅ Dependencies updated$(NC)\n"

# Update specific package
update-package: setup
ifndef PKG
	@printf "$(RED)❌ ERROR: PKG variable not set$(NC)\n"
	@echo "Usage: make update-package PKG=boto3"
	@exit 1
endif
	@printf "$(BLUE)⬆️  Updating $(PKG)...$(NC)\n"
	@$(UV) lock --upgrade-package $(PKG)
	@printf "$(GREEN)✅ $(PKG) updated$(NC)\n"

# Build idp_common package
build-idp-common: setup
	@printf "$(BLUE)🔨 Building idp_common package...$(NC)\n"
	@cd lib/idp_common_pkg && $(UV) build
	@printf "$(GREEN)✅ idp_common built$(NC)\n"

# Build idp_cli package
build-idp-cli: setup
	@printf "$(BLUE)🔨 Building idp_cli package...$(NC)\n"
	@cd idp_cli && $(UV) build
	@printf "$(GREEN)✅ idp_cli built$(NC)\n"

# Build all Python packages
build-packages: build-idp-common build-idp-cli
	@printf "$(GREEN)✅ All packages built$(NC)\n"

# Run tests in idp_common_pkg and idp_cli directories
test: setup
	@printf "$(BLUE)🧪 Running tests...$(NC)\n"
	@cd lib/idp_common_pkg && $(UV) run --all-extras --group dev pytest -m "unit"
	@cd idp_cli && $(UV) run --group dev pytest -v
	@printf "$(GREEN)✅ All tests passed$(NC)\n"

# Run linting checks and fix issues automatically
ruff-lint: setup
	@printf "$(BLUE)🔍 Running ruff linting...$(NC)\n"
	@$(UV) run --group dev ruff check --fix
	@printf "$(GREEN)✅ Linting complete$(NC)\n"

# Format code according to project standards
format: setup
	@printf "$(BLUE)✨ Formatting code...$(NC)\n"
	@$(UV) run --group dev ruff format
	@printf "$(GREEN)✅ Formatting complete$(NC)\n"

# Run both linting and formatting in one command
lint: ruff-lint format check-arn-partitions

# CI/CD version of lint that only checks but doesn't modify files
# Used in CI pipelines to verify code quality without making changes
lint-cicd: setup
	@printf "$(BLUE)Running code quality checks...$(NC)\n"
	@if ! $(UV) run --group dev ruff check; then \
		printf "$(RED)ERROR: Ruff linting failed!$(NC)\n"; \
		printf "$(YELLOW)Please run 'make ruff-lint' locally to fix these issues.$(NC)\n"; \
		exit 1; \
	fi
	@if ! $(UV) run --group dev ruff format --check; then \
		printf "$(RED)ERROR: Code formatting check failed!$(NC)\n"; \
		printf "$(YELLOW)Please run 'make format' locally to fix these issues.$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(GREEN)All code quality checks passed!$(NC)\n"

# Check CloudFormation templates for hardcoded AWS partition ARNs and service principals
check-arn-partitions:
	@printf "$(BLUE)Checking CloudFormation templates for hardcoded ARN partitions and service principals...$(NC)\n"
	@FOUND_ISSUES=0; \
	for template in template.yaml patterns/*/template.yaml patterns/*/sagemaker_classifier_endpoint.yaml options/*/template.yaml; do \
		if [ -f "$$template" ]; then \
			echo "Checking $$template..."; \
			ARN_MATCHES=$$(grep -n "arn:aws:" "$$template" | grep -v "arn:\$${AWS::Partition}:" || true); \
			if [ -n "$$ARN_MATCHES" ]; then \
				echo -e "$(RED)ERROR: Found hardcoded 'arn:aws:' references in $$template:$(NC)"; \
				echo "$$ARN_MATCHES" | sed 's/^/  /'; \
				echo -e "$(YELLOW)  These should use 'arn:\$${AWS::Partition}:' instead for GovCloud compatibility$(NC)"; \
				FOUND_ISSUES=1; \
			fi; \
			SERVICE_MATCHES=$$(grep -n "\.amazonaws\.com" "$$template" | grep -v "\$${AWS::URLSuffix}" | grep -v "^[[:space:]]*#" | grep -v "Description:" | grep -v "Comment:" | grep -v "cognito" | grep -v "ContentSecurityPolicy" || true); \
			if [ -n "$$SERVICE_MATCHES" ]; then \
				echo -e "$(RED)ERROR: Found hardcoded service principal references in $$template:$(NC)"; \
				echo "$$SERVICE_MATCHES" | sed 's/^/  /'; \
				echo -e "$(YELLOW)  These should use '\$${AWS::URLSuffix}' instead of 'amazonaws.com' for GovCloud compatibility$(NC)"; \
				echo -e "$(YELLOW)  Example: 'lambda.amazonaws.com' should be 'lambda.\$${AWS::URLSuffix}'$(NC)"; \
				FOUND_ISSUES=1; \
			fi; \
		fi; \
	done; \
	if [ $$FOUND_ISSUES -eq 0 ]; then \
		echo -e "$(GREEN)✅ No hardcoded ARN partition or service principal references found!$(NC)"; \
	else \
		echo -e "$(RED)❌ Found hardcoded references that need to be fixed for GovCloud compatibility$(NC)"; \
		exit 1; \
	fi

# Clean up build artifacts and caches
clean:
	@printf "$(BLUE)🧹 Cleaning build artifacts...$(NC)\n"
	@rm -rf .venv
	@rm -rf lib/idp_common_pkg/dist lib/idp_common_pkg/build lib/idp_common_pkg/*.egg-info
	@rm -rf idp_cli/dist idp_cli/build idp_cli/*.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@printf "$(GREEN)✅ Cleanup complete$(NC)\n"

# Clean everything including uv.lock (use with caution)
clean-all: clean
	@printf "$(YELLOW)⚠️  Removing uv.lock...$(NC)\n"
	@rm -f uv.lock
	@printf "$(GREEN)✅ Deep cleanup complete$(NC)\n"

# Show help
help:
	@printf "$(BLUE)IDP Accelerator Makefile - UV-based Python Development$(NC)\n"
	@echo ""
	@printf "$(YELLOW)Setup Targets:$(NC)"
	@echo "  make setup          - Install UV and create .venv (automatically done by other targets)"
	@echo "  make init           - Initialize workspace with dev dependencies"
	@echo "  make sync           - Sync development environment"
	@echo ""
	@printf "$(YELLOW)Development Targets:$(NC)"
	@echo "  make lint           - Run linting and formatting"
	@echo "  make ruff-lint      - Run ruff linting with auto-fix"
	@echo "  make format         - Format code with ruff"
	@echo "  make test           - Run all tests"
	@echo ""
	@printf "$(YELLOW)Dependency Management:$(NC)"
	@echo "  make lock           - Lock dependencies (update uv.lock)"
	@echo "  make update         - Update all dependencies"
	@echo "  make update-package PKG=<name>  - Update specific package"
	@echo ""
	@printf "$(YELLOW)Build Targets:$(NC)"
	@echo "  make build-idp-common  - Build idp_common package"
	@echo "  make build-idp-cli     - Build idp_cli package"
	@echo "  make build-packages    - Build all packages"
	@echo ""
	@printf "$(YELLOW)Cleanup Targets:$(NC)"
	@echo "  make clean          - Remove .venv and build artifacts"
	@echo "  make clean-all      - Remove .venv, build artifacts, and uv.lock"
	@echo ""
	@printf "$(YELLOW)Other Targets:$(NC)"
	@echo "  make all            - Run setup, lint, and test"
	@echo "  make check-arn-partitions  - Check CFN templates for GovCloud compatibility"
	@echo "  make help           - Show this help message"

# A convenience Makefile target that runs 
commit: lint test
	$(info Generating commit message...)
	export COMMIT_MESSAGE="$(shell q chat --no-interactive --trust-all-tools "Understand pending local git change and changes to be committed, then infer a commit message. Return this commit message only" | tail -n 1 | sed 's/\x1b\[[0-9;]*m//g')" && \
	git add . && \
	git commit -am "$${COMMIT_MESSAGE}" && \
	git push

.PHONY: all setup install-uv init lock sync update update-package \
        build-idp-common build-idp-cli build-packages test ruff-lint format \
        lint lint-cicd check-arn-partitions clean clean-all help commit
