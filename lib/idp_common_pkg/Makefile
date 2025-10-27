.PHONY: test test-cicd clean dev

# Use UV for all operations
UV := $(shell command -v uv 2> /dev/null)

test: test-unit

test-unit:
	$(UV) run --all-extras --group dev pytest -m "unit"

test-integration:
	$(UV) run --all-extras --group dev pytest -m "integration"

test-cicd: test-unit-cicd

test-unit-cicd:
	$(UV) sync --all-extras --group dev
	$(UV) run --all-extras --group dev pytest -m "unit" --cov=idp_common --cov-report=xml:test-reports/coverage.xml --cov-report=term --junitxml=test-reports/test-results.xml tests/

clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test-results.xml
	rm -rf test-reports
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	
# A convenience target to sync all features and dev dependencies
dev:
	$(UV) sync --all-extras --group dev
