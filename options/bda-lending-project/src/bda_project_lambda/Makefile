# Makefile for BDAProjectLambda
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/../../../../)

build-BDAProjectLambda:
	UV_NO_INSTALLER_METADATA=1 \
	UV_COMPILE_BYTECODE=1 \
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 --project "$(PROJECT_ROOT)" --group lambda-bda-project
	cp *.py "$(ARTIFACTS_DIR)/"
	rm -rf "$(ARTIFACTS_DIR)"/__pycache__

.PHONY: build-BDAProjectLambda
