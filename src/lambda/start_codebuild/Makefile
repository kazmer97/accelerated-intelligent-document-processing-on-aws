# Makefile for StartUICodeBuild and Pattern2CodeBuildTrigger
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/../../../)

build-StartUICodeBuild:
	UV_NO_INSTALLER_METADATA=1 \
	UV_COMPILE_BYTECODE=1 \
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 --python-platform x86_64-manylinux2014 --project "$(PROJECT_ROOT)" --group lambda-start-codebuild
	cp *.py "$(ARTIFACTS_DIR)/"
	rm -rf "$(ARTIFACTS_DIR)"/__pycache__

build-Pattern2CodeBuildTrigger:
	UV_NO_INSTALLER_METADATA=1 \
	UV_COMPILE_BYTECODE=1 \
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 --python-platform x86_64-manylinux2014 --project "$(PROJECT_ROOT)" --group lambda-start-codebuild
	cp *.py "$(ARTIFACTS_DIR)/"
	rm -rf "$(ARTIFACTS_DIR)"/__pycache__

.PHONY: build-StartUICodeBuild build-Pattern2CodeBuildTrigger
