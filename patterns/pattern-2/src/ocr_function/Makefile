# Makefile for OCRFunction
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/../../../../)

build-OCRFunction:
	# Export dependencies excluding workspace members (idp-common)
	cd "$(PROJECT_ROOT)" && \
	uv export --frozen --no-emit-workspace --no-dev --no-editable --group lambda-ocr -o "$(ARTIFACTS_DIR)/requirements.txt"
	# Install PyPI dependencies
	UV_NO_INSTALLER_METADATA=1 \
	UV_COMPILE_BYTECODE=1 \
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 -r "$(ARTIFACTS_DIR)/requirements.txt"
	rm "$(ARTIFACTS_DIR)/requirements.txt"
	# Install workspace member (idp-common) as regular package
	UV_NO_INSTALLER_METADATA=1 \
	UV_COMPILE_BYTECODE=1 \
	uv pip install --target "$(ARTIFACTS_DIR)" --python 3.12 "$(PROJECT_ROOT)/lib/idp_common_pkg"
	# Copy Lambda handler code
	cp *.py "$(ARTIFACTS_DIR)/"
	rm -rf "$(ARTIFACTS_DIR)"/__pycache__

.PHONY: build-OCRFunction
