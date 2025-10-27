#!/bin/bash
# Clean build script to remove all build artifacts and rebuild cleanly

echo "Cleaning all build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

echo "Syncing package with UV..."
cd ../.. && uv sync --all-extras --group dev

echo "Done!"
