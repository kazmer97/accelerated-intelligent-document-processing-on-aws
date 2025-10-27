#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

##############################################################################################
# WSL Development Environment Setup Script
# 
# This script automates the installation of development tools for the GenAI IDP accelerator
# on Windows Subsystem for Linux (WSL) Ubuntu systems. It installs Python 3, AWS CLI, 
# SAM CLI, Node.js, and other essential development tools.
#
# Usage: ./wsl_setup.sh
# Note: Run this script inside WSL Ubuntu environment
##############################################################################################

# exit on failure
set -ex

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install git unzip -y
sudo apt install python3 python3-pip python3-venv python3-full -y
sudo apt install build-essential make -y

# Verify Python version
python3 --version

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -  # nosemgrep: bash.curl.security.curl-pipe-bash.curl-pipe-bash - Official NodeSource repository with HTTPS verification for development environment only
sudo apt-get install -y nodejs

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscli.zip"
unzip awscli.zip
sudo ./aws/install
rm -rf aws awscli.zip

# Verify AWS CLI installation
aws --version

# Install AWS SAM CLI
wget https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip
unzip aws-sam-cli-linux-x86_64.zip -d sam-installation
sudo ./sam-installation/install
rm -rf sam-installation aws-sam-cli-linux-x86_64.zip

# Verify SAM installation
sam --version

echo "==> Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Verify UV installation
if command -v uv >/dev/null 2>&1; then
  echo "UV installed successfully: $(uv --version)"
else
  echo "WARNING: UV installation may require shell restart"
  echo "Run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "DONE - WSL development environment setup complete."
echo ""
echo "Next steps:"
echo "1. Navigate to project root: cd /path/to/aws-idp"
echo "2. Initialize workspace: make init"
echo "   (This creates .venv and installs all dependencies with UV)"
echo "3. Configure AWS CLI: aws configure"
echo ""
echo "To use UV in new shells, add to ~/.bashrc:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
