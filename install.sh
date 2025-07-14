#!/bin/bash
# One-click installation script for Crypto Price Alert Bot
# This script sets up a Python virtual environment, installs dependencies,
# and prepares the .env configuration file.

set -e

# Determine project root (directory of this script)
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Check for Python 3
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required but not found. Please install Python 3." >&2
    exit 1
fi

# Create virtual environment if not present
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
# shellcheck disable=SC1091
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

deactivate

# Create .env from example if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env. Please edit it with your configuration." 
fi

cat <<INFO
Installation complete.
To start the bot:
  source venv/bin/activate && python run.py
INFO
