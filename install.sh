#!/bin/bash
# One-click installation script for Crypto Price Alert Bot
# This script sets up a Python virtual environment, installs dependencies,
# and prepares the .env configuration file.

set -euo pipefail

# Determine project root (directory of this script)
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Check for Python 3.8+
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" >/dev/null 2>&1; then

    if command -v apt-get >/dev/null 2>&1; then
        echo "Python 3.8+ not found. Installing python3.8..."
        sudo apt-get update
        sudo apt-get install -y python3.8 python3.8-venv
        PYTHON=python3.8
    elif command -v yum >/dev/null 2>&1; then
        echo "Python 3.8+ not found. Installing python3 via yum..."
        sudo yum install -y python3
        PYTHON=python3
    elif command -v dnf >/dev/null 2>&1; then
        echo "Python 3.8+ not found. Installing python3 via dnf..."
        sudo dnf install -y python3
        PYTHON=python3
    else
        echo "Python 3.8+ is required but automatic installation failed." >&2
        echo "Please install Python 3.8 or newer and re-run this script." >&2
        exit 1
    fi

    echo "Python 3.8+ is required but not found. Please install Python 3.8 or newer." >&2
    exit 1

fi

# Verify that the Python version meets the minimum requirement
if ! "$PYTHON" - <<'EOF'
import sys
sys.exit(0 if sys.version_info >= (3, 8) else 1)
EOF
then
    echo "Python 3.8 or higher is required. Current version: $($PYTHON --version)" >&2
    exit 1
fi

# Create virtual environment if not present
if [ ! -d "venv" ]; then
    "$PYTHON" -m venv venv
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
