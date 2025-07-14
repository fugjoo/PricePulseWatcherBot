# AGENT Instructions

This repository hosts a minimal async Telegram bot notifying when a coin hits a target price.

## Setup
- Use **Python 3.8+**. Create a virtual environment and run `pip install -r requirements.txt`.
- Duplicate `.env.example` to `.env` and fill in your configuration. **Never commit real API tokens**.
- Use `install.sh` for one-click environment setup. It should remain executable.

## Development Guidelines
- Keep commits focused and descriptive in the present tense ("Add logging" not "Added logging").
- Ensure the work tree is clean (`git status`) before committing.
- Format Python with `black` and sort imports using `isort`.
- Run `flake8` and any unit tests with `pytest` if present.
- Lint shell scripts with `shellcheck` when modified.

## Pull Requests
- Summaries should include file and line citations where relevant.
- Mention any test results and note if commands could not be run.
- Do not expose secrets or credentials in commits or PR messages.

