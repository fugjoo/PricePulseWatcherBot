# Crypto Price Alert Bot

Simple Telegram bot that sends the current Bitcoin price every minute after the
`/start` command.

## Quickstart

Ensure you have **Python 3.8 or newer** available:

```bash
python3 --version
```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit variables
python run.py
```

Once the bot is running, send `/start` in a chat with your bot to receive the
current Bitcoin price every minute.

### One-click install

Simply run the provided script to set up everything at once. If Python 3.8+
is not available, the script will attempt to install it via `apt`, `yum` or
`dnf` (requires sudo privileges):

```bash
./install.sh
```

### Docker

```bash
docker build -t crypto-bot .
docker run --env-file .env crypto-bot
```
