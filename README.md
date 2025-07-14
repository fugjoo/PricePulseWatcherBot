# Crypto Price Alert Bot

Minimal async Telegram bot notifying when a coin hits a target price.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit variables
python run.py
```

### Docker

```bash
docker build -t crypto-bot .
docker run --env-file .env crypto-bot
```
