# Crypto Price Alert Bot

Telegram bot that sends price alerts for your favourite cryptocurrency.
Users can subscribe to coins and receive a message when the price changes more
than a chosen percentage.

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

Start the bot and use `/start` in a chat with your bot. You can then subscribe
to coin alerts with:

```bash
/subscribe <coin> [percent]
```

List active subscriptions with `/list` and remove them using
`/unsubscribe <coin>`.

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
