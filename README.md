# Crypto Price Alert Bot

Telegram bot that sends price alerts for your favourite cryptocurrency.
Users can subscribe to coins and receive a message when the price changes more
than a chosen percentage.

Create a `.env` file from the provided example and keep your
`TELEGRAM_TOKEN` private.

## Quickstart

Ensure you have **Python 3.8 or newer** available:

```bash
python3 --version
```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit TELEGRAM_TOKEN
cp config.json.example config.json  # optional defaults
python run.py
```

Adjust values in `config.json` to change the default alert threshold, the
default subscription interval or how often the bot queries the API.

The bot uses APScheduler to check prices every 60 seconds by default. Ensure the
`python-telegram-bot` package is installed with the `job-queue` extra as
specified in `requirements.txt`.

Start the bot and use `/start` in a chat with your bot. You can then subscribe
to coin alerts with:

```bash
/subscribe <coin> [percent] [interval]
```

Intervals can be specified in seconds or with suffixes like `1h`, `15m` or `30s`.

`price_check_interval` in `config.json` controls how often the bot polls the
API for price updates.

List active subscriptions with `/list` and remove them using `/unsubscribe <coin>`.

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

## License

This project is licensed under the [MIT License](LICENSE).
