# PricePulseWatcherBot

Telegram bot that notifies you when a cryptocurrency moves by a chosen
percentage. Create a `.env` from the example and keep your
`TELEGRAM_TOKEN` secret. Talk to **@PricePulseWatcherBot** to get started.

## Features

- Subscribe to trending coins and get alerts
- Inline button to refresh the suggested coin
- Autocompletion for all bot commands

## Quickstart

Ensure **Python 3.8+** is available and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # edit TELEGRAM_TOKEN
cp config.json.example config.json
python run.py
```

Adjust `config.json` to change the default threshold, subscription interval or
price check frequency. Start the bot and run `/start` in a chat with it.

### Commands

- `/subscribe <coin> [pct] [interval]` – subscribe to price alerts
- `/unsubscribe <coin>` – remove a subscription
- `/list` – list active subscriptions
- `/info <coin>` – show current coin data
- `/chart <coin> [days]` – plot price history (alias `/charts`)
- `/global` – show global market stats

Intervals accept plain seconds or values like `1h`, `15m` or `30s`.

### One‑click install

Run the provided script to set up a virtual environment automatically:

```bash
./install.sh
```

## License

Licensed under the [MIT License](LICENSE).
