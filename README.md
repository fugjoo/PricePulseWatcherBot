# PricePulseWatcherBot

Telegram bot that notifies you when a cryptocurrency moves by a chosen
percentage. Create a `.env` from the example and keep your
`TELEGRAM_TOKEN` secret. Talk to **@PricePulseWatcherBot** to get started.

## Features

 - Subscribe to trending coins and get alerts
 - Inline button to refresh the suggested coin
 - Suggest random coins from the top market cap list in the keyboard
- Autocompletion for all bot commands

## Quickstart

Ensure **Python 3.8+** is available and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # edit TELEGRAM_TOKEN
# DB_PATH sets the SQLite file used (default subs.db)
# COINGECKO_API_KEY is optional for higher rate limits
# COINGECKO_BASE_URL sets the CoinGecko endpoint (use the pro URL if you have a paid plan)
# LOG_LEVEL enables verbose output when set to DEBUG
# LOG_FILE writes logs to the given file (default bot.log)
cp config.json.example config.json
python run.py
```

Adjust `config.json` to change the default threshold, subscription interval or
price check frequency. Start the bot and run `/start` in a chat with it.

## Configuration

Create `.env` and `config.json` files from the provided examples. The env file holds credentials and runtime options:

- `TELEGRAM_TOKEN` – token from BotFather
- `DB_PATH` – SQLite database path (default `subs.db`)
- `COINGECKO_API_KEY` – optional CoinGecko key
- `COINGECKO_BASE_URL` – override to use the pro CoinGecko endpoint
- `PRICE_API_PROVIDER` – `coingecko` or `coinmarketcap`
- `COINMARKETCAP_API_KEY` – optional CoinMarketCap key
- `LOG_LEVEL` – log level such as INFO
- `LOG_FILE` – file to write logs to

`config.json` contains defaults controlling alert behaviour:

- `default_threshold` – percent change that triggers an alert
- `default_interval` – subscription interval when none is given
- `price_check_interval` – how often prices are checked

### Commands

- `/add <coin> [pct] [interval]` – subscribe to price alerts
- `/remove <coin>` – remove a subscription
- `/list` – list active subscriptions
- `/info <coin>` – show current coin data
- `/chart <coin> [days]` – plot price history (alias `/charts`)
- `/trends` – show trending coins
- `/global` – show global market stats

Intervals accept plain seconds or values like `1h`, `15m` or `30s`.

### One‑click install

Run the provided script to set up a virtual environment automatically:

```bash
./install.sh
```

## Contributing

Activate your virtual environment and run the following checks before opening a pull request:

```bash
isort .
black .
flake8
pytest
```

## License

Licensed under the [MIT License](LICENSE).
