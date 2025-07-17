# PricePulseWatcherBot

Telegram bot that notifies you when a cryptocurrency moves by a chosen
percentage. Create a `.env` from the example and keep your
`TELEGRAM_TOKEN` secret. Talk to **@PricePulseWatcherBot** to get started.

## Features

- Subscribe to trending coins and get alerts
- Suggest random coins from the top market cap list in the keyboard
- Autocompletion for all bot commands
- Monitor API health with `/status`
- Check recent coin news via `/news` (CryptoCompare)
- Optional futures liquidation alerts

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
# LOG_FILE writes logs to the given file (default bot.log) and recreates it if removed
# DEFAULT_THRESHOLD and DEFAULT_INTERVAL control new subscriptions
# PRICE_CHECK_INTERVAL sets how often prices are fetched
# ENABLE_MILESTONE_ALERTS toggles milestone notifications
# ENABLE_LIQUIDATION_ALERTS toggles futures liquidation alerts
# DEFAULT_VS_CURRENCY sets the reference currency used for prices
python run.py
```

Edit `.env` to change the default threshold, subscription interval or price
check frequency. Start the bot and run `/start` in a chat with it.

## Configuration

Create a `.env` file from the example. It holds credentials and runtime options:

- `TELEGRAM_TOKEN` – token from BotFather
- `DB_PATH` – SQLite database path (default `subs.db`)
- `COINGECKO_API_KEY` – optional CoinGecko key
- `COINGECKO_BASE_URL` – override to use the pro CoinGecko endpoint
- `LOG_LEVEL` – log level such as INFO
- `LOG_FILE` – file to write logs to (recreated if removed)

- `DEFAULT_THRESHOLD` – percent change that triggers an alert
- `DEFAULT_INTERVAL` – subscription interval when none is given
- `PRICE_CHECK_INTERVAL` – how often prices are checked
- `ENABLE_MILESTONE_ALERTS` – send messages for price milestones
- `ENABLE_LIQUIDATION_ALERTS` – enable liquidation event alerts
- `DEFAULT_VS_CURRENCY` – default currency used for API requests

### Commands

- `/add <coin> [pct] [interval]` – subscribe to price alerts
- `/remove <coin>` – remove a subscription
- `/clear` – remove all subscriptions
- `/list` – list active subscriptions
- `/info <coin>` – show current coin data
- `/chart <coin> [days]` – plot price history (alias `/charts`)
- `/news [coin]` – show latest news (uses subscriptions when omitted)
- `/trends` – show trending coins
- `/global` – show global market stats
- `/status` – display API status overview
- `/milestones [on|off]` – toggle milestone notifications (no args switch)
- `/settings [key value]` – show or change default settings (threshold,
  interval, milestones, liquidations, currency)

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
