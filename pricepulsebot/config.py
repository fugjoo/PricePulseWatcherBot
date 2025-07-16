import json
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

DB_FILE = os.getenv("DB_PATH", "subs.db")
BOT_NAME = "PricePulseWatcherBot"
DEFAULT_THRESHOLD = 0.1
DEFAULT_INTERVAL = 300
PRICE_CHECK_INTERVAL = 60

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE_URL = os.getenv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")
COINGECKO_HEADERS = (
    {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else None
)

COINS = ["bitcoin", "ethereum", "litecoin", "dogecoin"]
COIN_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "litecoin": "LTC",
    "dogecoin": "DOGE",
}
SYMBOL_TO_COIN = {v.lower(): k for k, v in COIN_SYMBOLS.items()}
TOP_COINS: list[str] = []

LOG_FILE = os.getenv("LOG_FILE")
_handlers = [logging.StreamHandler()]
if LOG_FILE:
    _handlers.append(logging.FileHandler(LOG_FILE))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    handlers=_handlers,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_duration(value: str) -> int:
    """Return seconds for a duration string like '15m' or '1h'."""
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"(\d+)([dhms])", value.lower())
    if not match:
        raise ValueError("invalid interval format")
    num, unit = match.groups()
    factor = {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
    return int(num) * factor


def format_interval(seconds: int) -> str:
    """Return a short string representation for a duration in seconds."""
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def load_config(path: str = "config.json") -> None:
    """Load defaults from a JSON config if present."""
    if not os.path.isfile(path):
        return
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to load config: %s", exc)
        return

    global DEFAULT_THRESHOLD, DEFAULT_INTERVAL, PRICE_CHECK_INTERVAL
    if "default_threshold" in data:
        try:
            DEFAULT_THRESHOLD = float(data["default_threshold"])
        except (TypeError, ValueError):
            logger.warning("invalid default_threshold in config")
    if "default_interval" in data:
        try:
            DEFAULT_INTERVAL = parse_duration(str(data["default_interval"]))
        except ValueError:
            logger.warning("invalid default_interval in config")
    if "price_check_interval" in data:
        try:
            PRICE_CHECK_INTERVAL = parse_duration(str(data["price_check_interval"]))
        except ValueError:
            logger.warning("invalid price_check_interval in config")
