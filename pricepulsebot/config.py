"""Configuration and helper utilities for PricePulseWatcherBot.

This module loads environment variables, configures logging and exposes
constants used across the bot.
"""

import logging
import os
import re
from logging.handlers import WatchedFileHandler

from dotenv import load_dotenv

load_dotenv()


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


def parse_timeframe(value: str) -> int:
    """Return seconds for a timeframe string.

    Pure numbers are interpreted as days while values with a trailing unit are
    delegated to :func:`parse_duration` to support hours or minutes.
    """

    if value.isdigit():
        return int(value) * 86400
    return parse_duration(value)


def format_interval(seconds: int) -> str:
    """Return a short string representation for a duration in seconds."""
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


DB_FILE = os.getenv("DB_PATH", "subs.db")
BOT_NAME = "PricePulseWatcherBot"
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.1"))
VOLUME_THRESHOLD = float(os.getenv("VOLUME_THRESHOLD", str(DEFAULT_THRESHOLD)))
DEFAULT_INTERVAL = parse_duration(os.getenv("DEFAULT_INTERVAL", "5m"))
PRICE_CHECK_INTERVAL = parse_duration(os.getenv("PRICE_CHECK_INTERVAL", "60s"))
CHART_CACHE_TTL = parse_duration(os.getenv("CHART_CACHE_TTL", "1h"))
DELETE_CHART_ON_RELOAD = os.getenv("DELETE_CHART_ON_RELOAD", "true").lower() == "true"
ENABLE_MILESTONE_ALERTS = os.getenv("ENABLE_MILESTONE_ALERTS", "true").lower() == "true"
ENABLE_VOLUME_ALERTS = os.getenv("ENABLE_VOLUME_ALERTS", "true").lower() == "true"
VS_CURRENCY = os.getenv("DEFAULT_VS_CURRENCY", "usd").lower()
DEFAULT_OVERVIEW = os.getenv("DEFAULT_OVERVIEW", "off").lower()

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE_URL = (
    os.getenv("COINGECKO_BASE_URL") or "https://api.coingecko.com/api/v3"
)
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
    _handlers.append(WatchedFileHandler(LOG_FILE))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    handlers=_handlers,
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
